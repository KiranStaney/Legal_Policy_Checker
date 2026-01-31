import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable

from pydantic import BaseModel, Field

from app.models.request_models import ClauseIssue, RewriteSuggestion
from app.services.llm import llm_client
from app.services.rag import rag_policy_checker


# ---------- SHARED CONTEXT + BASE AGENT ----------

@dataclass
class AgentContext:
    """
    Shared state that flows through all agents.
    """
    document_id: Optional[str] = None
    raw_text: Optional[str] = None

    # Outputs from different stages
    policy_matches: List[str] = field(default_factory=list)
    issues: List[ClauseIssue] = field(default_factory=list)
    rewrites: List[RewriteSuggestion] = field(default_factory=list)
    risk_score: Optional[float] = None

    # Extra metadata + debug trace
    meta: Dict[str, Any] = field(default_factory=dict)
    trace: List[Dict[str, Any]] = field(default_factory=list)


class AnalyzerOutput(BaseModel):
    issues: List[Dict[str, Any]] = Field(
        description="Detected contract issues with risk scoring."
    )


class BaseAgent:
    """
    Simple synchronous base agent. Each agent reads + updates AgentContext.
    """
    name: str = "BaseAgent"

    def run(self, ctx: AgentContext) -> AgentContext:
        raise NotImplementedError


# ---------- TOOL REGISTRY (for tool-calling) ----------

ToolFn = Callable[[AgentContext], AgentContext]


class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, ToolFn] = {}

    def register(self, name: str, fn: ToolFn):
        print(f"[ToolRegistry] Registering tool: {name}")
        self.tools[name] = fn

    def get(self, name: str) -> ToolFn:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not registered.")
        return self.tools[name]

    def list_tools(self) -> List[str]:
        return list(self.tools.keys())

    def execute(self, name: str, ctx: AgentContext) -> AgentContext:
        """
        Helper to execute a tool by name on the given context.
        """
        tool_fn = self.get(name)
        return tool_fn(ctx)


# ---------- ANALYZER AGENT ----------
class AnalyzerAgent(BaseAgent):
    def __init__(self):
        self.client = llm_client
        self.name = "AnalyzerAgent"

    def run(self, ctx: AgentContext) -> AgentContext:
        """
        Wraps the old analyze(document_text, policy_matches) into
        an agent-style method using AgentContext.
        """
        document_text = ctx.raw_text or ""
        policy_matches = ctx.policy_matches or []

        issues = self._analyze(document_text, policy_matches)
        ctx.issues = issues

        ctx.trace.append({
            "agent": self.name,
            "action": "analyze",
            "issues_count": len(issues),
        })
        return ctx

    # --- MAIN ANALYZE LOGIC ---
    def _analyze(self, document_text: str, policy_matches: List[str]) -> List[ClauseIssue]:
        print("Analyzer: Starting structured LLM analysis...")

        policy_list = "\n".join(policy_matches) if policy_matches else "NO EXPLICIT POLICIES MATCHED."

        prompt = f"""
You are a Legal Risk Analyzer.

Compare the DOCUMENT TEXT against the POLICIES below and detect contract risks.

For each issue you find, return an object with:
- "clause_title": short title of the issue
- "issue_description": explanation INCLUDING the original risky wording
- "risk_score_impact": number between 0.0 (low) and 1.0 (very high)

Return a SINGLE JSON object ONLY in this exact format:

{{
  "issues": [
    {{
      "clause_title": "...",
      "issue_description": "...",
      "risk_score_impact": 0.7
    }},
    ...
  ]
}}

Do NOT add any extra text before or after the JSON.
Your output MUST be valid JSON.

DOCUMENT:
{document_text[:3500]}

POLICIES:
{policy_list}
"""

        raw_response = ""
        try:
            raw_response = self.client.get_text_response(prompt).strip()
            print("Analyzer raw response:", raw_response[:200], "...")

            # If LLM itself failed (quota etc.), skip JSON parsing and go to heuristic
            if raw_response.startswith("LLM Text Error:"):
                print("Analyzer: LLM error detected, using heuristic analyzer.")
                return self._heuristic_analyze(document_text)

            # strip ```json ``` if model wraps output
            if raw_response.startswith("```"):
                raw_response = raw_response.strip("`")
                if raw_response.lower().startswith("json"):
                    raw_response = raw_response[4:].strip()

            data = json.loads(raw_response)

            issues_raw = data.get("issues", []) or []

            issues: List[ClauseIssue] = []
            for issue in issues_raw:
                title = issue.get("clause_title", "Untitled Issue")
                desc = issue.get("issue_description", "")
                risk = issue.get("risk_score_impact", 0.5)
                try:
                    risk = float(risk)
                except Exception:
                    risk = 0.5
                risk = max(0.0, min(1.0, risk))

                issues.append(
                    ClauseIssue(
                        clause_title=title,
                        issue_description=desc,
                        risk_score_impact=risk,
                    )
                )

            print(f"Analyzer: {len(issues)} issues detected.")

            if not issues:
                # Nothing from LLM → still fall back to heuristic
                return self._heuristic_analyze(document_text)

            return issues

        except Exception as e:
            print("Analyzer failed:", e)
            print("Raw LLM response was:", raw_response)
            # LLM or JSON failed → use heuristic
            return self._heuristic_analyze(document_text)

    # --- HEURISTIC ANALYZER (NO LLM) ---
    def _heuristic_analyze(self, document_text: str) -> List[ClauseIssue]:
        """
        Simple rule-based analyzer so that the system still works
        even when the LLM is unavailable or fails.
        """
        print("Analyzer: Running heuristic fallback analysis...")
        text_lower = document_text.lower()
        issues: List[ClauseIssue] = []

        # Rule 1: Unlimited liability
        if "liability" in text_lower and "unlimited" in text_lower:
            issues.append(
                ClauseIssue(
                    clause_title="Unlimited Provider Liability",
                    issue_description=(
                        "The clause states that the Provider's liability is unlimited, "
                        "which exposes the Provider to uncapped financial risk. "
                        "Standard practice is to cap liability to a multiple of fees or insurance coverage."
                    ),
                    risk_score_impact=1.0,
                )
            )

        # Rule 2: Very short termination notice (15 days)
        if ("terminate" in text_lower or "termination" in text_lower) and "15 days" in text_lower:
            issues.append(
                ClauseIssue(
                    clause_title="Short Client Termination Notice",
                    issue_description=(
                        "The clause allows the client to terminate the agreement with only 15 days' notice. "
                        "This short notice period can cause severe operational and revenue planning challenges "
                        "for the Provider, and a longer notice period (60–90 days) is generally preferred."
                    ),
                    risk_score_impact=0.8,
                )
            )

        # If still nothing, create a softer generic issue
        if not issues:
            issues.append(
                ClauseIssue(
                    clause_title="Potential Unspecified Risk",
                    issue_description=(
                        "No specific issues were detected by the heuristic analyzer, "
                        "but the document could still contain legal or commercial risks that require manual review."
                    ),
                    risk_score_impact=0.3,
                )
            )

        print(f"Heuristic Analyzer: {len(issues)} issues detected.")
        return issues


# ---------- REWRITE AGENT ----------
class RewriteAgent(BaseAgent):
    def __init__(self):
        self.client = llm_client
        self.name = "RewriteAgent"

    def run(self, ctx: AgentContext) -> AgentContext:
        """
        Wraps old rewrite_issues(document_text, issues) into agent-style run().
        """
        document_text = ctx.raw_text or ""
        issues = ctx.issues or []

        if not issues:
            ctx.trace.append({
                "agent": self.name,
                "action": "rewrite_skipped",
                "reason": "No issues found to rewrite."
            })
            return ctx

        suggestions = self.rewrite_issues(document_text, issues)
        ctx.rewrites = suggestions

        ctx.trace.append({
            "agent": self.name,
            "action": "rewrite",
            "rewrite_count": len(suggestions),
        })
        return ctx

    def rewrite_issues(self, document_text: str, issues: List[ClauseIssue]) -> List[RewriteSuggestion]:
        print(f"Rewrite Agent: generating rewrites for {len(issues)} issues.")
        suggestions: List[RewriteSuggestion] = []

        for issue in issues:
            prompt = f"""
You are a senior contract lawyer.

You will receive 1 ISSUE from a contract review.

Rewrite the problematic clause into a safer, policy-compliant clause.

STRICT FORMAT:
- FIRST LINE: only the improved clause text. Do NOT add labels, bullets, or commentary.
- REMAINING LINES: short explanation why this rewrite reduces legal risk.

Do NOT write phrases like "Here is the rewritten clause" or "Reasoning:".
Just output the clause on the first line, then the explanation in the next lines.

ISSUE TITLE: {issue.clause_title}
ISSUE DESCRIPTION: {issue.issue_description}
"""

            try:
                response = self.client.get_text_response(prompt)

                # If Gemini returns a quota/error string, treat as failure
                if response.startswith("LLM Text Error:"):
                    raise RuntimeError(response)

                lines = [l.strip() for l in response.split("\n") if l.strip()]

                rewrite = lines[0] if lines else "Rewrite error"
                reasoning = "\n".join(lines[1:]) if len(lines) > 1 else "No reasoning provided."

                suggestions.append(
                    RewriteSuggestion(
                        original_text="Extracted from issue_description",
                        suggested_rewrite=rewrite,
                        reasoning=reasoning,
                    )
                )
            except Exception as e:
                print("RewriteAgent failed:", e)
                # LLM is not available → fall back to heuristic rewrite
                suggestions.append(self._heuristic_rewrite(issue))

        return suggestions

    # --- HEURISTIC REWRITE (NO LLM) ---
    def _heuristic_rewrite(self, issue: ClauseIssue) -> RewriteSuggestion:
        """
        Simple rule-based rewrite when LLM is unavailable.
        """
        title_lower = issue.clause_title.lower()

        # Case 1: Unlimited liability
        if "liability" in title_lower and "unlimited" in title_lower:
            clause = (
                "The Provider's aggregate liability under or in connection with this Agreement, "
                "whether arising in contract, tort (including negligence) or otherwise, "
                "shall not exceed the total fees paid by the Client to the Provider "
                "under this Agreement in the twelve (12) months preceding the event giving rise to the claim."
            )
            reasoning = (
                "This rewrite introduces a clear financial cap on the Provider's liability, "
                "reducing the risk of catastrophic, uncapped losses and aligning the exposure "
                "with the commercial value of the contract."
            )
            return RewriteSuggestion(
                original_text="Clause with unlimited liability",
                suggested_rewrite=clause,
                reasoning=reasoning,
            )

        # Case 2: Short termination notice
        if "termination" in title_lower or "terminate" in title_lower:
            clause = (
                "The Client may terminate this Agreement for convenience by providing not less than "
                "sixty (60) days' prior written notice to the Provider."
            )
            reasoning = (
                "Extending the notice period to 60 days provides the Provider with sufficient time "
                "to plan resource reallocation and manage revenue impact, reducing operational risk."
            )
            return RewriteSuggestion(
                original_text="Clause with 15-day termination notice",
                suggested_rewrite=clause,
                reasoning=reasoning,
            )

        # Generic fallback
        generic_clause = (
            "The parties agree that any termination of this Agreement shall be subject to a reasonable "
            "prior written notice period and shall not relieve either party of obligations accrued "
            "prior to the effective date of termination."
        )
        generic_reasoning = (
            "This generic rewrite adds a reasonable notice period requirement and preserves obligations "
            "that arose before termination, reducing abrupt termination risk."
        )
        return RewriteSuggestion(
            original_text="Issue detected but no specific heuristic rewrite",
            suggested_rewrite=generic_clause,
            reasoning=generic_reasoning,
        )


# ---------- SCORING AGENT ----------

class ScoringAgent(BaseAgent):
    def __init__(self):
        self.name = "ScoringAgent"

    def run(self, ctx: AgentContext) -> AgentContext:
        score = self.calculate_score(ctx.issues or [])
        ctx.risk_score = score

        ctx.trace.append({
            "agent": self.name,
            "action": "score",
            "risk_score": score,
        })
        return ctx

    def calculate_score(self, issues: List[ClauseIssue]) -> float:
        print("Scoring Agent: calculating...")

        if not issues:
            return 100.0

        total = sum(issue.risk_score_impact for issue in issues)

        # each issue's impact * 20 => max 100
        deduction = min(100.0, total * 20)

        final = max(0, round(100 - deduction, 1))

        print("Final Risk Score =", final)
        return final


# ---------- REFLECTION / SAFETY AGENT (Day 4) ----------

class ReflectionAgent(BaseAgent):
    """
    Day 4: Self-reflection + safety layer.

    - Reviews issues, rewrites, and final risk score.
    - Checks if rewrites look safer than original issues.
    - Adds an overall assessment + safety notes into ctx.meta["reflection"].
    """

    def __init__(self):
        self.client = llm_client
        self.name = "ReflectionAgent"

    def run(self, ctx: AgentContext) -> AgentContext:
        # Prepare compact payload for the LLM
        issues_payload = [
            {
                "title": i.clause_title,
                "description": i.issue_description,
                "risk": i.risk_score_impact,
            }
            for i in ctx.issues
        ]

        rewrites_payload = [
            {
                "suggested_rewrite": r.suggested_rewrite,
                "reasoning": r.reasoning,
            }
            for r in ctx.rewrites
        ]

        prompt = f"""
You are a senior legal QA reviewer and safety checker.

You will be given:
- A list of detected contract issues.
- A list of suggested rewrites for those issues.
- A final risk score.

Your job:
1. Check whether the rewrites are generally safer and more compliant than the original issues.
2. Flag any rewrite that looks unsafe, incomplete, or worse than the original clause.
3. Provide a short, high-level explanation of overall contract risk and safety.

ISSUES:
{json.dumps(issues_payload, ensure_ascii=False)}

REWRITES:
{json.dumps(rewrites_payload, ensure_ascii=False)}

FINAL_RISK_SCORE: {ctx.risk_score}
"""

        # ✅ Use Gemini structured output instead of manual JSON parsing
        schema = {
            "type": "object",
            "properties": {
                "overall_assessment": {"type": "string"},
                "unsafe_rewrites": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "reason": {"type": "string"}
                        },
                        "required": ["index", "reason"]
                    }
                },
                "safety_notes": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["overall_assessment", "unsafe_rewrites", "safety_notes"]
        }

        try:
            data = self.client.get_structured_response(prompt, schema)
            print("ReflectionAgent structured response:", data)
        except Exception as e:
            print("ReflectionAgent failed (structured):", e)
            # Smart local reflection fallback
            data = self._local_reflection(ctx)

        # Attach reflection data to context metadata
        ctx.meta["reflection"] = data

        ctx.trace.append({
            "agent": self.name,
            "action": "reflect",
            "summary": data.get("overall_assessment", ""),
            "unsafe_rewrites_count": len(data.get("unsafe_rewrites", [])),
        })

        return ctx

    def _local_reflection(self, ctx: AgentContext) -> Dict[str, Any]:
        """
        Local, non-LLM reflection based on score + counts.
        """
        score = ctx.risk_score if ctx.risk_score is not None else 100.0
        issue_count = len(ctx.issues)
        rewrite_count = len(ctx.rewrites)

        if score >= 80:
            risk_level = "low to moderate"
        elif score >= 50:
            risk_level = "moderate"
        else:
            risk_level = "high"

        overall = (
            f"The document has {issue_count} detected issue(s) with an overall risk score of {score}, "
            f"indicating {risk_level} risk. "
        )
        if rewrite_count > 0:
            overall += (
                f"{rewrite_count} rewrite suggestion(s) have been generated to reduce risk, "
                "but they should be reviewed by a human lawyer before adoption."
            )
        else:
            overall += (
                "No automated rewrite suggestions are currently available, likely due to "
                "LLM or quota limitations. Manual review is recommended."
            )

        safety_notes = [
            "Use the risk score and issues list as the primary signal.",
            "Have a qualified lawyer review high-impact clauses such as liability and termination.",
        ]

        return {
            "overall_assessment": overall,
            "unsafe_rewrites": [],
            "safety_notes": safety_notes,
        }



# ---------- PLANNER AGENT (Day 2 + Day 3 brain) ----------

class PlannerAgent(BaseAgent):
    def __init__(self, tools: ToolRegistry):
        self.name = "PlannerAgent"
        self.tools = tools
        self.client = llm_client

    def run(self, ctx: AgentContext) -> AgentContext:
        """
        Day 2: Decide full sequence once and execute.
        (Kept for backward compatibility, though agent loop is preferred.)
        """
        user_goal = ctx.meta.get("user_goal", "Do a full risk analysis and rewrite risky clauses.")

        tool_list = self.tools.list_tools()

        prompt = f"""
You are the Planner for a legal contract assistant.

Available tools (functions you can call), in order-sensitive JSON array:

{tool_list}

Each tool does:
- "rag_check": Match document against standard policies and store matched policies.
- "analyze_document": Detect issues using LLM and policy matches.
- "rewrite_issues": Rewrite risky clauses discovered by analysis.
- "compute_risk_score": Compute overall risk score based on issues.

User goal: "{user_goal}"

Rules:
- If the user asks only for summary or explanation, you may skip "rewrite_issues".
- If the user asks only for risk score, you may call "rag_check", "analyze_document", "compute_risk_score" only.
- For a full review, call all tools in a logical order.

Return ONLY a JSON array of tool names, like:
["rag_check", "analyze_document", "rewrite_issues", "compute_risk_score"]
"""

        raw = self.client.get_text_response(prompt).strip()
        print("[PlannerAgent] Raw plan:", raw)

        # Clean ```json ... ``` if present
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].strip()

        try:
            sequence = json.loads(raw)
        except Exception as e:
            print("[PlannerAgent] Failed to parse JSON plan:", e)
            print("[PlannerAgent] Fallback to default full pipeline.")
            sequence = ["rag_check", "analyze_document", "rewrite_issues", "compute_risk_score"]

        # Execute tools in order
        for tool_name in sequence:
            try:
                tool_fn = self.tools.get(tool_name)
            except KeyError as e:
                print("[PlannerAgent] Unknown tool in plan:", e)
                continue

            print(f"[PlannerAgent] Executing tool: {tool_name}")
            ctx = tool_fn(ctx)

        ctx.trace.append({
            "agent": self.name,
            "action": "plan_and_execute",
            "plan": sequence,
        })
        return ctx

    def decide_next_tool(self, ctx: AgentContext) -> str:
        """
        Day 3: Planner decides the NEXT best tool based on current context + user_goal.
        Used by AgentManager.run_agent_loop().
        """
        goal = ctx.meta.get("user_goal", "")

        prompt = f"""
You are a contract-analysis planner agent.

USER GOAL:
{goal}

CURRENT CONTEXT SUMMARY:
- Issues detected: {len(ctx.issues)}
- Rewrites generated: {len(ctx.rewrites)}
- Risk score: {ctx.risk_score}

Choose the NEXT best tool to run.

Allowed tools:
- rag_check
- analyze_document
- rewrite_issues
- compute_risk_score
- stop

Rules:
- If no issues are detected yet and the document has not been analyzed, call "analyze_document".
- If the document has not been checked against policies and goal involves compliance, call "rag_check" first.
- If issues exist but rewrites are empty AND the goal mentions rewriting, call "rewrite_issues".
- If risk_score is None and issues exist, call "compute_risk_score".
- If the goal is only risk scoring, you may skip "rewrite_issues".
- When all needed work for the goal is complete, return "stop".

Respond with ONLY one of:
"rag_check", "analyze_document", "rewrite_issues", "compute_risk_score", or "stop".
"""

        try:
            response = self.client.get_text_response(prompt).strip().lower()
            response = response.replace("```", "").replace(".", "").strip()
            print("[PlannerAgent] decide_next_tool raw:", response)

            # If LLM failed (quota etc.), just stop and let fallback handle scoring
            if response.startswith("llm text error:"):
                print("[PlannerAgent] LLM error in decide_next_tool, stopping loop.")
                return "stop"

            if "rag_check" in response:
                return "rag_check"
            if "analyze_document" in response:
                return "analyze_document"
            if "rewrite_issues" in response:
                return "rewrite_issues"
            if "compute_risk_score" in response:
                return "compute_risk_score"
            if "stop" in response:
                return "stop"

            return "stop"
        except Exception as e:
            print("[PlannerAgent] decide_next_tool error:", e)
            return "stop"


# ---------- AGENT MANAGER ----------

class AgentManager:
    """
    Agentic manager.

    Day 1: fixed pipeline RAG → Analyzer → Rewrite → Score.
    Day 2: tools + planner (batch plan).
    Day 3: full agent loop with step-wise planning.
    """

    def __init__(self):
        self.analyzer = AnalyzerAgent()
        self.rewriter = RewriteAgent()
        self.scorer = ScoringAgent()
        self.reflector = ReflectionAgent()   # 🔵 Day 4: new agent
        # tools setup
        self.tools = ToolRegistry()
        self.tools.register("rag_check", self._tool_rag_check)
        self.tools.register("analyze_document", self.analyzer.run)
        self.tools.register("rewrite_issues", self.rewriter.run)
        self.tools.register("compute_risk_score", self.scorer.run)

        self.planner = PlannerAgent(self.tools)

    # ---- individual tools ----

    def _tool_rag_check(self, ctx: AgentContext) -> AgentContext:
        if not ctx.raw_text:
            return ctx
        print("[Tool] rag_check running...")
        ctx.policy_matches = rag_policy_checker.check_compliance(ctx.raw_text)
        ctx.trace.append({
            "agent": "RAGPolicyChecker",
            "action": "check_compliance",
            "match_count": len(ctx.policy_matches),
        })
        return ctx

    # ---- Day 1 helper: fixed pipeline ----

    def run_full_pipeline(self, document_id: str, document_text: str) -> AgentContext:
        """
        Keeps backward-compatible full pipeline: RAG -> Analyzer -> Rewrite -> Score
        """
        ctx = AgentContext(
            document_id=document_id,
            raw_text=document_text,
        )

        ctx = self._tool_rag_check(ctx)
        ctx = self.analyzer.run(ctx)
        ctx = self.rewriter.run(ctx)
        ctx = self.scorer.run(ctx)
        # 🔵 Day 4: final reflection + safety check
        ctx = self.reflector.run(ctx)
        return ctx

    # ---- Day 2: planner-based pipeline (one-shot plan) ----

        # ---- Day 2/3: planner-based pipeline, now using the full agent loop ----

    def run_with_goal(
        self,
        user_goal: str,
        document_id: str,
        document_text: str,
        max_steps: int = 5,
    ) -> AgentContext:
        """
        Wrapper that uses the full agent loop for a given goal.
        This reuses run_agent_loop so you also get reflection (Day 4).
        """
        return self.run_agent_loop(
            user_goal=user_goal,
            document_id=document_id,
            document_text=document_text,
            max_steps=max_steps,
        )

       # ---- Day 3: full agent loop with step-wise planning ----

    def run_agent_loop(
        self,
        user_goal: str,
        document_id: str,
        document_text: str,
        max_steps: int = 5
    ) -> AgentContext:
        """
        Day 3: Full multi-step agent loop.
        Planner evaluates context after each tool call and decides next step.
        Stops when goal seems satisfied or max steps reached.

        If the planner fails and no tools are executed at all,
        we FALL BACK to the fixed full pipeline (RAG -> Analyzer -> Rewrite -> Score -> Reflection).
        """
        ctx = AgentContext(
            document_id=document_id,
            raw_text=document_text
        )

        ctx.meta["user_goal"] = user_goal

        executed_any_tool = False  # ✅ track whether *any* tool actually ran

        for step in range(max_steps):
            # 1) Planner decides next tool
            next_tool = self.planner.decide_next_tool(ctx)

            ctx.trace.append({
                "step": step,
                "agent": "PlannerAgent",
                "decided_tool": next_tool
            })

            # 2) Stop condition
            if next_tool == "stop" or next_tool is None:
                print("[AgentManager] Agent loop stopping at step", step)
                break

            # 3) Execute selected tool
            ctx = self.tools.execute(next_tool, ctx)
            executed_any_tool = True  # ✅ at least one tool ran

            # 4) Save intermediate output (for debugging)
            ctx.trace.append({
                "step_completed": step,
                "tool_executed": next_tool,
                "issues": len(ctx.issues),
                "rewrites": len(ctx.rewrites),
                "risk_score": ctx.risk_score
            })

        # ✅ If planner never ran any tool (likely LLM error), use the old fixed pipeline
        if not executed_any_tool:
            print("[AgentManager] Planner executed no tools. Falling back to fixed full pipeline.")
            return self.run_full_pipeline(document_id=document_id, document_text=document_text)

        # ✅ Fallback: if planner skipped scoring, compute it once locally
        if ctx.risk_score is None:
            print("[AgentManager] Fallback scoring because planner stopped early or skipped scoring.")
            ctx = self.scorer.run(ctx)

        # 🔵 Day 4: after all steps, run reflection/safety layer
        ctx = self.reflector.run(ctx)

        return ctx

