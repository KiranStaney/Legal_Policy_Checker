from fastapi import APIRouter
from pydantic import BaseModel

from app.models.request_models import AnalysisRequest, AnalyzeFullResponse
from app.services.agentic_ai import AgentManager

router = APIRouter(prefix="/analysis")


# ---------- Existing full pipeline endpoint (Day 1) ----------

@router.post("/analyze-full", response_model=AnalyzeFullResponse)
async def analyze_full_workflow(request: AnalysisRequest):
    """
    Backward-compatible: always do full RAG + analyze + rewrite + score.
    """
    manager = AgentManager()
    ctx = manager.run_full_pipeline(
        document_id=request.document_id,
        document_text=request.extracted_text,
    )

    return AnalyzeFullResponse(
        document_id=ctx.document_id,
        status="Analysis Complete",
        final_risk_score=ctx.risk_score,
        issues_detected=ctx.issues,
        policy_matches=ctx.policy_matches,
        rewrite_suggestions=ctx.rewrites,
        reflection=ctx.meta.get("reflection")
    )


# ---------- New agentic endpoint (Day 2 – Planner + tools) ----------

class AgentRunRequest(BaseModel):
    document_id: str
    extracted_text: str
    user_goal: str  # e.g. "Only risk score", "Full analysis with rewrites"


@router.post("/agent-run", response_model=AnalyzeFullResponse)
async def agent_run(request: AgentRunRequest):
    """
    Agentic endpoint: Planner chooses tools based on user_goal.
    """
    manager = AgentManager()
    ctx = manager.run_with_goal(
        user_goal=request.user_goal,
        document_id=request.document_id,
        document_text=request.extracted_text,
        max_steps=5
    )

    return AnalyzeFullResponse(
        document_id=ctx.document_id,
        status="Agent Run Complete",
        final_risk_score=ctx.risk_score,
        issues_detected=ctx.issues,
        policy_matches=ctx.policy_matches,
        rewrite_suggestions=ctx.rewrites,
        reflection=ctx.meta.get("reflection")   # 🔵 Day 4: expose reflection
    )
