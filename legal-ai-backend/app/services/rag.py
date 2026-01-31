import os
from functools import lru_cache
from typing import List, Dict, Any

# ---------- P-B Dependencies (Corrected Try/Except Block) ----------
try:
    from langchain_community.vectorstores import FAISS
    # Using the new package to satisfy the deprecation warning
    from langchain_huggingface import HuggingFaceEmbeddings

    # Define mock classes only if the import fails
    class VectorStoreMock:
        def __init__(self, policies): pass
        def similarity_search_with_score(self, query, k):
            # Mock high-risk matches
            return [
                (type('Doc', (object,), {'page_content': "Policy: Liability must be capped.", 'metadata': {'policy_name': 'Liability Limit'}}), 0.1),
                (type('Doc', (object,), {'page_content': "Policy: Notice must be 90 days.", 'metadata': {'policy_name': 'Termination Notice'}}), 0.2)
            ]

except ImportError as e:
    # This block executes if FAISS or HuggingFaceEmbeddings cannot be found.
    print(f"FATAL RAG ERROR: LangChain dependencies not found. Using simple RAG Mock. (Details: {e})")

    # Define mock classes here for when the import fails
    class FAISS:
        @staticmethod
        def load_local(*args, **kwargs): return VectorStoreMock({})
    class HuggingFaceEmbeddings:
        def __init__(self, model_name): pass
    class VectorStoreMock:
        def __init__(self, policies): pass
        def similarity_search_with_score(self, query, k):
            # Mock high-risk matches
            return [
                (type('Doc', (object,), {'page_content': "Policy: Liability must be capped.", 'metadata': {'policy_name': 'Liability Limit'}}), 0.1),
                (type('Doc', (object,), {'page_content': "Policy: Notice must be 90 days.", 'metadata': {'policy_name': 'Termination Notice'}}), 0.2)
            ]


# ---------- Paths & constants (Adjusted for main app structure) ----------
# policy_db must be at the root level alongside the 'app' folder
DB_PATH = os.path.join(os.getcwd(), "policy_db") 
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SCORE_THRESHOLD = 0.55 # Cosine distance: lower score means higher similarity (closer to 0)

# ---------- Cached loaders ----------
@lru_cache(maxsize=1)
def _get_embeddings():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


@lru_cache(maxsize=1)
def _get_vectorstore():
    # This assumes 'policy_db' has been created by Person B's separate script
    embeddings = _get_embeddings()
    return FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

# ---------- RAG Policy Checker Class (Main Integration Point) ----------
class RAGPolicyChecker:
    
    def __init__(self, score_threshold: float = SCORE_THRESHOLD):
        self.score_threshold = score_threshold
        # Initial check to load the vector store
        try:
            # We call the cached function here
            self._get_vectorstore()
        except Exception as e:
            print(f"RAG WARNING: Failed to load vector store from {DB_PATH}. Using mock data.")
            # Set a local method to ensure we don't crash on subsequent calls
            self._get_vectorstore = lambda: VectorStoreMock({})

    def _get_vectorstore(self):
        return _get_vectorstore()

    def check_compliance(self, document_text: str) -> List[str]:
        """
        Retrieves relevant policies based on similarity and returns them
        formatted for the Analyzer Agent (P-D).
        """
        print(f"[RAG] Running retrieval. Threshold: {self.score_threshold}")
        
        vectorstore = self._get_vectorstore()
        docs_and_scores = vectorstore.similarity_search_with_score(
            document_text,
            k=5
        )

        matched_policies = []
        
        for doc, score in docs_and_scores:
            # Lower score = better match
            if score <= self.score_threshold:
                policy_name = doc.metadata.get("policy_name", "Unknown Policy")
                
                # Format: Rule text and source for the LLM
                formatted_policy = (
                    f"Policy: {policy_name} | Confidence: {score:.3f}\n"
                    f"Rule: {doc.page_content.strip()}"
                )
                matched_policies.append(formatted_policy)

        print(f"[RAG] Found {len(matched_policies)} policies matching threshold.")
        return matched_policies


# Instantiate the main object Person A's route will import
rag_policy_checker = RAGPolicyChecker()