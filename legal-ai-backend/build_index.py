# build_index.py (Run this once to create the FAISS index)
import os
import shutil
from langchain_community.vectorstores import FAISS
# Using the new standalone package for HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from typing import List

# Paths (relative to the root where this script runs)
# CORRECTED PATH based on your folder structure: 'app/data/policies'
POLICY_DIR = os.path.join(os.getcwd(), 'app', 'data', 'policies')
DB_PATH = os.path.join(os.getcwd(), 'policy_db') 
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_policies_from_text() -> List[Document]:
    """
    Reads policy files from the corrected directory and creates LangChain Document objects.
    Includes self-healing logic to ensure the directory exists and contains data.
    """
    docs = []
    
    # 1. Self-healing: Ensure the policy directory exists
    if not os.path.isdir(POLICY_DIR):
        os.makedirs(POLICY_DIR, exist_ok=True)
        print(f"Created missing policy directory at {POLICY_DIR}")

    # 2. Check for policies, and create a dummy one if none are found.
    policy_files = [f for f in os.listdir(POLICY_DIR) if f.endswith(".txt")]
    
    if not policy_files:
        dummy_policy_path = os.path.join(POLICY_DIR, "default_liability_policy.txt")
        # Ensure the content strongly violates a hypothetical standard policy to aid RAG testing
        dummy_content = (
            "Critical Policy Rule: The company's liability for ALL claims, regardless of cause, "
            "shall be strictly LIMITED to the total fees paid by the client in the preceding "
            "twelve (12) months. This is a crucial non-negotiable term for risk mitigation."
        )
        with open(dummy_policy_path, "w") as f:
            f.write(dummy_content)
        print(f"Created dummy policy file to ensure index build success: {dummy_policy_path}")
        # Re-scan to include the new file
        policy_files = [f for f in os.listdir(POLICY_DIR) if f.endswith(".txt")] 


    for filename in policy_files:
        file_path = os.path.join(POLICY_DIR, filename)
        policy_name = filename.replace('.txt', '').replace('_', ' ').title()

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()

        docs.append(Document(
            page_content=content,
            metadata={"policy_name": policy_name, "source": filename}
        ))
        
    return docs

def build_faiss_index():
    if os.path.exists(DB_PATH):
        # Remove old index to rebuild fresh
        shutil.rmtree(DB_PATH)
        print(f"Removed existing index at {DB_PATH}")

    docs = load_policies_from_text()
    if not docs:
        print("No policies loaded. Aborting index build.")
        return

    print(f"Building index from {len(docs)} documents...")
    
    # Initialize the specific HuggingFace embedding model (downloads if necessary)
    embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
    
    # Create the FAISS vector store
    vectorstore = FAISS.from_documents(docs, embeddings)
    
    # Save the index locally for persistence
    vectorstore.save_local(DB_PATH)
    print(f"FAISS index built and saved successfully at {DB_PATH}")

if __name__ == "__main__":
    build_faiss_index()