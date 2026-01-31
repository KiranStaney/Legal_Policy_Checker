import os
import hashlib
from typing import List, Dict, Any, Union
from dotenv import load_dotenv

POLICY_DIR = os.path.join(os.getcwd(), 'app', 'data', 'data_policies')
CHROMA_DB_PATH = os.path.join(os.getcwd(), 'policy_db_chroma')

class Document:
    """Mock/Fallback for LangChain Document class."""
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata if metadata is not None else {}

class MockVectorDB:
    """A mock implementation of a vector store for when dependencies fail."""
    def similarity_search(self, question, k): 
        print("MOCK: Chroma DB not initialized. Returning mock policy data.")
        return [
            Document(page_content="Mock Policy: Liability must be capped at 12 months fees.", metadata={"policy_name": "Liability Limit"}),
            Document(page_content="Mock Policy: Termination requires 90 days notice.", metadata={"policy_name": "Termination Notice"})
        ]

try:
    from langchain_community.vectorstores import Chroma
    from langchain_text_splitters import RecursiveCharacterTextSplitter 
    from langchain_core.documents import Document as RealDocument 
    Document = RealDocument 
except ImportError:
    print("WARNING: LangChain/Chroma dependencies not found. Using Mock KnowledgeBase.")
    class Chroma:
        @classmethod
        def from_documents(cls, *args, **kwargs): return MockVectorDB()
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size, chunk_overlap): 
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def create_documents(self, texts, metadatas=None): 
            return [Document(page_content=t, metadata=metadatas[i] if metadatas and len(metadatas) > i else {}) for i, t in enumerate(texts)]

try:
    from google import genai 
    from app.services.llm import llm_client 
    if 'llm_client' not in locals():
         class LLMClientMock:
             def get_embeddings(self): return None 
         llm_client = LLMClientMock() 
         print("WARNING: LLM Client dependency needed for embeddings not found. Using simple mock.")

except ImportError:
    class genai:
        @staticmethod
        def Client(): return None
    print("WARNING: Google GenAI SDK not installed. KnowledgeBase won't use it directly.")


class KnowledgeBase:
    """
    Manages the policy documents, vector indexing, and RAG retrieval.
    This class can use ChromaDB for persistence or fall back to a mock.
    """
    def __init__(self):
        load_dotenv()
        self.CHROMA_DB_PATH = CHROMA_DB_PATH
        self.POLICY_DIR = POLICY_DIR
        self.db = None
        self.embeddings = self._load_embeddings()

    def _load_embeddings(self):
        """Loads the embedding model from the LLM client."""
        try:
            return llm_client.get_embeddings() 
        except Exception as e:
            print(f"ERROR: Could not load embeddings from LLM client. Using mock. {e}")
            return None 

    def load_or_build_index(self):
        if self.embeddings is None:
             print("ERROR: Embeddings failed to load. KnowledgeBase is in a non-functional state.")
             self.db = MockVectorDB() 
             return

        try:
            if os.path.exists(self.CHROMA_DB_PATH):
                print(f"[KB] Loading existing Chroma DB from {self.CHROMA_DB_PATH}...")
                self.db = Chroma(
                    persist_directory=self.CHROMA_DB_PATH, 
                    embedding_function=self.embeddings
                )
                print("[KB] Chroma DB loaded successfully.")
                return

            print("[KB] Existing index not found. Initiating build process.")
            self._build_index()

        except Exception as e:
            print(f"FATAL ERROR: Failed to load or build Chroma index. Using mock. {e}")
            self.db = MockVectorDB()


    def _load_policies_from_text(self) -> List[Document]:
        docs = []
        if not os.path.isdir(self.POLICY_DIR):
            print(f"Error: Policy directory not found at {self.POLICY_DIR}. Cannot build index.")
            return []

        for filename in os.listdir(self.POLICY_DIR):
            if filename.endswith(".txt"):
                file_path = os.path.join(self.POLICY_DIR, filename)
                policy_name = filename.replace('.txt', '').replace('_', ' ').title()

                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()

                docs.append(Document(
                    page_content=content,
                    metadata={"policy_name": policy_name, "source": filename}
                ))
        return docs

    def _build_index(self):
        raw_documents = self._load_policies_from_text()
        if not raw_documents:
            print("No policies loaded. Aborting index build.")
            return
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        
        documents_for_chroma = []
        for doc in raw_documents:
             chunks = splitter.create_documents([doc.page_content], metadatas=[doc.metadata])
             documents_for_chroma.extend(chunks)

        if not documents_for_chroma:
            print("WARNING: No chunks created after splitting. Aborting KB build.")
            return

        # 2. Build and persist vector DB
        print(f"[KB] Building Chroma DB from {len(documents_for_chroma)} chunks and persisting to {self.CHROMA_DB_PATH}...")
        self.db = Chroma.from_documents(
            documents_for_chroma,
            embedding=self.embeddings,
            persist_directory=self.CHROMA_DB_PATH
        )
        self.db.persist()
        print("[KB] Chroma DB build and persistence complete.")


    def query(self, question: str) -> List[Document]:
        if not self.db:
            print("ERROR: Knowledge base not initialized. Returning mock data.")
            return MockVectorDB().similarity_search(question, k=3)

        # Chroma uses similarity_search internally by default with L2 (Euclidean) distance
        # NOTE: Lower score = better match for L2 (Euclidean distance)
        return self.db.similarity_search(question, k=5)

# NOTE: The KnowledgeBase object needs to be instantiated once the policies are loaded 
# (e.g., in the main application startup after `build_index.py` is run).
knowledge_base_client = KnowledgeBase()
# It is crucial to call load_or_build_index() somewhere in the application startup.
# Example: knowledge_base_client.load_or_build_index()