import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from ..config import Config

class Retriever:
    def __init__(self):
        if not os.path.exists(Config.VECTOR_STORE_PATH) or not os.listdir(Config.VECTOR_STORE_PATH):
            raise FileNotFoundError(
                f"Vector store not found at {Config.VECTOR_STORE_PATH}. "
                "Please run 'python scripts/ingest_data.py' to create it."
            )
            
        print("Loading embedding model for retrieval...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=Config.EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'} # Use 'cuda' if GPU is available
        )
        
        print(f"Loading FAISS index from {Config.VECTOR_STORE_PATH}...")
        self.index = FAISS.load_local(
            Config.VECTOR_STORE_PATH, 
            self.embeddings,
            allow_dangerous_deserialization=True # Required for FAISS/pickle
        )
        print("FAISS index loaded.")

    def retrieve_docs(self, query: str, k: int = 4) -> list:
        """
        Retrieves the top-k relevant document chunks for a given query.
        
        :param query: The (English) query string.
        :param k: The number of documents to retrieve.
        :return: A list of LangChain Document objects.
        """
        print(f"Retrieving top {k} docs for query: '{query[:50]}...'")
        
        # Use the FAISS index to find similar documents
        # This returns a list of (Document, score) tuples
        # results = self.index.similarity_search_with_score(query, k=k)
        
        # For simplicity, just get the documents
        documents = self.index.similarity_search(query, k=k)
        
        return documents