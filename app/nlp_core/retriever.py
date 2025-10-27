import logging
import pickle
from typing import List, Dict, Tuple
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from app.config import ModelConfig, VECTOR_STORE_DIR

logger = logging.getLogger(__name__)

class DocumentRetriever:
    """
    Handles semantic search over legal documents using FAISS and sentence embeddings.
    """
    
    def __init__(self, vector_store_path: Path = VECTOR_STORE_DIR):
        self.vector_store_path = vector_store_path
        self.index_file = vector_store_path / "faiss_index.bin"
        self.metadata_file = vector_store_path / "metadata.pkl"
        
        # Load embedding model
        logger.info(f"Loading embedding model: {ModelConfig.EMBEDDING_MODEL}")
        self.embedding_model = SentenceTransformer(ModelConfig.EMBEDDING_MODEL)
        self.embedding_model.to(ModelConfig.DEVICE)
        
        # Initialize FAISS index
        self.index = None
        self.documents = []
        self.metadata = []
        
        # Load existing index if available
        if self.index_file.exists() and self.metadata_file.exists():
            self.load_index()
        else:
            logger.warning("No existing vector store found. Please run ingest_data.py first.")
    
    def load_index(self):
        """Load FAISS index and metadata from disk."""
        try:
            logger.info("Loading FAISS index...")
            self.index = faiss.read_index(str(self.index_file))
            
            with open(self.metadata_file, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.metadata = data['metadata']
            
            logger.info(f"Loaded index with {len(self.documents)} documents")
        except Exception as e:
            logger.error(f"Error loading index: {e}")
            raise
    
    def save_index(self):
        """Save FAISS index and metadata to disk."""
        try:
            logger.info("Saving FAISS index...")
            faiss.write_index(self.index, str(self.index_file))
            
            with open(self.metadata_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'metadata': self.metadata
                }, f)
            
            logger.info("Index saved successfully")
        except Exception as e:
            logger.error(f"Error saving index: {e}")
            raise
    
    def create_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Create embeddings for a list of texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Numpy array of embeddings
        """
        logger.info(f"Creating embeddings for {len(texts)} texts...")
        embeddings = self.embedding_model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings.astype('float32')
    
    def build_index(self, documents: List[str], metadata: List[Dict]):
        """
        Build FAISS index from documents.
        
        Args:
            documents: List of document chunks
            metadata: List of metadata dictionaries for each chunk
        """
        logger.info(f"Building index for {len(documents)} documents...")
        
        self.documents = documents
        self.metadata = metadata
        
        # Create embeddings
        embeddings = self.create_embeddings(documents)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings)
        
        logger.info(f"Index built with {self.index.ntotal} vectors")
        
        # Save to disk
        self.save_index()
    
    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve most relevant documents for a query.
        
        Args:
            query: Search query in English
            top_k: Number of documents to retrieve
            
        Returns:
            List of dictionaries containing retrieved documents and metadata
        """
        if self.index is None:
            raise ValueError("Index not loaded. Please run ingest_data.py first.")
        
        if top_k is None:
            top_k = ModelConfig.TOP_K_DOCUMENTS
        
        # Create query embedding
        query_embedding = self.create_embeddings([query])
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Prepare results
        results = []
        for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
            if idx < len(self.documents):  # Valid index
                results.append({
                    'rank': i + 1,
                    'document': self.documents[idx],
                    'metadata': self.metadata[idx],
                    'distance': float(dist),
                    'relevance_score': float(1 / (1 + dist))  # Convert distance to relevance
                })
        
        logger.info(f"Retrieved {len(results)} documents for query")
        return results
    
    def get_context_for_generation(self, retrieved_docs: List[Dict], max_length: int = 2048) -> str:
        """
        Prepare context string from retrieved documents for generation.
        
        Args:
            retrieved_docs: List of retrieved documents
            max_length: Maximum character length of context
            
        Returns:
            Formatted context string
        """
        context_parts = []
        current_length = 0
        
        for doc in retrieved_docs:
            doc_text = doc['document']
            metadata = doc['metadata']
            
            # Format with source information
            formatted_doc = f"[Source: {metadata.get('source', 'Unknown')}]\n{doc_text}\n"
            
            if current_length + len(formatted_doc) <= max_length:
                context_parts.append(formatted_doc)
                current_length += len(formatted_doc)
            else:
                break
        
        context = "\n---\n".join(context_parts)
        return context