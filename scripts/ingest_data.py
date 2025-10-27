"""
Script to ingest legal documents and build the vector store.
Place your PDF/TXT documents in data/legal_docs/ before running this script.

Usage:
    python scripts/ingest_data.py
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.nlp_core.retriever import DocumentRetriever
from app.config import LEGAL_DOCS_DIR, ModelConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Process legal documents into chunks for indexing."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=ModelConfig.CHUNK_SIZE,
            chunk_overlap=ModelConfig.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
    
    def read_pdf(self, file_path: Path) -> str:
        """Extract text from PDF file."""
        try:
            reader = PdfReader(str(file_path))
            text = ""
            for page_num, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {page_num + 1}]\n{page_text}\n"
            return text
        except Exception as e:
            logger.error(f"Error reading PDF {file_path}: {e}")
            return ""
    
    def read_txt(self, file_path: Path) -> str:
        """Read text from TXT file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error reading TXT {file_path}: {e}")
            return ""
    
    def process_document(self, file_path: Path) -> List[Dict]:
        """
        Process a single document into chunks with metadata.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of dictionaries containing chunks and metadata
        """
        logger.info(f"Processing document: {file_path.name}")
        
        # Read document based on file type
        if file_path.suffix.lower() == '.pdf':
            text = self.read_pdf(file_path)
        elif file_path.suffix.lower() == '.txt':
            text = self.read_txt(file_path)
        else:
            logger.warning(f"Unsupported file type: {file_path.suffix}")
            return []
        
        if not text.strip():
            logger.warning(f"No text extracted from {file_path.name}")
            return []
        
        # Split into chunks
        chunks = self.text_splitter.split_text(text)
        
        # Create chunk documents with metadata
        chunk_docs = []
        for i, chunk in enumerate(chunks):
            chunk_docs.append({
                'text': chunk,
                'metadata': {
                    'source': file_path.name,
                    'chunk_id': i,
                    'total_chunks': len(chunks),
                    'file_path': str(file_path)
                }
            })
        
        logger.info(f"Created {len(chunk_docs)} chunks from {file_path.name}")
        return chunk_docs
    
    def process_all_documents(self, docs_dir: Path) -> tuple[List[str], List[Dict]]:
        """
        Process all documents in a directory.
        
        Args:
            docs_dir: Directory containing legal documents
            
        Returns:
            Tuple of (document_texts, metadata_list)
        """
        all_chunks = []
        
        # Find all PDF and TXT files
        pdf_files = list(docs_dir.glob("*.pdf"))
        txt_files = list(docs_dir.glob("*.txt"))
        all_files = pdf_files + txt_files
        
        if not all_files:
            logger.error(f"No PDF or TXT files found in {docs_dir}")
            return [], []
        
        logger.info(f"Found {len(all_files)} documents to process")
        
        # Process each document
        for file_path in tqdm(all_files, desc="Processing documents"):
            chunks = self.process_document(file_path)
            all_chunks.extend(chunks)
        
        # Separate texts and metadata
        documents = [chunk['text'] for chunk in all_chunks]
        metadata = [chunk['metadata'] for chunk in all_chunks]
        
        logger.info(f"Total chunks created: {len(documents)}")
        return documents, metadata


def main():
    """Main function to ingest documents and build vector store."""
    logger.info("=" * 50)
    logger.info("NyayaBot Document Ingestion Script")
    logger.info("=" * 50)
    
    # Check if documents directory exists
    if not LEGAL_DOCS_DIR.exists():
        logger.error(f"Documents directory not found: {LEGAL_DOCS_DIR}")
        logger.error("Please create the directory and add your legal documents.")
        sys.exit(1)
    
    # Initialize processor
    processor = DocumentProcessor()
    
    # Process all documents
    logger.info(f"Processing documents from: {LEGAL_DOCS_DIR}")
    documents, metadata = processor.process_all_documents(LEGAL_DOCS_DIR)
    
    if not documents:
        logger.error("No documents were processed. Please check your documents directory.")
        sys.exit(1)
    
    # Initialize retriever and build index
    logger.info("Building vector store...")
    retriever = DocumentRetriever()
    retriever.build_index(documents, metadata)
    
    logger.info("=" * 50)
    logger.info("Document ingestion completed successfully!")
    logger.info(f"Total documents indexed: {len(documents)}")
    logger.info(f"Vector store saved to: {retriever.vector_store_path}")
    logger.info("=" * 50)
    logger.info("\nYou can now start the API server with:")
    logger.info("  python -m app.main")
    logger.info("\nOr use uvicorn:")
    logger.info("  uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()