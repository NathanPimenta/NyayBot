import os
import sys
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Add the app directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.config import Config

def create_vector_store():
    """
    Reads documents from the legal_docs directory, splits them,
    creates embeddings, and saves them to a FAISS vector store.
    """
    print(f"Loading documents from {Config.LEGAL_DOCS_PATH}...")

    # --- UPDATED DOCUMENT LOADING ---
    # The 'loader_cls_map' argument was removed/deprecated.
    # We now load each file type with its own loader and a specific glob pattern.

    # 1. Load .txt files
    print("Loading .txt files...")
    txt_loader = DirectoryLoader(
        Config.LEGAL_DOCS_PATH,
        glob="**/*.txt",
        loader_cls=TextLoader,
        show_progress=True,
        use_multithreading=True
    )
    txt_documents = txt_loader.load()
    print(f"Loaded {len(txt_documents)} .txt files.")

    # 2. Load .pdf files
    print("Loading .pdf files...")
    pdf_loader = DirectoryLoader(
        Config.LEGAL_DOCS_PATH,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
        use_multithreading=True
    )
    pdf_documents = pdf_loader.load()
    print(f"Loaded {len(pdf_documents)} .pdf files.")

    # 3. Combine the documents
    documents = txt_documents + pdf_documents
    # --- END OF UPDATED LOADING ---

    if not documents:
        print(f"No documents found in {Config.LEGAL_DOCS_PATH}. Exiting.")
        print("Please add your .pdf and .txt legal documents to that directory.")
        return

    print(f"Loaded {len(documents)} documents.")

    # Split documents into chunks
    print("Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150
    )
    docs_chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(docs_chunks)} chunks.")

    # Initialize the embedding model
    print(f"Loading embedding model: {Config.EMBEDDING_MODEL_NAME}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'} # Use 'cuda' if GPU is available
    )

    # Create the FAISS vector store
    print("Creating FAISS vector store from document chunks...")
    vector_store = FAISS.from_documents(docs_chunks, embeddings)

    # Save the vector store locally
    print(f"Saving vector store to {Config.VECTOR_STORE_PATH}...")
    vector_store.save_local(Config.VECTOR_STORE_PATH)

    print("\nDone!")
    print(f"Vector store is saved and ready at: {Config.VECTOR_STORE_PATH}")

if __name__ == "__main__":
    # Create the data/vector_store directory if it doesn't exist
    if not os.path.exists(Config.VECTOR_STORE_PATH):
        os.makedirs(Config.VECTOR_STORE_PATH)
        
    create_vector_store()



