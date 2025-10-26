import os

class Config:
    # --- Paths ---
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    ROOT_DIR = os.path.dirname(BASE_DIR)
    
    # Path to the directory containing legal documents
    LEGAL_DOCS_PATH = os.path.join(ROOT_DIR, "data", "legal_docs")
    
    # Path where the FAISS vector index will be saved
    VECTOR_STORE_PATH = os.path.join(ROOT_DIR, "data", "vector_store")

    # --- Model Names (from Hugging Face) ---

    # Multilingual model for semantic search (embeddings)
    EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # Model for generative Question Answering / Summarization
    QA_MODEL_NAME = "google/flan-t5-small"

    # Translation models (Helsinki-NLP)
    TRANSLATION_MODELS = {
        "en-hi": "Helsinki-NLP/opus-mt-en-hi",
        "hi-en": "Helsinki-NLP/opus-mt-hi-en",
        "en-mr": "Helsinki-NLP/opus-mt-en-mr",
        "mr-en": "Helsinki-NLP/opus-mt-mr-en",
    }