import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
LEGAL_DOCS_DIR = DATA_DIR / "legal_docs"
VECTOR_STORE_DIR = DATA_DIR / "vector_store"

# Ensure directories exist
LEGAL_DOCS_DIR.mkdir(parents=True, exist_ok=True)
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Model configurations optimized for 8GB VRAM
class ModelConfig:
    # Embedding model - lightweight and multilingual
    EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_DIMENSION = 384
    
    # Generator model - efficient for 8GB VRAM
    # Using a smaller, quantized model for generation
    GENERATOR_MODEL = "google/flan-t5-base"  # 250M params, manageable on 8GB
    
    # Alternative: Use a quantized LLaMA model if you want better quality
    # GENERATOR_MODEL = "TheBloke/Llama-2-7B-Chat-GPTQ"  # Requires GPTQ support
    
    # Device configuration
    DEVICE = "cuda"  # Will use your 4060
    USE_8BIT = True  # Enable 8-bit quantization to save VRAM
    
    # Generation parameters
    MAX_NEW_TOKENS = 512
    TEMPERATURE = 0.3
    TOP_P = 0.9
    
    # Retrieval parameters
    TOP_K_DOCUMENTS = 5
    CHUNK_SIZE = 512
    CHUNK_OVERLAP = 50
    
    # Translation cache
    TRANSLATION_CACHE_SIZE = 1000

# API Configuration
class APIConfig:
    TITLE = "NyayaBot API"
    DESCRIPTION = "Multilingual Legal Question Answering System for Maharashtra"
    VERSION = "1.0.0"
    HOST = "0.0.0.0"
    PORT = 8000
    
# Supported languages
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "mr": "Marathi"
}

# Language codes mapping
LANG_CODE_MAPPING = {
    "english": "en",
    "hindi": "hi",
    "marathi": "mr"
}