from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict

from .services.qa_service import QAService, ServiceInitializationError

# --- Pydantic Models for API ---

class QAQuery(BaseModel):
    query: str = Field(..., example="What are my fundamental rights?")
    language: str = Field(..., example="en", description="Language code: 'en', 'hi', or 'mr'")

class SourceDocument(BaseModel):
    source: str = Field(..., example="constitution_of_india.pdf, page 5")
    content_snippet: str = Field(..., example="Article 14: Equality before law...")

class QAResponse(BaseModel):
    answer: str = Field(..., example="The Constitution grants several fundamental rights...")
    sources: List[SourceDocument]

# --- FastAPI App Initialization ---

app = FastAPI(
    title="NyayaBot API",
    description="Multilingual legal Q&A for Maharashtra (English, Hindi, Marathi).",
    version="1.0.0"
)

# --- Global Service Instance ---

# We initialize the service here on startup.
# This pre-loads all models into memory.
try:
    qa_service = QAService()
except ServiceInitializationError as e:
    print(f"FATAL ERROR: Could not initialize QAService. {e}")
    # In a real app, you might want to exit or prevent startup
    qa_service = None
except Exception as e:
    print(f"An unexpected error occurred during startup: {e}")
    qa_service = None

# --- API Endpoints ---

@app.on_event("startup")
def on_startup():
    if qa_service is None:
        print("Warning: QAService failed to initialize. API will not be functional.")
    else:
        print("NyayaBot API started successfully. Models are loaded.")

@app.get("/", summary="Health Check")
def read_root():
    """Root endpoint to check if the API is running."""
    if qa_service is None:
        return {"status": "error", "message": "NyayaBot API is running, but the QA Service failed to initialize."}
    return {"status": "ok", "message": "NyayaBot API is running."}


@app.post("/ask", response_model=QAResponse, summary="Ask a Legal Question")
def ask_question(request: QAQuery):
    """
    Main endpoint to ask a question in English, Hindi, or Marathi.
    """
    if qa_service is None:
        raise HTTPException(
            status_code=503, 
            detail="Service Unavailable: The QA models are not loaded."
        )

    # Validate language input
    supported_languages = {'en', 'hi', 'mr'}
    if request.language not in supported_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Please use one of: {supported_languages}"
        )

    try:
        # Call the service to get the answer
        response_data = qa_service.ask_question(
            query=request.query,
            source_lang=request.language
        )
        
        return QAResponse(
            answer=response_data["answer"],
            sources=[SourceDocument(**s) for s in response_data["sources"]]
        )

    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Configuration Error: A required file or directory was not found. Have you run the 'scripts/ingest_data.py' script?"
        )
    except Exception as e:
        print(f"Internal Server Error: {e}")
        # Log the full exception here in a real app
        # import traceback
        # traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"An internal error occurred: {e}"
        )