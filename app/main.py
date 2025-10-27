import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
from app.config import APIConfig, SUPPORTED_LANGUAGES
from app.services.qa_service import QAService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=APIConfig.TITLE,
    description=APIConfig.DESCRIPTION,
    version=APIConfig.VERSION
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize QA service (lazy loading on first request)
qa_service: Optional[QAService] = None

def get_qa_service() -> QAService:
    """Get or initialize QA service."""
    global qa_service
    if qa_service is None:
        logger.info("Initializing QA Service...")
        qa_service = QAService()
    return qa_service


# Pydantic models for API requests and responses
class QuestionRequest(BaseModel):
    query: str = Field(..., description="User question in any supported language")
    language: Optional[str] = Field(None, description="Language code (en/hi/mr). Auto-detected if not provided")
    top_k: Optional[int] = Field(5, description="Number of documents to retrieve", ge=1, le=10)
    include_sources: bool = Field(True, description="Include source documents in response")


class QuestionResponse(BaseModel):
    answer: str
    language: str
    original_query: str
    english_query: Optional[str] = None
    sources: Optional[List[dict]] = None
    success: bool


class BatchQuestionRequest(BaseModel):
    queries: List[str] = Field(..., description="List of questions")
    language: Optional[str] = Field(None, description="Language code for all queries")


class HealthResponse(BaseModel):
    status: str
    components: dict


class SupportedLanguagesResponse(BaseModel):
    languages: dict


# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": APIConfig.TITLE,
        "version": APIConfig.VERSION,
        "description": APIConfig.DESCRIPTION,
        "endpoints": {
            "ask": "/ask",
            "batch_ask": "/batch-ask",
            "health": "/health",
            "languages": "/languages"
        }
    }


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a legal question in any supported language.
    
    The system will:
    1. Auto-detect the language (or use provided language)
    2. Retrieve relevant legal documents
    3. Generate a citizen-friendly answer
    4. Return the answer in the original language
    """
    try:
        service = get_qa_service()
        result = service.answer_question(
            query=request.query,
            language=request.language,
            top_k=request.top_k,
            include_sources=request.include_sources
        )
        
        if not result.get("success", False):
            raise HTTPException(status_code=500, detail=result.get("error", "Failed to process question"))
        
        return result
    
    except Exception as e:
        logger.error(f"Error in /ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch-ask")
async def batch_ask_questions(request: BatchQuestionRequest):
    """
    Ask multiple questions in batch.
    """
    try:
        service = get_qa_service()
        results = service.batch_answer_questions(
            queries=request.queries,
            language=request.language
        )
        return {"results": results}
    
    except Exception as e:
        logger.error(f"Error in /batch-ask endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check the health status of the API and its components.
    """
    try:
        service = get_qa_service()
        status = service.health_check()
        return {
            "status": status["overall"],
            "components": status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "components": {"error": str(e)}
        }


@app.get("/languages", response_model=SupportedLanguagesResponse)
async def get_supported_languages():
    """
    Get list of supported languages.
    """
    return {"languages": SUPPORTED_LANGUAGES}


@app.get("/document-summary/{document_name}")
async def get_document_summary(document_name: str):
    """
    Get a summary of a specific legal document.
    """
    try:
        service = get_qa_service()
        result = service.get_document_summary(document_name)
        
        if not result.get("success", False):
            raise HTTPException(status_code=404, detail="Document not found or error generating summary")
        
        return result
    
    except Exception as e:
        logger.error(f"Error in /document-summary endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    logger.info("Starting NyayaBot API...")
    logger.info(f"API will be available at http://{APIConfig.HOST}:{APIConfig.PORT}")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down NyayaBot API...")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=APIConfig.HOST,
        port=APIConfig.PORT,
        reload=True
    )