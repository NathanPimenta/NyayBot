import logging
from typing import Dict, List, Optional
from app.nlp_core.translator import Translator
from app.nlp_core.retriever import DocumentRetriever
from app.nlp_core.generator import AnswerGenerator
from app.config import ModelConfig

logger = logging.getLogger(__name__)

class QAService:
    """
    Orchestrates the complete question-answering pipeline:
    1. Translation (if needed)
    2. Document retrieval
    3. Answer generation
    4. Translation back (if needed)
    """
    
    def __init__(self):
        logger.info("Initializing QA Service...")
        
        # Initialize components
        self.translator = Translator(cache_size=ModelConfig.TRANSLATION_CACHE_SIZE)
        self.retriever = DocumentRetriever()
        self.generator = AnswerGenerator()
        
        logger.info("QA Service initialized successfully")
    
    def answer_question(
        self,
        query: str,
        language: Optional[str] = None,
        top_k: int = None,
        include_sources: bool = True
    ) -> Dict:
        """
        Complete pipeline to answer a legal question.
        
        Args:
            query: User question in any supported language
            language: Language code (will auto-detect if None)
            top_k: Number of documents to retrieve
            include_sources: Whether to include source documents in response
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        try:
            logger.info(f"Processing query: {query[:50]}...")
            
            # Step 1: Translate query to English if needed
            translation_result = self.translator.translate_query(query, language)
            detected_language = translation_result['detected_language']
            english_query = translation_result['english_query']
            
            logger.info(f"Detected language: {detected_language}")
            logger.info(f"English query: {english_query}")
            
            # Step 2: Retrieve relevant documents
            retrieved_docs = self.retriever.retrieve(english_query, top_k)
            
            if not retrieved_docs:
                return {
                    "answer": "I couldn't find relevant information to answer your question. Please try rephrasing or ask about a different topic.",
                    "language": detected_language,
                    "sources": [],
                    "success": False
                }
            
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            # Step 3: Prepare context for generation
            context = self.retriever.get_context_for_generation(retrieved_docs)
            
            # Step 4: Generate answer in English
            english_answer = self.generator.generate_answer(english_query, context)
            
            logger.info(f"Generated English answer: {english_answer[:100]}...")
            
            # Step 5: Translate answer back to user's language
            final_answer = self.translator.translate_answer(english_answer, detected_language)
            
            logger.info(f"Final answer: {final_answer[:100]}...")
            
            # Prepare response
            response = {
                "answer": final_answer,
                "language": detected_language,
                "original_query": query,
                "english_query": english_query,
                "success": True
            }
            
            # Include sources if requested
            if include_sources:
                response["sources"] = self._format_sources(retrieved_docs)
            
            return response
            
        except Exception as e:
            logger.error(f"Error in QA pipeline: {e}", exc_info=True)
            return {
                "answer": "I apologize, but I encountered an error while processing your question. Please try again.",
                "language": language or "en",
                "success": False,
                "error": str(e)
            }
    
    def _format_sources(self, retrieved_docs: List[Dict]) -> List[Dict]:
        """
        Format retrieved documents for response.
        
        Args:
            retrieved_docs: List of retrieved documents
            
        Returns:
            Formatted source information
        """
        sources = []
        for doc in retrieved_docs:
            sources.append({
                "rank": doc['rank'],
                "text": doc['document'][:300] + "..." if len(doc['document']) > 300 else doc['document'],
                "source": doc['metadata'].get('source', 'Unknown'),
                "page": doc['metadata'].get('page', None),
                "relevance_score": round(doc['relevance_score'], 3)
            })
        return sources
    
    def batch_answer_questions(self, queries: List[str], language: Optional[str] = None) -> List[Dict]:
        """
        Answer multiple questions in batch.
        
        Args:
            queries: List of questions
            language: Language code (same for all queries)
            
        Returns:
            List of answer dictionaries
        """
        results = []
        for query in queries:
            result = self.answer_question(query, language)
            results.append(result)
        return results
    
    def get_document_summary(self, document_name: str) -> Dict:
        """
        Get a summary of a specific legal document.
        
        Args:
            document_name: Name of the document
            
        Returns:
            Document summary
        """
        try:
            # Search for document chunks
            retrieved_docs = self.retriever.retrieve(document_name, top_k=3)
            
            if not retrieved_docs:
                return {
                    "summary": "Document not found.",
                    "success": False
                }
            
            # Combine document chunks
            full_text = "\n".join([doc['document'] for doc in retrieved_docs])
            
            # Generate summary
            summary = self.generator.generate_summary(full_text)
            
            return {
                "summary": summary,
                "source": retrieved_docs[0]['metadata'].get('source', 'Unknown'),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Error generating document summary: {e}")
            return {
                "summary": "Error generating summary.",
                "success": False,
                "error": str(e)
            }
    
    def health_check(self) -> Dict:
        """
        Check if all components are functioning properly.
        
        Returns:
            Health status of each component
        """
        status = {
            "translator": "ok",
            "retriever": "ok",
            "generator": "ok",
            "overall": "ok"
        }
        
        try:
            # Test translator
            self.translator.translate_to_english("test", "en")
        except Exception as e:
            status["translator"] = f"error: {str(e)}"
            status["overall"] = "degraded"
        
        try:
            # Test retriever
            if self.retriever.index is None:
                status["retriever"] = "error: index not loaded"
                status["overall"] = "degraded"
        except Exception as e:
            status["retriever"] = f"error: {str(e)}"
            status["overall"] = "degraded"
        
        try:
            # Test generator (simple check)
            if self.generator.model is None:
                status["generator"] = "error: model not loaded"
                status["overall"] = "degraded"
        except Exception as e:
            status["generator"] = f"error: {str(e)}"
            status["overall"] = "degraded"
        
        return status