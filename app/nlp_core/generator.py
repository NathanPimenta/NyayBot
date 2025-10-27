import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from app.config import ModelConfig

logger = logging.getLogger(__name__)

class AnswerGenerator:
    """
    Generates answers to legal questions using a language model.
    Optimized for 8GB VRAM with quantization.
    """
    
    def __init__(self):
        self.device = ModelConfig.DEVICE
        logger.info(f"Initializing generator on device: {self.device}")
        
        # Load tokenizer
        logger.info(f"Loading tokenizer: {ModelConfig.GENERATOR_MODEL}")
        self.tokenizer = AutoTokenizer.from_pretrained(ModelConfig.GENERATOR_MODEL)
        
        # Configure 8-bit quantization for memory efficiency
        if ModelConfig.USE_8BIT and torch.cuda.is_available():
            logger.info("Using 8-bit quantization")
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0
            )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                ModelConfig.GENERATOR_MODEL,
                quantization_config=quantization_config,
                device_map="auto"
            )
        else:
            # Load model normally
            logger.info("Loading model without quantization")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(ModelConfig.GENERATOR_MODEL)
            self.model.to(self.device)
        
        self.model.eval()
        logger.info("Generator initialized successfully")
    
    def create_prompt(self, query: str, context: str) -> str:
        """
        Create a prompt for the model combining query and context.
        
        Args:
            query: User question in English
            context: Retrieved document context
            
        Returns:
            Formatted prompt
        """
        prompt = f"""Based on the following legal documents, answer the question in a clear and citizen-friendly manner. 
Provide specific article numbers, sections, or provisions when available.

Legal Context:
{context}

Question: {query}

Answer:"""
        return prompt
    
    def generate_answer(self, query: str, context: str) -> str:
        """
        Generate an answer to the query using the provided context.
        
        Args:
            query: User question in English
            context: Retrieved document context
            
        Returns:
            Generated answer
        """
        # Create prompt
        prompt = self.create_prompt(query, context)
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate answer
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=ModelConfig.MAX_NEW_TOKENS,
                    temperature=ModelConfig.TEMPERATURE,
                    top_p=ModelConfig.TOP_P,
                    do_sample=True,
                    num_beams=2,
                    early_stopping=True
                )
            
            # Decode output
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Post-process answer
            answer = self.post_process_answer(answer)
            
            logger.info("Answer generated successfully")
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I apologize, but I encountered an error while generating the answer. Please try rephrasing your question."
    
    def post_process_answer(self, answer: str) -> str:
        """
        Post-process the generated answer for better readability.
        
        Args:
            answer: Raw generated answer
            
        Returns:
            Processed answer
        """
        # Remove extra whitespace
        answer = ' '.join(answer.split())
        
        # Ensure answer starts with capital letter
        if answer and answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        
        # Ensure answer ends with punctuation
        if answer and answer[-1] not in '.!?':
            answer += '.'
        
        return answer
    
    def generate_summary(self, text: str, max_length: int = 150) -> str:
        """
        Generate a summary of a long text.
        
        Args:
            text: Text to summarize
            max_length: Maximum length of summary
            
        Returns:
            Summary text
        """
        prompt = f"Summarize the following text concisely:\n\n{text}\n\nSummary:"
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.3,
                    do_sample=True,
                    num_beams=2
                )
            
            summary = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.post_process_answer(summary)
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return text[:max_length] + "..."