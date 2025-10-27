import logging
from typing import Dict, Optional
from functools import lru_cache
from googletrans import Translator as GoogleTranslator

logger = logging.getLogger(__name__)

class Translator:
    """
    Handles translation between English, Hindi, and Marathi.
    Uses Google Translate API with caching for efficiency.
    """
    
    def __init__(self, cache_size: int = 1000):
        self.translator = GoogleTranslator()
        self.cache_size = cache_size
        logger.info("Translator initialized")
    
    @lru_cache(maxsize=1000)
    def translate_to_english(self, text: str, source_lang: str) -> str:
        """
        Translate text from source language to English.
        
        Args:
            text: Input text to translate
            source_lang: Source language code ('hi', 'mr', 'en')
            
        Returns:
            Translated text in English
        """
        if not text or not text.strip():
            return ""
        
        # If already in English, return as is
        if source_lang == 'en':
            return text
        
        try:
            result = self.translator.translate(text, src=source_lang, dest='en')
            translated_text = result.text
            logger.info(f"Translated from {source_lang} to English")
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Return original text if translation fails
            return text
    
    @lru_cache(maxsize=1000)
    def translate_from_english(self, text: str, target_lang: str) -> str:
        """
        Translate text from English to target language.
        
        Args:
            text: Input text in English
            target_lang: Target language code ('hi', 'mr', 'en')
            
        Returns:
            Translated text in target language
        """
        if not text or not text.strip():
            return ""
        
        # If target is English, return as is
        if target_lang == 'en':
            return text
        
        try:
            result = self.translator.translate(text, src='en', dest=target_lang)
            translated_text = result.text
            logger.info(f"Translated from English to {target_lang}")
            return translated_text
        except Exception as e:
            logger.error(f"Translation error: {e}")
            # Return original text if translation fails
            return text
    
    def detect_language(self, text: str) -> str:
        """
        Detect the language of input text.
        
        Args:
            text: Input text
            
        Returns:
            Detected language code ('en', 'hi', 'mr')
        """
        if not text or not text.strip():
            return 'en'
        
        try:
            detection = self.translator.detect(text)
            detected_lang = detection.lang
            
            # Map to supported languages
            if detected_lang in ['hi', 'mr', 'en']:
                return detected_lang
            else:
                # Default to English for unsupported languages
                logger.warning(f"Unsupported language detected: {detected_lang}, defaulting to English")
                return 'en'
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return 'en'
    
    def translate_query(self, query: str, source_lang: Optional[str] = None) -> Dict[str, str]:
        """
        Translate query to English for processing.
        Auto-detects language if not provided.
        
        Args:
            query: User query
            source_lang: Source language (optional, will auto-detect if None)
            
        Returns:
            Dictionary with original query, detected language, and English translation
        """
        if source_lang is None:
            source_lang = self.detect_language(query)
        
        english_query = self.translate_to_english(query, source_lang)
        
        return {
            "original_query": query,
            "detected_language": source_lang,
            "english_query": english_query
        }
    
    def translate_answer(self, answer: str, target_lang: str) -> str:
        """
        Translate answer from English to target language.
        
        Args:
            answer: Answer in English
            target_lang: Target language code
            
        Returns:
            Translated answer
        """
        return self.translate_from_english(answer, target_lang)