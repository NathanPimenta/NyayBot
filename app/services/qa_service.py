import os
from ..config import Config
from ..nlp_core.translator import Translator
from ..nlp_core.retriever import Retriever
from ..nlp_core.generator import AnswerGenerator

class ServiceInitializationError(Exception):
    """Custom exception for service startup failures."""
    pass

class QAService:
    def __init__(self):
        """
        Initializes all necessary NLP components.
        This will pre-load all models into memory.
        """
        try:
            print("Initializing QAService...")
            self.translator = Translator()
            self.retriever = Retriever()
            self.generator = AnswerGenerator()
            print("QAService initialized successfully.")
        except FileNotFoundError as e:
            # This is a critical, user-fixable error (no vector store)
            raise ServiceInitializationError(e)
        except Exception as e:
            # Catch other potential model loading errors
            print(f"Error during QAService initialization: {e}")
            raise ServiceInitializationError(e)

    def ask_question(self, query: str, source_lang: str) -> dict:
        """
        Orchestrates the full Q&A pipeline.
        
        1. Translates query to English.
        2. Retrieves relevant documents.
        3. Generates an answer in English.
        4. Translates the answer back to the source language.
        5. Formats the response with sources.
        """
        print(f"\n--- New Query ---")
        print(f"Query: '{query}', Language: {source_lang}")

        # 1. Translate Query to English (processing language)
        if source_lang == 'en':
            eng_query = query
        else:
            eng_query = self.translator.translate(query, source_lang, 'en')
        print(f"Translated query: '{eng_query}'")

        # 2. Retrieve Relevant Documents
        # The retriever uses the English query
        try:
            relevant_docs = self.retriever.retrieve_docs(eng_query, k=3)
        except Exception as e:
            print(f"Error during retrieval: {e}")
            # This can happen if the index is corrupted or not found
            raise FileNotFoundError(f"Failed to retrieve from vector store. Is it initialized? Error: {e}")

        if not relevant_docs:
            print("No relevant documents found.")
            return {
                "answer": "I could not find any relevant information in my documents for your question.",
                "sources": []
            }

        # 3. Generate Answer (in English)
        eng_answer = self.generator.generate_answer(eng_query, relevant_docs)
        print(f"Generated (EN) answer: '{eng_answer[:100]}...'")

        # 4. Back-Translate Answer to Source Language
        if source_lang == 'en':
            final_answer = eng_answer
        else:
            final_answer = self.translator.translate(eng_answer, 'en', source_lang)
        print(f"Final ({source_lang}) answer: '{final_answer[:100]}...'")

        # 5. Format Sources
        sources = []
        for doc in relevant_docs:
            source_name = os.path.basename(doc.metadata.get('source', 'Unknown'))
            page = doc.metadata.get('page')
            source_display = f"{source_name}"
            if page is not None:
                source_display += f", page {page + 1}" # PyPDF page numbers are 0-indexed
                
            sources.append({
                "source": source_display,
                "content_snippet": doc.page_content[:150] + "..."
            })

        return {
            "answer": final_answer,
            "sources": sources
        }