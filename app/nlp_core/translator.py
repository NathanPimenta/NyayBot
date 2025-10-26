from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ..config import Config

class Translator:
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        
        print("Loading translation models...")
        for key, model_name in Config.TRANSLATION_MODELS.items():
            try:
                print(f"  - Loading {key} ({model_name})...")
                self.tokenizers[key] = AutoTokenizer.from_pretrained(model_name)
                self.models[key] = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            except Exception as e:
                print(f"Failed to load model {model_name}: {e}")
                raise e
        print("Translation models loaded.")

    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        """
        Translates text from a source language to a target language.
        
        :param text: The text to translate.
        :param source_lang: 'en', 'hi', or 'mr'
        :param target_lang: 'en', 'hi', or 'mr'
        :return: The translated text.
        """
        # If no translation is needed
        if source_lang == target_lang:
            return text
        
        model_key = f"{source_lang}-{target_lang}"
        
        if model_key not in self.models:
            print(f"No model found for translation: {model_key}")
            # As a fallback, return the original text
            return text
            
        # Get the appropriate model and tokenizer
        tokenizer = self.tokenizers[model_key]
        model = self.models[model_key]

        # Prepare the text for translation
        # For Hindi/Marathi, we may need to prefix (check model card)
        # For Helsinki models, this is often not required.
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)

        # Generate translation
        translated_tokens = model.generate(**inputs, max_length=512, num_beams=5, early_stopping=True)
        
        # Decode
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        
        return translated_text