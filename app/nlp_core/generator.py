from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from ..config import Config

class AnswerGenerator:
    def __init__(self):
        print(f"Loading QA/Generator model: {Config.QA_MODEL_NAME}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(Config.QA_MODEL_NAME)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(Config.QA_MODEL_NAME)
        except Exception as e:
            print(f"Failed to load model {Config.QA_MODEL_NAME}: {e}")
            raise e
        print("QA/Generator model loaded.")

    def generate_answer(self, query: str, context_docs: list) -> str:
        """
        Generates a simplified, citizen-friendly answer based on the query
        and the retrieved context.
        
        :param query: The (English) user query.
        :param context_docs: A list of LangChain Document objects.
        :return: A generated answer string.
        """
        
        # 1. Combine the context
        context_string = "\n\n".join([doc.page_content for doc in context_docs])
        
        # 2. Create a prompt for the model (FLAN-T5 is instruction-tuned)
        # We explicitly ask it to answer *only* from the context and to be simple.
        prompt = f"""
        Answer the following question in a simple, easy-to-understand way for a citizen.
        Use *only* the provided context. Do not use any outside knowledge.
        
        Question: {query}
        
        Context:
        {context_string}
        
        Simplified Answer:
        """
        
        # 3. Tokenize and Generate
        print("Generating answer...")
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        outputs = self.model.generate(
            **inputs, 
            max_length=256, 
            num_beams=4, 
            early_stopping=True
        )
        
        # 4. Decode
        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return answer