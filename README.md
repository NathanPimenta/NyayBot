NyayaBot - Multilingual Legal Question Answering System
NyayaBot is an intelligent legal question-answering system designed to help citizens of Maharashtra easily access information about their legal and constitutional rights. It supports queries in English, Hindi, and Marathi.

Features
🌍 Multilingual Support: Ask questions in English, Hindi, or Marathi
🔍 Semantic Search: Uses advanced NLP to find relevant legal documents
🤖 AI-Powered Answers: Generates clear, citizen-friendly explanations
📚 Document-Backed: All answers are grounded in authoritative legal sources
⚡ GPU Optimized: Efficient models optimized for 8GB VRAM (RTX 4060)
🔒 Local Processing: All processing happens locally for data privacy
System Requirements
GPU: NVIDIA GPU with 8GB+ VRAM (tested on RTX 4060)
RAM: 16GB+ recommended
Storage: 10GB+ for models and data
OS: Ubuntu 20.04+ / Windows 10+ with WSL2
Python: 3.10+
Installation
1. Clone the Repository
bash
git clone <your-repo-url>
cd nyayabot
2. Create Virtual Environment
bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
bash
pip install -r requirements.txt
4. Install PyTorch with CUDA Support
bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
Data Preparation
1. Add Legal Documents
Place your legal documents (PDF or TXT format) in the data/legal_docs/ directory:

bash
data/legal_docs/
├── constitution_of_india.pdf
├── maharashtra_state_laws.pdf
├── ipc_sections.pdf
└── ...
Recommended Documents:

Constitution of India
Indian Penal Code (IPC)
Maharashtra State Laws
Criminal Procedure Code (CrPC)
Civil Procedure Code (CPC)
2. Build Vector Store
Process the documents and create the vector index:

bash
python scripts/ingest_data.py
This will:

Extract text from all PDF/TXT files
Split documents into chunks
Create embeddings using multilingual model
Build FAISS index for fast retrieval
Save the index to data/vector_store/
Note: This process may take 10-30 minutes depending on document size.

Running the API
Start the Server
bash
# Using uvicorn
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Or using Python directly
python -m app.main
The API will be available at: http://localhost:8000

API Documentation
Interactive API documentation is available at:

Swagger UI: http://localhost:8000/docs
ReDoc: http://localhost:8000/redoc
Usage Examples
1. Using cURL
bash
# Ask a question in English
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are fundamental rights?",
    "language": "en",
    "include_sources": true
  }'

# Ask a question in Hindi
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "मौलिक अधिकार क्या हैं?",
    "include_sources": true
  }'

# Ask a question in Marathi
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "मूलभूत हक्क काय आहेत?",
    "include_sources": true
  }'
2. Using Python
python
import requests

# API endpoint
url = "http://localhost:8000/ask"

# Ask a question
response = requests.post(url, json={
    "query": "What is Article 21 of the Indian Constitution?",
    "language": "en",
    "top_k": 5,
    "include_sources": True
})

result = response.json()
print("Answer:", result["answer"])
print("Sources:", result["sources"])
3. Using JavaScript/Fetch
javascript
const response = await fetch('http://localhost:8000/ask', {
  method: 'POST',
  headers: {
    'Content-Type': 'application/json',
  },
  body: JSON.stringify({
    query: 'What are my rights as a citizen?',
    include_sources: true
  })
});

const data = await response.json();
console.log('Answer:', data.answer);
API Endpoints
POST /ask
Ask a legal question

Request Body:

json
{
  "query": "Your question here",
  "language": "en",  // Optional: en/hi/mr (auto-detected if not provided)
  "top_k": 5,        // Number of documents to retrieve
  "include_sources": true
}
Response:

json
{
  "answer": "Generated answer...",
  "language": "en",
  "original_query": "Your question",
  "english_query": "Translated query if needed",
  "sources": [...],
  "success": true
}
GET /health
Check system health

GET /languages
Get supported languages

POST /batch-ask
Ask multiple questions at once

Docker Deployment
Build Docker Image
bash
docker build -t nyayabot:latest .
Run Container
bash
docker run -d \
  --name nyayabot \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  nyayabot:latest
Configuration
Edit app/config.py to customize:

Models: Change embedding or generation models
Chunk Size: Adjust document chunking parameters
Retrieval: Modify number of documents retrieved
Generation: Tune temperature, max tokens, etc.
Using Different Models
For better quality (requires more VRAM):

python
# In app/config.py
GENERATOR_MODEL = "google/flan-t5-large"  # 780M params
For faster inference:

python
GENERATOR_MODEL = "google/flan-t5-small"  # 80M params
Performance Optimization
Memory Usage
FLAN-T5-Base: ~2GB VRAM
Embeddings: ~1GB VRAM
8-bit Quantization: Saves ~50% VRAM
Total Usage: ~4-5GB VRAM
Speed Optimization
Use batch processing for multiple queries
Enable caching for translations
Use GPU inference for faster generation
Troubleshooting
CUDA Out of Memory
bash
# Enable 8-bit quantization in config.py
USE_8BIT = True

# Or use a smaller model
GENERATOR_MODEL = "google/flan-t5-small"
Translation Errors
bash
# Check internet connection (Google Translate API requires internet)
# Or implement offline translation with IndicTrans
Documents Not Found
bash
# Rebuild vector store
python scripts/ingest_data.py
Project Structure
nyayabot/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application
│   ├── config.py            # Configuration
│   ├── nlp_core/
│   │   ├── translator.py    # Translation module
│   │   ├── retriever.py     # Document retrieval
│   │   └── generator.py     # Answer generation
│   └── services/
│       └── qa_service.py    # QA pipeline orchestration
├── data/
│   ├── legal_docs/          # Place your documents here
│   └── vector_store/        # Generated vector index
├── scripts/
│   └── ingest_data.py       # Document processing script
├── requirements.txt
├── Dockerfile
└── README.md
Contributing
Contributions are welcome! Please:

Fork the repository
Create a feature branch
Make your changes
Submit a pull request
License
[Your chosen license]

Acknowledgments
Constitution of India
Maharashtra State Government
Hugging Face Transformers
FAISS by Facebook Research
LangChain
Contact
For questions or support, please contact: [your-email@example.com]

Note: This system is for informational purposes only and should not be considered as legal advice. Always consult with a qualified legal professional for specific legal matters.

