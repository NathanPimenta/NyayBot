NyayaBot Project

NyayaBot is a multilingual legal question-answering system for Maharashtra, supporting English, Hindi, and Marathi.

This project provides the complete backend API and data processing pipeline.

Project Structure

/app: Contains the main FastAPI application.

/main.py: The API endpoint definitions.

/config.py: Central configuration for model names and paths.

/nlp_core: Core NLP modules for translation, retrieval, and generation.

/services: Orchestration logic that combines NLP modules.

/data: Holds raw data and the processed vector index.

/legal_docs: Place your legal documents (.pdf, .txt) here.

/vector_store: The FAISS vector index will be stored here.

/scripts: Contains helper scripts.

/ingest_data.py: A one-time script to read your documents, process them, and create the vector store.

requirements.txt: Python dependencies.

Dockerfile: For containerizing the application.

How to Run

1. Install Dependencies:

pip install -r requirements.txt


2. Add Legal Documents:

Place your authoritative legal documents (e.g., constitution_of_india.pdf, maharashtra_state_laws.txt) into the data/legal_docs/ directory.

3. Process Your Documents (Ingestion):

You must run this script once to build the semantic search index. It will read all documents from /data/legal_docs, split them into chunks, create vector embeddings, and save a FAISS index to /data/vector_store/.

python scripts/ingest_data.py


4. Run the API Server:

This command starts the FastAPI server. The first time it runs, it will download the Hugging Face models, which may take some time.

uvicorn app.main:app --host 0.0.0.0 --port 8000


5. Query the API:

The server is now running. You can send a POST request to the /ask endpoint.

Example using curl:

curl -X 'POST' \
  '[http://127.0.0.1:8000/ask](http://127.0.0.1:8000/ask)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "What are my fundamental rights?",
  "language": "en"
}'


Example in Marathi:

curl -X 'POST' \
  '[http://127.0.0.1:8000/ask](http://127.0.0.1:8000/ask)' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "query": "माझे मूलभूत अधिकार काय आहेत?",
  "language": "mr"
}'
