📄 PDF RAG API — FastAPI + LlamaParse + Qdrant + OpenAI

Upload PDFs → Parse → Chunk → Embed → Store → Query using RAG

This project provides a simple Retrieval-Augmented Generation (RAG) pipeline using:

FastAPI – API framework

LlamaParse – PDF parsing to Markdown

LlamaIndex – chunking + embeddings + vector retrieval

Qdrant – vector database

OpenAI – embeddings + LLM answering

🚀 Features

✔ Upload PDF documents
✔ Parse to Markdown using LlamaParse
✔ Chunk using SentenceSplitter
✔ Generate embeddings with text-embedding-3-large
✔ Store embeddings in Qdrant
✔ Query documents with GPT-4o using RAG
✔ Simple and extendable API endpoints

📦 Installation
1. Clone the repository
git clone https://github.com/your-username/pdf-rag-api.git
cd pdf-rag-api

2. Create a virtual environment
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

🔑 Environment Variables

Create a .env file in the root directory:

QDRANT_URL=https://your-qdrant-instance-url
QDRANT_API_KEY=your_qdrant_api_key
OPENAI_API_KEY=your_openai_api_key
LLAMAPARSE_API_KEY=your_llamaparse_api_key


⚠️ Make sure your Qdrant collection allows vectors with the embedding size of 3072 (text-embedding-3-large).

▶️ Run the API

Start the server:

uvicorn main:app --reload


API will be available at:

http://localhost:8000


Interactive Swagger docs:

http://localhost:8000/docs

📤 Upload PDF

POST /upload

Example (curl)
curl -X POST "http://localhost:8000/upload" \
  -F "file=@example.pdf"

Response example
{
  "message": "File 'example.pdf' processed and stored successfully.",
  "chunks": 42
}

🔍 Query Your Documents

GET /query?query=your question

Example
curl "http://localhost:8000/query?query=What is the summary of section 2?"

Example response
{
  "query": "What is the summary of section 2?",
  "response": "Section 2 discusses..."
}

🧠 Architecture Overview
        ┌───────────────┐
        │   PDF Upload   │
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │  LlamaParse    │ → Markdown
        └───────┬───────┘
                ▼
        ┌───────────────┐
        │ SentenceSplitter│ → chunks
        └───────┬────────┘
                ▼
        ┌───────────────┐
        │ OpenAI Embedding│ → vectors
        └───────┬────────┘
                ▼
        ┌───────────────┐
        │    Qdrant      │ (vector store)
        └───────┬────────┘
                ▼
        ┌───────────────┐
        │ GPT-4o Query   │ + context
        └───────────────┘

📝 Folder Structure
📦 pdf-rag-api
 ┣ 📄 main.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md
 ┗ 📄 .env