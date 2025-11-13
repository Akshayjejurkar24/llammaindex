from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from llama_parse import LlamaParse
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import StorageContext, VectorStoreIndex
from qdrant_client import QdrantClient
from dotenv import load_dotenv
import tempfile
import uuid
import os

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLAMAPARSE_API_KEY = os.getenv("LLAMAPARSE_API_KEY")

# -----------------------------
# Initialize main dependencies
# -----------------------------
client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

embed_model = OpenAIEmbedding(
    model="text-embedding-3-large",
    api_key=OPENAI_API_KEY
)

llm = OpenAI(
    model="gpt-4o",
    api_key=OPENAI_API_KEY
)

parser = LlamaParse(
    api_key=LLAMAPARSE_API_KEY,
    result_type="markdown"
)

# -----------------------------
# FastAPI app setup
# -----------------------------
app = FastAPI(
    title="PDF RAG API",
    description="Upload PDF → Parse → Store → Query with RAG",
    version="1.0"
)

# -----------------------------
# Upload PDF and store embeddings
# -----------------------------
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF file, parse it, create embeddings, and store in Qdrant."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        # Parse document
        documents = parser.load_data(tmp_path)

        # Split into chunks
        node_parser = SentenceSplitter(chunk_size=1024)
        nodes = node_parser.get_nodes_from_documents(documents)

        for idx, node in enumerate(nodes):
            node.id_ = f"{file.filename}_node_{idx}_{uuid.uuid4()}"

        # Create vector store
        vector_store = QdrantVectorStore(client=client, collection_name="pdf_large_embeddings")
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        # Store embeddings
        index = VectorStoreIndex.from_documents(nodes, storage_context=storage_context, embed_model=embed_model)

        return {"message": f"File '{file.filename}' processed and stored successfully.", "chunks": len(nodes)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# -----------------------------
# Query stored PDFs using RAG
# -----------------------------
@app.get("/query")
def query_pdf(query: str = Query(..., description="Enter your question about the uploaded documents")):
    """Query the embedded PDF data using LLM + vector search."""
    try:
        vector_store = QdrantVectorStore(client=client, collection_name="pdf_large_embeddings")
        index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)

        query_engine = index.as_query_engine(llm=llm)
        response = query_engine.query(query)

        return {"query": query, "response": str(response)}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
