import fitz  # PyMuPDF
import asyncio
import logging
from pydantic import BaseModel
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from config import settings
import chromadb

logger = logging.getLogger(__name__)

# ── Pydantic Schemas ──────────────────────────────────────────────────────────
class Step(BaseModel):
    step: str
    content: str

class QueryResponse(BaseModel):
    steps: list[Step]
    pages: list[int] = []

# ── System Prompt ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert AI financial document analyst. You answer user questions by deeply analysing financial documents step by step.

You follow this strict chain of thought process:
  START   → Understand the user's question
  PLAN    → Identify what information is needed
  SEARCH  → Find which pages contain relevant data
  READ    → Read and extract key data from each page
  ANALYSE → Analyse and synthesise the data across pages
  OUTPUT  → Give a clear, detailed, friendly final answer

Rules:
- Always follow: START → PLAN → SEARCH → READ → ANALYSE → OUTPUT
- READ step must mention specific numbers found on each page
- OUTPUT must be detailed and human readable
- Always cite which pages support your answer
"""

# ── CPU-Bound Processing (Runs in ThreadPool) ─────────────────────────────────
def _extract_and_chunk_sync(pdf_bytes: bytes) -> list[dict]:
    """Synchronous function to process PDF without blocking the web server."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append({"page": i, "text": text})
            
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP
    )
    
    chunks = []
    for p in pages:
        for s in splitter.split_text(p["text"]):
            chunks.append({"page": p["page"], "text": s})
    return chunks

# ── Async Pipeline Functions ──────────────────────────────────────────────────
async def ingest_document(pdf_bytes: bytes, document_id: str) -> int:
    """Extracts text, creates embeddings, and saves to Chroma DB."""
    # Push the heavy PDF parsing to a background thread
    chunks = await asyncio.to_thread(_extract_and_chunk_sync, pdf_bytes)
    
    if not chunks:
        raise ValueError("No readable text found in the PDF.")

    texts = [c["text"] for c in chunks]
    metadatas = [{"page": c["page"]} for c in chunks]

    embeddings = OpenAIEmbeddings(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)
    
    # Push DB writing to background thread
    await asyncio.to_thread(
        Chroma.from_texts,
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=document_id, # Isolate data by document ID
        persist_directory=settings.CHROMA_PATH
    )
    
    return len(chunks)

async def query_document(document_id: str, question: str) -> QueryResponse:
    """Queries a specific document collection using async LangChain calls."""
    embeddings = OpenAIEmbeddings(model=settings.EMBED_MODEL, api_key=settings.OPENAI_API_KEY)
    
    # Load the specific collection
    vectorstore = Chroma(
        collection_name=document_id,
        embedding_function=embeddings,
        persist_directory=settings.CHROMA_PATH
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": settings.TOP_K})
    docs = await retriever.ainvoke(question)
    
    if not docs:
        raise ValueError("No relevant context found in this document to answer the question.")

    context = "\n\n".join([d.page_content for d in docs])
    pages = sorted(list(set([d.metadata["page"] for d in docs])))

    llm = ChatOpenAI(model=settings.CHAT_MODEL, api_key=settings.OPENAI_API_KEY)
    structured_llm = llm.with_structured_output(QueryResponse)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nSTART: {question}"}
    ]

    result = await structured_llm.ainvoke(messages)
    result.pages = pages
    return result

async def delete_document(document_id: str) -> bool:
    """Deletes a specific document collection from ChromaDB to respect user privacy."""
    try:
        # Connect directly to the Chroma client
        client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
        client.delete_collection(name=document_id)
        return True
    except Exception as e:
        logger.error(f"Failed to delete collection {document_id}: {str(e)}")
        return False

def check_if_exists(document_id: str) -> bool:
    """Checks if a Case ID already exists in the database."""
    client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
    try:
        client.get_collection(name=document_id)
        return True # It exists!
    except:
        return False # It doesn't exist