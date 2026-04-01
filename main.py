import logging
import re
import hashlib
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Security
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_pipeline import ingest_document, query_document, delete_document, check_if_exists, QueryResponse
from config import settings

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="DocLens Finance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.API_SECRET_KEY:
        raise HTTPException(status_code=403, detail="Could not validate API credentials")
    return api_key

class QueryRequest(BaseModel):
    case_id: str
    question: str

# In-memory dictionary to track file hashes to prevent exact duplicate PDFs
# In a real app, you would save this to a Postgres database
file_hash_db = set()

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/ingest", tags=["Document Processing"])
async def upload_pdf(
    file: UploadFile = File(...), 
    case_id: str = Form(...), # User provides the ID here
    api_key: str = Depends(verify_api_key)
):
    # 1. Rule Validation: Case ID must be 3 Letters, a dash, and 4 Numbers (e.g., DOC-1234)
    if not re.match(r"^[A-Z]{3}-\d{4}$", case_id):
        raise HTTPException(
            status_code=400, 
            detail="Invalid Case ID format. It must be 3 capital letters, a dash, and 4 numbers (e.g., FIN-2024)."
        )

    # 2. Duplicate ID Check
    if check_if_exists(case_id):
        raise HTTPException(
            status_code=409, 
            detail=f"Case ID '{case_id}' already exists. Please try again with a different ID."
        )

    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDFs are accepted.")
    
    try:
        pdf_bytes = await file.read()
        
        # 3. Duplicate PDF File Check (Digital Fingerprint)
        file_hash = hashlib.sha256(pdf_bytes).hexdigest()
        if file_hash in file_hash_db:
            raise HTTPException(
                status_code=409, 
                detail="This exact PDF has already been uploaded under a different Case ID. Please try again."
            )
        
        # If it passes all checks, ingest the document
        chunks_count = await ingest_document(pdf_bytes, case_id)
        
        # Save the hash so we remember this file
        file_hash_db.add(file_hash)
        
        return {
            "case_id": case_id, 
            "chunks_processed": chunks_count,
            "message": "Document successfully processed and stored."
        }
    except HTTPException:
        raise # Pass through our custom errors
    except Exception as e:
        logger.error(f"Failed to ingest document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/query", response_model=QueryResponse, tags=["Document Processing"])
async def ask_question(
    request: QueryRequest, 
    api_key: str = Depends(verify_api_key)
):
    try:
        result = await query_document(request.case_id, request.question)
        return result
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))


# 4. The Delete Endpoint
@app.delete("/delete/{case_id}", tags=["Data Management"])
async def delete_case_data(
    case_id: str,
    api_key: str = Depends(verify_api_key)
):
    if not check_if_exists(case_id):
        raise HTTPException(status_code=404, detail="Case ID not found.")
        
    success = await delete_document(case_id)
    
    if success:
        return {"message": f"All data for Case ID '{case_id}' has been permanently deleted."}
    else:
        raise HTTPException(status_code=500, detail="Failed to delete data. Please contact support.")

@app.get("/health", tags=["System"])
def health_check():
    return {"status": "healthy"}