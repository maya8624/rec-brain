"""
app/api/routes/documents.py

POST /api/documents/ingest — receive a document from an Azure Function trigger and store it in pgvector.
The Azure Function fires when a file is uploaded to blob storage and POSTs the file here.
"""
from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.api.dependencies import verify_internal_key
from app.services.document_ingestion_service import DocumentIngestionError

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/documents", tags=["documents"])

ALLOWED_CONTENT_TYPES = {
    "application/pdf",
    "text/plain",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}
MAX_FILE_SIZE_MB = 20


class IngestResponse(BaseModel):
    success: bool
    filename: str
    property_id: str
    doc_type: str
    chunk_count: int
    message: str


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    request: Request,
    file: UploadFile = File(..., description="PDF, TXT, or DOCX"),
    property_id: str = Form(default="", description="Property ID this document belongs to — optional for global documents"),
    doc_type: str = Form(default="", description="Optional — auto-classified if omitted"),
    _: str = Depends(verify_internal_key),
) -> IngestResponse:
    """
    Parse, classify, chunk, embed, and upsert a document into pgvector.
    Called by an Azure Function when a file is uploaded to blob storage.
    """
    filename = file.filename or "unknown"
    logger.info("ingest_request", filename=filename, property_id=property_id)

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, TXT, DOCX",
        )

    content = await file.read()

    if len(content) / (1024 * 1024) > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large. Maximum: {MAX_FILE_SIZE_MB}MB",
        )

    try:
        result = await request.app.state.document_ingestion_service.ingest(
            content=content,
            filename=filename,
            property_id=property_id,
            doc_type=doc_type,
        )
        return IngestResponse(
            success=True,
            filename=filename,
            property_id=property_id,
            doc_type=result["doc_type"],
            chunk_count=result["chunk_count"],
            message=f"Ingested {filename} — {result['chunk_count']} chunks stored",
        )

    except DocumentIngestionError as exc:
        logger.warning("ingest_error", error=str(exc))
        raise HTTPException(status_code=400, detail=str(exc))

    except Exception as exc:
        logger.exception("ingest_unexpected_error", error=str(exc))
        raise HTTPException(status_code=500, detail="Document ingestion failed. Please try again.")
