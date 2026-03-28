"""
app/api/routes/documents.py

POST /api/documents/ingest — upload a property document into the vector store.

Called by .NET backend when a property manager uploads a new document.
Not called by React frontend directly.
"""
from __future__ import annotations

import os
import tempfile

import structlog
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel

from app.services.document_loader import DocumentLoaderError, DocumentLoaderService

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


class IngestErrorResponse(BaseModel):
    success: bool = False
    error: str
    filename: str = ""


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(..., description="PDF, TXT, or DOCX"),
    property_id: str = Form(...,
                            description="Property ID this document belongs to"),
    doc_type: str = Form(
        default="",
        description="lease | contract | strata | inspection_report | document",
    ),
) -> IngestResponse:
    """
    Ingest a property document into the vector store.
    After ingestion, searchable via the search_documents tool.
    """
    filename = file.filename or "unknown"
    tmp_path = None

    logger.info("ingest_request", property_id=property_id,
                filename=filename, doc_type=doc_type)

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Allowed: PDF, TXT, DOCX",
        )

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)

    if size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {size_mb:.1f}MB. Maximum: {MAX_FILE_SIZE_MB}MB",
        )

    try:
        suffix = os.path.splitext(filename)[1] or ".pdf"
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix=f"ingest_{property_id}_"
        ) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        loader = DocumentLoaderService()
        result = loader.ingest_file(
            file_path=tmp_path,
            property_id=property_id,
            doc_type=doc_type or None,
        )

        logger.info(
            "ingest_complete",
            property_id=property_id,
            filename=filename,
            chunks=result["chunk_count"],
        )

        return IngestResponse(
            success=True,
            filename=filename,
            property_id=property_id,
            doc_type=result["doc_type"],
            chunk_count=result["chunk_count"],
            message=f"Ingested {filename} — {result['chunk_count']} chunks stored",
        )

    except DocumentLoaderError as exc:
        logger.warning("ingest_loader_error", error=str(exc))
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as exc:
        logger.exception("ingest_unexpected_error", error=str(exc))
        raise HTTPException(
            status_code=500, detail="Document ingestion failed. Please try again.")

    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
