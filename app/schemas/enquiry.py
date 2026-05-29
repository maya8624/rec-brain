"""
app/schemas/enquiry.py

API request/response types for the enquiry endpoints.
"""

from pydantic import BaseModel

from app.schemas.rag import SourceChunk


class EnquiryRequest(BaseModel):
    id: str
    body: str
    tenant_id: str | None
    property_id: str | None
    intent: str | None


class EnquiryResponse(BaseModel):
    draft: str
    sources: list[SourceChunk]
