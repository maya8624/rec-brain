import logging

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse

from app.api.dependencies import verify_internal_key
from app.schemas.enquiry import EnquiryRequest, EnquiryResponse
from app.services.enquiry_service import EnquiryService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/enquiry", tags=["enquiry"])


@router.post("/draft", dependencies=[Depends(verify_internal_key)])
async def enquiry_draft(
    enquriy: EnquiryRequest,
    request: Request,
) -> EnquiryResponse:
    service: EnquiryService = request.app.state.enquiry_service
    return await service.draft_response(enquriy)


@router.post("/draft/stream", dependencies=[Depends(verify_internal_key)])
async def enquiry_draft_stream(
    enquiry: EnquiryRequest,
    request: Request,
) -> StreamingResponse:
    """
    Streaming enquiry draft — returns pipeline step events via Server-Sent Events.

    SSE events:
        data: {"type": "step",   "step": "intent_classified", "label": "...", "meta": "billing_dispute · 18ms"}
        data: {"type": "step",   "step": "rag_retrieval",     "label": "...", "meta": "3 docs · 94ms"}
        data: {"type": "step",   "step": "llm_draft",         "label": "...", "meta": "gpt-4o-mini · 1.4s"}
        data: {"type": "step",   "step": "compliance_check",  "label": "...", "meta": "NSW tenancy rules"}
        data: {"type": "result", "draft": "...", "sources": [...]}
        data: {"type": "error",  "message": "..."}   (on failure)
        data: [DONE]
    """
    service: EnquiryService = request.app.state.enquiry_service
    return StreamingResponse(
        service.stream_draft_response(enquiry),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
