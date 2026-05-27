import logging

from fastapi import APIRouter, Depends, Request

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
