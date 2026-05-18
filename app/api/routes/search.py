import logging

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import verify_internal_key
from app.schemas.search import TenantPreference, PreferenceSearchResponse, SuburbSummaryRequest, SuburbSummaryResponse
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/preferences", dependencies=[Depends(verify_internal_key)])
async def search_by_preferences(
    pref: TenantPreference,
    request: Request,
) -> PreferenceSearchResponse:
    service: SearchService = request.app.state.search_service
    return await service.search_by_preferences(pref)


@router.post("/suburb-summary", dependencies=[Depends(verify_internal_key)])
async def suburb_summary(
    body: SuburbSummaryRequest,
    request: Request,
) -> SuburbSummaryResponse:
    service: SearchService = request.app.state.search_service
    return await service.get_suburb_summary(body.suburbs)
