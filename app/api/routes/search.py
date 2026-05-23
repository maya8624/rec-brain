import logging

from fastapi import APIRouter, Depends, Request

from app.api.dependencies import verify_internal_key
from app.schemas.search import TenantPreference, PreferenceSearchResponse, SuburbSummaryRequest, SuburbSummaryResponse, TenancyDocsRequest, TenancyDocsResponse
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/search", tags=["search"])


@router.post("/preferences", dependencies=[Depends(verify_internal_key)])
async def search_by_preferences(
    pref: TenantPreference,
    request: Request,
) -> PreferenceSearchResponse:
    service: SearchService = request.app.state.search_service
    results = await service.search_by_preferences(pref)
    return results


@router.post("/suburb-summary", dependencies=[Depends(verify_internal_key)])
async def suburb_summary(
    body: SuburbSummaryRequest,
    request: Request,
) -> SuburbSummaryResponse:
    service: SearchService = request.app.state.search_service
    return await service.get_suburb_summary(body.suburbs)


@router.post("/tenancy-docs", dependencies=[Depends(verify_internal_key)])
async def tenancy_docs(
    body: TenancyDocsRequest,
    request: Request,
) -> TenancyDocsResponse:
    service: SearchService = request.app.state.search_service
    return await service.get_tenancy_docs(body.property_id, body.tenant_id)
