import asyncio
import structlog
from typing import Any

from fastapi import HTTPException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from app.services.sql_service import SqlViewService
from app.services.rag_service import RagRetriever
from app.prompts.sql import build_search_summary_prompt
from app.prompts.rag import build_suburb_summary_prompt, build_tenancy_details_prompt
from app.schemas.property import Listing
from app.schemas.search import (
    TenantPreference,
    PreferenceSearchResponse,
    SuburbSummaryResponse,
    TenancyDetails,
    TenancyDocsResponse
)

logger = structlog.get_logger(__name__)

DISPLAY_COUNT = 4


def _to_search_query(pref: TenantPreference) -> str:
    parts = []

    if pref.minBeds and pref.maxBeds and pref.minBeds != pref.maxBeds:
        parts.append(f"{pref.minBeds}-{pref.maxBeds} bedroom")
    elif pref.minBeds:
        parts.append(f"at least {pref.minBeds} bedroom")
    elif pref.maxBeds:
        parts.append(f"up to {pref.maxBeds} bedroom")

    if pref.petFriendly:
        parts.append("pet friendly")

    desc = (" ".join(parts) + " property") if parts else "property"

    if len(pref.suburbs) > 1:
        desc += f" in {', '.join(pref.suburbs[:-1])} or {pref.suburbs[-1]}"
    elif pref.suburbs:
        desc += f" in {pref.suburbs[0]}"

    if pref.maxRent is not None:
        desc += f" under ${pref.maxRent:.0f}/wk"

    if pref.availableWithinDays is not None:
        desc += f" available within {pref.availableWithinDays} days"

    return f"Find me a {desc}"


class SearchService:
    def __init__(self, sql: SqlViewService, rag: RagRetriever, llm: BaseChatModel) -> None:
        self._sql = sql
        self._rag = rag
        self._llm: BaseChatModel = llm

    async def search_by_preferences(self, pref: TenantPreference) -> PreferenceSearchResponse:
        query = _to_search_query(pref)
        result = await self._sql.search_listings(query)
        all_listings = [
            Listing.model_validate(result) for result in (result.output or [])
        ]
        total = len(all_listings)
        top = all_listings[:DISPLAY_COUNT]
        message = await self._generate_summary(pref, top, total)

        return PreferenceSearchResponse(
            message=message,
            listings=all_listings,
            display_count=len(top),
            total_count=total,
            has_more=total > DISPLAY_COUNT,
        )

    async def _generate_summary(
        self,
        pref: TenantPreference,
        top_listings: list[Listing],
        total: int,
    ) -> str:
        if len(pref.suburbs) > 1:
            suburb_str = ", ".join(
                pref.suburbs[:-1]) + f" and {pref.suburbs[-1]}"
        else:
            suburb_str = pref.suburbs[0] if pref.suburbs else "your preferred suburbs"

        summaries = "\n".join(
            f"- {listing.address_line1} {listing.suburb}, ${listing.price:.0f}/wk, {listing.bedrooms} bed"
            for listing in top_listings
        )

        prompt = build_search_summary_prompt(
            pref=pref,
            suburb_str=suburb_str,
            total=total,
            summaries=summaries,
        )
        response = await self._llm.ainvoke([HumanMessage(content=prompt)])
        return response.content.strip()

    async def get_suburb_summary(self, suburbs: list[str]) -> SuburbSummaryResponse:
        if not suburbs:
            return SuburbSummaryResponse()

        results = await asyncio.gather(
            *[self._rag.aretrieve(f"suburb summary {suburb}", doc_type="guide") for suburb in suburbs],
            return_exceptions=True,
        )

        context_blocks = []
        for suburb, nodes in zip(suburbs, results):
            if isinstance(nodes, Exception) or not nodes:
                continue
            content = "\n".join(n.node.get_content() for n in nodes)
            context_blocks.append(f"### {suburb}\n{content}")

        if not context_blocks:
            return SuburbSummaryResponse()

        suburb_str = ", ".join(suburbs)
        context = "\n\n".join(context_blocks)
        prompt = build_suburb_summary_prompt(suburb_str, context)

        structured_llm = self._llm.with_structured_output(SuburbSummaryResponse)
        result: SuburbSummaryResponse = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        return result

    async def get_tenancy_docs(self, property_id: str, user_id: str) -> TenancyDocsResponse:
        nodes = await self._rag.aretrieve(
            query="TENANCY DETAILS agreement type commencement rent bond",
            doc_type="contract",
            file_name="tenancy_agreement_47_Harrington_Street_Cronulla.pdf"
        )

        if not nodes:
            raise HTTPException(
                status_code=404, detail="Tenancy document not found")

        context = "\n".join(n.node.get_content() for n in nodes)
        prompt = build_tenancy_details_prompt(context)
        structured_llm = self._llm.with_structured_output(TenancyDetails)
        try:
            tenancy: TenancyDetails = await structured_llm.ainvoke([HumanMessage(content=prompt)])
        except Exception as exc:
            logger.exception("search_tenancy_llm_failed", error=str(exc))
            raise HTTPException(
                status_code=422, detail="Failed to extract tenancy details from document")
        return TenancyDocsResponse(tenancy=tenancy)
