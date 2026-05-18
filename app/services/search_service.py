import logging

from langchain_core.messages import HumanMessage

from app.services.sql_service import SqlViewService
from app.services.rag_service import RagRetriever
from app.schemas.search import TenantPreference, PreferenceSearchResponse, SuburbSummaryResponse

logger = logging.getLogger(__name__)

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
    def __init__(self, sql: SqlViewService, rag: RagRetriever, llm) -> None:
        self._sql = sql
        self._rag = rag
        self._llm = llm

    async def search_by_preferences(self, pref: TenantPreference) -> PreferenceSearchResponse:
        query = _to_search_query(pref)
        result = await self._sql.search_listings(query)
        all_listings = result.output or []
        top = all_listings[:DISPLAY_COUNT]
        message = await self._sql.generate_summary(pref, top, len(all_listings))

        return PreferenceSearchResponse(
            message=message,
            listings=all_listings,
            display_count=len(top),
            total_count=len(all_listings),
            has_more=len(all_listings) > DISPLAY_COUNT,
        )

    async def get_suburb_summary(self, suburbs: list[str]) -> SuburbSummaryResponse:
        query = "suburb profile " + " ".join(suburbs) if suburbs else "suburb overview"
        nodes = await self._rag.aretrieve(query)

        if not nodes:
            return SuburbSummaryResponse(summary=None)

        context = "\n\n".join(n.node.get_content() for n in nodes)
        suburb_str = ", ".join(suburbs) if suburbs else "the suburb"

        prompt = (
            f"Summarise the following suburb profile for {suburb_str} in 2-3 sentences. "
            f"Focus on lifestyle, amenities, and rental appeal.\n\n{context}"
        )
        response = await self._llm.ainvoke([HumanMessage(content=prompt)])
        return SuburbSummaryResponse(summary=response.content.strip())
