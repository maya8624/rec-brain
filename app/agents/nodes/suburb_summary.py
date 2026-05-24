"""
suburb_summary_node — retrieves and formats suburb summaries from RAG.

Called when intent = 'suburb_summary'.
Calls SearchService.get_suburb_summary() directly — no agent_node needed.
Adds the formatted text as an AIMessage so follow-up questions have conversation context.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys, StateKeys
from app.schemas.search import SuburbSummaryResponse
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


async def suburb_summary_node(state: RealEstateAgentState, config: RunnableConfig) -> dict[str, Any]:
    """Fetches structured suburb summary and sets retrieved_docs for agent_node to stream."""
    suburbs = config.get(AppStateKeys.CONFIGURABLE, {}).get(AppStateKeys.SUBURBS)
    if not suburbs:
        location = (state.get(StateKeys.SEARCH_CONTEXT) or {}).get("location")
        if not location:
            logger.warning("suburb_summary_node | no suburbs in config or state")
            return {}
        suburbs = [location]

    try:
        service: SearchService = config.get(AppStateKeys.CONFIGURABLE, {}).get(AppStateKeys.SEARCH_SERVICE)
        summary: SuburbSummaryResponse = await service.get_suburb_summary(suburbs)
    except Exception as exc:
        logger.exception("suburb_summary_node | failed | %s", exc)
        return {}

    return {
        StateKeys.SUBURB_SUMMARY_RESULT: summary.model_dump(),
        StateKeys.RETRIEVED_DOCS: summary.model_dump_json(),
    }
