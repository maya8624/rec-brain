"""
listing_search_node — handles "search" intent directly.

Flow:
    1. Extract the latest HumanMessage from state
    2. Send to SqlViewService — LLM generates SQL, runs against v_listings
    3. Store results in state as SystemMessage for agent_node to format
    4. agent_node (_LLM_PLAIN) formats results into human-readable response

No tool calls involved — bypasses LLM tool-calling entirely.
"""

import json
import logging
from typing import Any

from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import last_human_message, resolve_app_service
from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys, Node

logger = logging.getLogger(__name__)


async def listing_search_node(state: RealEstateAgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Direct listing search — no tool calls.

    Gets user question, passes to SqlViewService which generates
    and executes SQL against v_listings, stores raw results in
    state for agent_node to format into a human-readable response.
    """
    question = last_human_message(state)
    if not question:
        logger.warning("listing_search_node | no human message found")
        return {}

    sql_service = resolve_app_service(
        config, AppStateKeys.SQL_VIEW_SERVICE, Node.LISTING_SEARCH
    )

    if sql_service is None:
        return {}

    try:
        result = await sql_service.search_listings(question)

        logger.info(
            "listing_search_node | success=%s | count=%d",
            result.get("success"),
            result.get("result_count", 0),
        )

        result_message = SystemMessage(content=json.dumps({
            "search_results": result.get("output"),
            "result_count": result.get("result_count", 0),
            "success": result.get("success"),
            "error": result.get("error"),
        }))

        return {"messages": [result_message]}

    except Exception as exc:
        logger.exception("listing_search_node | failed | %s", exc)
        return {}
