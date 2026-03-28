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

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from app.agents.state import RealEstateAgentState
from app.services.sql_search import SqlViewService

logger = logging.getLogger(__name__)


async def listing_search_node(
    state: RealEstateAgentState,
    runnable_config: RunnableConfig,
) -> dict[str, Any]:
    """
    Direct listing search — no tool calls.

    Gets user question, passes to SqlViewService which generates
    and executes SQL against v_listings, stores raw results in
    state for agent_node to format into a human-readable response.
    """
    question = _get_last_human_message(state)
    if not question:
        logger.warning("listing_search_node | no human message found")
        return {}

    sql_view_service = _resolve_sql_view_service(runnable_config)
    if sql_view_service is None:
        return {}

    try:
        result = await sql_view_service.search_listings(question)

        logger.info(
            "listing_search_node | success=%s | count=%d",
            result.get("success"),
            result.get("result_count", 0),
        )

        # Store as SystemMessage so agent_node can read and format it
        # SystemMessage distinguishes it from user HumanMessages in state
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


def _get_last_human_message(state: RealEstateAgentState) -> str:
    """Return the content of the most recent HumanMessage in state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else ""
    return ""


def _resolve_sql_view_service(config: RunnableConfig) -> SqlViewService | None:
    """
    Extract SqlViewService from FastAPI request in RunnableConfig.
    Returns None and logs error rather than raising.
    """
    try:
        request = config.get("configurable", {}).get("request")
        if request is None:
            raise ValueError("no 'request' key in configurable")
        return request.app.state.sql_view_service
    except Exception as exc:
        logger.error(
            "listing_search_node | could not resolve sql_view_service: %s", exc
        )
        return None
