"""
listing_search_node — handles "search" and "search_then_book" intents directly.

Responsibility:
    - Resolve SqlViewService from FastAPI app state via RunnableConfig
    - Extract question and search context from state
    - Run SQL search via context path (no LLM) or LLM-generated SQL
    - Returns { messages: [AIMessage], search_results, retrieved_docs }
      or { search_results, retrieved_docs, user_intent } for search_then_book

Never calls the LLM for formatting — reply is built in code.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import (
    format_search_reply,
    last_human_message,
    resolve_app_service,
    search_error_response,
    slim_rows,
)
from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys, Messages, Node, StateKeys
from app.services.sql_service import SqlViewService

logger = logging.getLogger(__name__)


async def listing_search_node(
        state: RealEstateAgentState,
        config: RunnableConfig) -> dict[str, Any]:
    """
    Direct listing search — no tool calls, no LLM formatting.
    """
    question = last_human_message(state)
    if not question:
        logger.warning("listing_search_node | no human message found")
        return {}

    sql_service: SqlViewService = resolve_app_service(
        config, AppStateKeys.SQL_VIEW_SERVICE, Node.LISTING_SEARCH
    )

    try:
        ctx = state.get(StateKeys.SEARCH_CONTEXT)
        if ctx and (ctx.get("property_id") or ctx.get("location") or ctx.get("address")):
            result = await sql_service.search_from_context(ctx)
        else:
            result = await sql_service.search_listings(question)

        rows = slim_rows(result.output or [])

        if state.get(StateKeys.USER_INTENT) == "search_then_book" and rows:
            return {
                StateKeys.SEARCH_RESULTS: rows,
                StateKeys.RETRIEVED_DOCS: None,
                StateKeys.USER_INTENT: "booking",
            }

        if rows:
            reply = format_search_reply(rows, result.result_count)
        else:
            reply = Messages.NO_RESULTS

        return {
            "messages": [AIMessage(content=reply)],
            StateKeys.SEARCH_RESULTS: rows,
            StateKeys.RETRIEVED_DOCS: None,
        }
    except Exception as exc:
        logger.exception("listing_search_node | failed | %s", exc)
        return search_error_response()
