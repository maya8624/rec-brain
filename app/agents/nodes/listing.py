"""
listing_search_node — handles "search" intent directly.

Flow:
    1. Extract the latest HumanMessage from state
    2. Send to SqlViewService — LLM generates SQL, runs against v_listings
    3. Store results in state as SystemMessage for agent_node to format
    4. agent_node (_LLM_PLAIN) formats results into human-readable response

No tool calls involved — bypasses LLM tool-calling entirely.
"""

import logging
from typing import Any

from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import last_human_message, listing_summary, resolve_app_service, slim_rows
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

        count = result.get("result_count", 0)
        logger.info(
            "listing_search_node | success=%s | count=%d",
            result.get("success"), count,
        )

        rows = slim_rows(result.get("output") or [])
        summary = listing_summary(rows) if rows else result.get("error", "No results found.")

        pagination_note = (
            "\nNote: Only the top 10 results are shown. "
            "Tell the customer they can ask to see more by narrowing their search criteria."
            if count == 10 else ""
        )
        retrieved_docs = (
            f"[PROPERTY SEARCH RESULTS — {count} listing(s) found. "
            f"Format these for the customer using the FORMATTING SEARCH RESULTS rules.]\n"
            f"{summary}"
            f"{pagination_note}"
        )

        return {
            "retrieved_docs": retrieved_docs,
            "search_results": rows,
        }

    except Exception as exc:
        logger.exception("listing_search_node | failed | %s", exc)
        return {}
