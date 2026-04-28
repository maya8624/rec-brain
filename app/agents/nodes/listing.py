"""
listing_search_node — handles "search" intent directly.

Flow:
    1. Extract the latest HumanMessage from state
    2. Send to SqlViewService — LLM generates SQL, runs against v_listings
    3. Format results in code and append AIMessage directly to messages
       (route_after_search then routes to END, skipping agent_node)

No tool calls involved — bypasses LLM tool-calling entirely.
"""

import logging
from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import (
    format_search_reply,
    last_human_message,
    resolve_app_service,
    slim_rows,
)
from app.agents.state import RealEstateAgentState
from app.core.constants import AppStateKeys, Node
from app.services.sql_service import SqlViewService

logger = logging.getLogger(__name__)


async def listing_search_node(state: RealEstateAgentState, config: RunnableConfig) -> dict[str, Any]:
    """
    Direct listing search — no tool calls, no LLM formatting.

    Gets user question, passes to SqlViewService which generates
    and executes SQL against v_listings, then formats the reply
    in code and appends it as an AIMessage so route_after_search
    can skip agent_node entirely.
    """
    question = last_human_message(state)
    if not question:
        logger.warning("listing_search_node | no human message found")
        return {}

    sql_service: SqlViewService | None = resolve_app_service(
        config, AppStateKeys.SQL_VIEW_SERVICE, Node.LISTING_SEARCH
    )

    if sql_service is None:
        return {}

    try:
        ctx = state.get("search_context") or {}

        if ctx.get("property_id") or ctx.get("location") or ctx.get("address"):
            # Fast path — entities already extracted by intent_node, no LLM call
            result = await sql_service.search_from_context(ctx)
            logger.info(
                "listing_search_node | context-path | count=%d",
                result.get("result_count", 0),
            )
        else:
            # Fallback — complex query, let LLM generate SQL
            result = await sql_service.search_listings(question)
            logger.info(
                "listing_search_node | llm-path | count=%d",
                result.get("result_count", 0),
            )

        count = result.get("result_count", 0)
        rows = slim_rows(result.get("output") or [])

        # For search_then_book: hand off to agent_node for check_availability
        # instead of ending. Only when results exist — no results falls through
        # to the normal "nothing found" reply so the user isn't left in silence.
        if state.get("user_intent") == "search_then_book" and rows:
            logger.info(
                "listing_search_node | search_then_book | found=%d → booking",
                count,
            )
            return {
                "search_results": rows,
                "retrieved_docs": None,
                "user_intent": "booking",
            }

        if rows:
            reply = format_search_reply(rows, count)
        else:
            reply = "No properties matched your search. Try broadening your criteria — for example, a nearby suburb or a higher price range."

        logger.info(
            "listing_search_node | reply built in code | count=%d", count)

        return {
            "messages": [AIMessage(content=reply)],
            "search_results": rows,
            "retrieved_docs": None,
        }

        # --- OLD LLM-based path (kept for reference) ---
        # if rows:
        #     summary = listing_summary(rows)
        #     pagination_note = (
        #         "\nNote: Only the top 10 results are shown. "
        #         "Tell the customer they can ask to see more by narrowing their search criteria."
        #         if count == 10 else ""
        #     )
        #     instruction = (
        #         f"[PROPERTY SEARCH RESULTS — {count} listing(s) found. "
        #         f"Format these for the customer using the FORMATTING SEARCH RESULTS rules. "
        #         f"Present the results only. "
        #         f"Do NOT ask if they want to book an inspection. "
        #         f"Do NOT add any follow-up questions or offers. "
        #         f"Stop after presenting the results.]"
        #     )
        #     retrieved_docs = f"{instruction}\n{summary}{pagination_note}"
        # else:
        #     summary = result.get("error", "No results found.")
        #     retrieved_docs = (
        #         f"[PROPERTY SEARCH RESULTS — 0 listing(s) found. "
        #         f"IMPORTANT: Do NOT reference or repeat any listings from previous responses. "
        #         f"Tell the customer no properties matched their search and suggest broadening the criteria.]\n"
        #         f"{summary}"
        #     )
        # return {
        #     "retrieved_docs": retrieved_docs,
        #     "search_results": rows,
        # }

    except Exception as exc:
        logger.exception("listing_search_node | failed | %s", exc)
        return {}
