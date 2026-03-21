"""
app.agents.nodes.sql
~~~~~~~~~~~~~~~~~~~~
sql_search_node — handles search_listings and check_availability tool
calls via the SQL agent service.

Responsibility:
    - Resolve SqlAgentService from the FastAPI app state via RunnableConfig
    - Find every matching tool call on the latest AIMessage
    - Call search_listings.ainvoke for each one (sql_service injected,
      never exposed to the LLM)
    - Wrap each result as a ToolMessage (JSON-serialised)
    - Return {"messages": [ToolMessage, ...]}

Never calls the LLM — agent_node handles synthesis.
"""

import logging

from langchain_core.runnables import RunnableConfig

from app.agents.nodes._base import build_tool_message, error_content, last_ai_message
from app.agents.state import RealEstateAgentState
from app.services.sql_service import SqlAgentService
from app.tools.search_listings import search_listings

logger = logging.getLogger(__name__)

_HANDLED_TOOLS = frozenset({"search_listings", "check_availability"})


async def sql_search_node(
    state: RealEstateAgentState,
    runnable_config: RunnableConfig,
) -> dict:
    """
    Process all search_listings / check_availability tool calls on the
    latest AIMessage.

    Expects SqlAgentService to be available at:
        runnable_config["configurable"]["request"].app.state.sql_service

    Returns partial state: { messages: [ToolMessage, ...] }
    Returns {} if there is no AIMessage, no matching tool calls, or the
    service cannot be resolved.
    """
    last_ai = last_ai_message(state)
    if not last_ai:
        return {}

    tool_calls = [
        tc for tc in last_ai.tool_calls if tc["name"] in _HANDLED_TOOLS]
    if not tool_calls:
        return {}

    sql_service = _resolve_sql_service(runnable_config)
    if sql_service is None:
        # Logged inside _resolve_sql_service; return empty so safety_node
        # can increment error_count rather than crashing the graph.
        return {}

    tool_messages = []

    for tc in tool_calls:
        tool_name = tc["name"]
        question = tc["args"].get("question", "")
        tool_call_id = tc["id"]

        logger.info("sql_search_node | tool=%s | question=%.80s",
                    tool_name, question)

        try:
            # .ainvoke() takes a single input dict; sql_service is an
            # InjectedToolArg — pass it as a separate keyword so it is
            # never visible to the LLM.
            result = await search_listings.ainvoke(
                {"question": question},
                sql_service=sql_service,
            )
            content = (
                result
                if isinstance(result, dict)
                else {"success": True, "answer": str(result)}
            )
        except Exception as exc:
            logger.exception(
                "sql_search_node | failed | tool=%s | question=%.80s",
                tool_name,
                question,
            )
            content = error_content(exc)

        tool_messages.append(build_tool_message(
            tool_call_id, tool_name, content))

    return {"messages": tool_messages}


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _resolve_sql_service(config: RunnableConfig) -> SqlAgentService | None:
    """
    Extract SqlAgentService from the FastAPI request stored in RunnableConfig.

    Returns None and logs an error rather than raising, so the caller can
    decide how to handle the missing service.
    """
    try:
        request = config.get("configurable", {}).get("request")
        if request is None:
            raise ValueError("no 'request' key in configurable")
        return request.app.state.sql_service
    except Exception as exc:
        logger.error(
            "sql_search_node | could not resolve sql_service: %s", exc)
        return None
