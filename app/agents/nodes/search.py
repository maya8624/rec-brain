# app/agents/nodes/search.py
"""
Search nodes for the LangGraph agent.

    vector_search_node  — handles search_documents tool calls via Chroma
    sql_search_node     — handles search_listings / check_availability via SQL agent

Both nodes:
    - extract question from the LLM's tool call arguments
    - call the appropriate service
    - wrap results as ToolMessages so agent_node can read them
    - never call the LLM directly — agent_node handles synthesis
"""
import logging

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.runnables import RunnableConfig

from app.agents.state import RealEstateAgentState
from app.services.sql_service import SqlAgentService
from app.tools.search_listings import search_listings
from app.services.vector_search import perform_vector_search

logger = logging.getLogger(__name__)


# ------------------------------------
# vector_search_node
# ------------------------------------

async def vector_search_node(state: RealEstateAgentState) -> dict:
    """
    Handles search_documents tool calls.
    Calls perform_vector_search and wraps result as ToolMessage.
    """
    last_ai = _last_ai_message(state)
    if not last_ai:
        return {}

    tool_messages = []

    for tc in last_ai.tool_calls:
        if tc["name"] != "search_documents":
            continue

        question = tc["args"].get("question", "")
        tool_call_id = tc["id"]

        logger.info("vector_search_node | question=%s", question)

        try:
            result = await perform_vector_search(question)

            content = (
                {
                    "success": True,
                    "answer": result.get("answer", ""),
                    "source": result.get("source", "vector_db"),
                }
                if result
                else {"success": False, "error": "No results found"}
            )

        except Exception as e:
            logger.exception(
                "vector_search_node | failed | question=%s", question)
            content = {"success": False, "error": str(e)}

        tool_messages.append(
            ToolMessage(
                content=str(content),
                tool_call_id=tool_call_id,
                name="search_documents",
            )
        )

    return {"messages": tool_messages}


# ------------------------------------
# sql_search_node
# ------------------------------------

async def sql_search_node(state: RealEstateAgentState, runnable_config: RunnableConfig) -> dict:
    """
    Handles search_listings and check_availability tool calls.
    Injects sql_service into the tool via InjectedToolArg.
    """

    last_ai = _last_ai_message(state)
    if not last_ai:
        return {}

    http_request = runnable_config.get("configurable", {}).get("request")

    if http_request is None:
        logger.error("sql_search_node | no request in config")
        return {}

    sql_service: SqlAgentService = http_request.app.state.sql_service
    tool_messages = []

    for tc in last_ai.tool_calls:
        if tc["name"] not in ("search_listings", "check_availability"):
            continue

        question = tc["args"].get("question", "")
        tool_call_id = tc["id"]

        logger.info("sql_search_node | tool=%s | question=%s",
                    tc["name"], question)

        result = await search_listings.ainvoke(
            question=question,
            sql_service=sql_service,  # injected — LLM never sees this
        )

        tool_messages.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call_id,
                name=tc["name"],
            )
        )

        return {"messages": tool_messages}


# ── Private helper ─────────────────────────────────────────────────────────────

def _last_ai_message(state: RealEstateAgentState) -> AIMessage | None:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg
    return None


# How it works

# Both nodes follow the same pattern:
# ```
# last AIMessage
# │
# └── tool_calls loop
# │
# ├── extract question from tc["args"]["question"]
# ├── call service(perform_sql_search / perform_vector_search)
# ├── wrap result as {"success": True/False, "answer": ...}
# └── append ToolMessage with matching tool_call_id
