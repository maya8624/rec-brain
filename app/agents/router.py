"""
Conditional edge functions for the LangGraph graph.

Router map:
    route_agent_output    → after agent_node → vector_search | sql_search | tools | end
    route_after_search    → after vector_search_node or sql_search_node → agent | end
    route_after_tools     → after tool_node → context_update | safety | end
    route_after_context   → after context_update_node → agent | end
    route_after_safety    → after safety_node → agent | end
"""

import json
import logging
from langchain_core.messages import AIMessage, ToolMessage
from app.agents.state import RealEstateAgentState
from app.core.constants import ToolNames, Node

logger = logging.getLogger(__name__)

TOOL_ROUTES: dict[str, str] = {
    ToolNames.SEARCH_DOCUMENTS: Node.VECTOR_SEARCH,
    ToolNames.SEARCH_LISTINGS:   Node.SQL_SEARCH,
    ToolNames.CHECK_AVAILABILITY: Node.SQL_SEARCH,
    ToolNames.BOOK_INSPECTION:   Node.TOOLS,
    ToolNames.CANCEL_INSPECTION: Node.TOOLS,
}


def route_agent_output(state: RealEstateAgentState) -> str:
    """
    Reads tool_calls from the last AIMessage and routes to:
        "vector_search" — LLM called search_documents
        "sql_search"    — LLM called search_listings / check_availability
        "tools"         — LLM called book_inspection / cancel_inspection
        "end"           — plain text response or escalation
    """
    if _requires_human(state, "route_agent_output"):
        return Node.END

    message = _last_ai_message(state)
    if message is None:
        logger.warning("route_agent_output | no AI message → end")
        return Node.END

    if not getattr(message, "tool_calls", None):
        logger.info("route_agent_output | plain response → end")
        return Node.END

    for tool_call in message.tool_calls:
        if destination := TOOL_ROUTES.get(tool_call["name"]):

            logger.info(
                "route_agent_output | tool=%s → %s",
                tool_call["name"],
                destination
            )
            return destination

    logger.warning(
        "route_agent_output | unrecognised tools=%s → end",
        message.tool_calls
    )

    return Node.END


def route_after_search(state: RealEstateAgentState) -> str:
    """
    Always returns to agent so LLM can formulate a response from the search results.
    """
    if _requires_human(state, "route_after_search"):
        return Node.END

    logger.info("route_after_search | → agent")
    return Node.AGENT


def route_after_tools(state: RealEstateAgentState) -> str:
    """
    Inspects ToolMessage results from the current turn and routes to:
        "context_update" — at least one tool succeeded
        "safety"         — no results found, or all tools failed
    """
    if _requires_human(state, "route_after_tools"):
        return Node.END

    results = _extract_tool_results(list(state["messages"]))

    if not results:
        logger.warning("route_after_tools | no tool results → safety")
        return Node.SAFETY

    succeeded = sum(1 for result in results if result.get("success"))

    if succeeded == 0:
        logger.warning(
            "route_after_tools | all %d tool(s) failed → safety",
            len(results)
        )
        return Node.SAFETY

    logger.info(
        "route_after_tools | %d/%d succeeded → context_update",
        succeeded,
        len(results)
    )

    return Node.CONTEXT_UPDATE


def route_after_context(state: RealEstateAgentState) -> str:
    """Returns to agent so the LLM can respond using the updated context."""
    if _requires_human(state, "route_after_context"):
        return Node.END

    logger.info("route_after_context | → agent")
    return Node.AGENT


def route_after_safety(state: RealEstateAgentState) -> str:
    """Returns to agent if the safety threshold has not been reached."""
    if _requires_human(state, "route_after_safety"):
        return Node.END

    logger.info("route_after_safety | under threshold → agent")
    return Node.AGENT


# ------------------------
# Private helpers
# ------------------------
def _requires_human(state: RealEstateAgentState, caller: str) -> bool:
    """Return True and log if the escalation flag is set."""
    if state.get("requires_human"):
        logger.warning("%s | requires_human=True → end", caller)
        return True

    return False


def _last_ai_message(state: RealEstateAgentState) -> AIMessage | None:
    """Return the most recent AIMessage in state, or None if absent."""
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _extract_tool_results(messages: list) -> list[dict]:
    """
    Parse the most recent batch of ToolMessages.
    Stops at the first AIMessage — only reads the latest batch.
    """
    results: list[dict] = []

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            break

        if not isinstance(msg, ToolMessage):
            continue

        try:
            content = (
                json.loads(msg.content)
                if isinstance(msg.content, str)
                else msg.content
            )

            if isinstance(content, dict):
                content.setdefault("success", False)
                results.append(content)
            else:
                results.append({"success": False, "output": content})

        except (json.JSONDecodeError, TypeError) as exc:
            logger.error(
                "_extract_tool_results | failed to parse ToolMessage: %s",
                exc
            )
            results.append({"success": False, "error": str(msg.content)})

    return results
