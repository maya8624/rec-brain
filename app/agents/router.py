"""
Conditional edge functions for the LangGraph graph.

Router map:
    route_intent_output  → after intent_node → listing_search | vector_search | agent | end
    route_agent_output   → after agent_node → tools | end
    route_after_search   → after listing_search_node or vector_search_node → agent | end
    route_after_tools    → after tool_node → context_update | safety | end
    route_after_context  → after context_update_node → agent | end
    route_after_safety   → after safety_node → agent | end
"""

import json
import logging
from langchain_core.messages import AIMessage, ToolMessage
from app.agents.state import RealEstateAgentState
from app.core.constants import ToolNames, Node, StateKeys

logger = logging.getLogger(__name__)

TOOL_ROUTES: dict[str, str] = {
    ToolNames.CHECK_AVAILABILITY:  Node.TOOLS,
    ToolNames.BOOK_INSPECTION:     Node.TOOLS,
    ToolNames.CANCEL_INSPECTION:   Node.TOOLS,
    ToolNames.GET_BOOKING: Node.TOOLS,
}


def route_intent_output(state: RealEstateAgentState) -> str:
    """
    Routes after intent_node based on user_intent in state.
    """
    if state.get(StateKeys.EARLY_RESPONSE):
        return Node.END

    intent = state.get(StateKeys.USER_INTENT, "general")

    route = {
        "search":           Node.LISTING_SEARCH,
        "search_then_book": Node.LISTING_SEARCH,
        "document_query":   Node.VECTOR_SEARCH,
        "hybrid_search":    Node.HYBRID_SEARCH,
        "booking":          Node.AGENT,
        "cancellation":     Node.AGENT,
        "booking_lookup":   Node.AGENT,
        "general":          Node.AGENT,
    }.get(intent, Node.AGENT)

    return route


def route_agent_output(state: RealEstateAgentState) -> str:
    """
    Reads tool_calls from the last AIMessage and routes to:
        "tools" — LLM called check_availability / book_inspection / cancel_inspection
        "end"   — plain text response or escalation
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
                destination,
            )
            return destination

    logger.warning(
        "route_agent_output | unrecognised tools=%s → end",
        message.tool_calls,
    )
    return Node.END


def route_after_search(state: RealEstateAgentState) -> str:
    """
    Routes after listing_search_node / vector_search_node / hybrid_search_node.
    """
    if _requires_human(state, "route_after_search"):
        return Node.END

    last = state["messages"][-1] if state["messages"] else None
    if isinstance(last, AIMessage):
        return Node.END

    return Node.AGENT


def route_after_tools(state: RealEstateAgentState) -> str:
    """
    Inspects ToolMessage results from the current turn and routes to:
        "context_update" — at least one tool succeeded
        "safety"         — no results or all tools failed
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
            len(results),
        )
        return Node.SAFETY

    return Node.CONTEXT_UPDATE


def route_after_context(state: RealEstateAgentState) -> str:
    """Returns to agent so LLM can respond using updated context."""
    if _requires_human(state, "route_after_context"):
        return Node.END
    return Node.AGENT


def route_after_safety(state: RealEstateAgentState) -> str:
    """Returns to agent if safety threshold has not been reached."""
    if _requires_human(state, "route_after_safety"):
        return Node.END
    return Node.AGENT


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _requires_human(state: RealEstateAgentState, caller: str) -> bool:
    if state.get(StateKeys.REQUIRES_HUMAN):
        logger.warning("%s | requires_human=True → end", caller)
        return True
    return False


def _last_ai_message(state: RealEstateAgentState) -> AIMessage | None:
    for msg in reversed(state["messages"]):
        if isinstance(msg, AIMessage):
            return msg
    return None


def _extract_tool_results(messages: list) -> list[dict]:
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
                "_extract_tool_results | failed to parse ToolMessage: %s", exc
            )
            results.append({"success": False, "error": str(msg.content)})

    return results
