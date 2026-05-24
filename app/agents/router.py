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
from app.agents.state import ConversationPhase, RealEstateAgentState
from app.core.constants import Intent, ToolNames, Node, StateKeys

logger = logging.getLogger(__name__)

TOOL_ROUTES: dict[str, str] = {
    ToolNames.CHECK_AVAILABILITY:  Node.TOOLS,
    ToolNames.BOOK_INSPECTION:     Node.TOOLS,
    ToolNames.CANCEL_INSPECTION:   Node.TOOLS,
    ToolNames.GET_BOOKING:         Node.TOOLS,
    ToolNames.GET_DEPOSIT:         Node.TOOLS,
}


def route_intent_output(state: RealEstateAgentState) -> str:
    """Booking/deposit pre-searches for listings unless search results are already shown."""
    if state.get(StateKeys.EARLY_RESPONSE):
        return Node.END

    intent = state.get(StateKeys.USER_INTENT, Intent.GENERAL)
    phase = state.get(StateKeys.PHASE, ConversationPhase.IDLE)

    if (intent in (Intent.BOOKING, Intent.DEPOSIT_PAYMENT) and phase != ConversationPhase.SEARCH_RESULTS_SHOWN):
        return Node.LISTING_SEARCH

    route = {
        Intent.SEARCH:          Node.LISTING_SEARCH,
        Intent.DOCUMENT_QUERY:  Node.VECTOR_SEARCH,
        Intent.HYBRID_SEARCH:   Node.HYBRID_SEARCH,
        Intent.SUBURB_SUMMARY:  Node.SUBURB_SUMMARY,
        Intent.BOOKING:         Node.AGENT,
        Intent.CANCELLATION:    Node.AGENT,
        Intent.BOOKING_LOOKUP:  Node.AGENT,
        Intent.DEPOSIT_PAYMENT: Node.AGENT,
        Intent.GENERAL:         Node.AGENT,
    }.get(intent, Node.AGENT)

    return route


def route_agent_output(state: RealEstateAgentState) -> str:
    """Routes to tools if the last AIMessage has tool calls, otherwise end."""
    if _requires_human(state, "route_agent_output"):
        return Node.END

    ai_message = _last_ai_message(state)
    if ai_message is None:
        return Node.END

    if not getattr(ai_message, "tool_calls", None):
        return Node.END

    for tool_call in ai_message.tool_calls:
        if destination := TOOL_ROUTES.get(tool_call["name"]):
            return destination

    logger.warning(
        "route_agent_output | unrecognised tools=%s → end",
        ai_message.tool_calls,
    )
    return Node.END


def route_after_search(state: RealEstateAgentState) -> str:
    if _requires_human(state, "route_after_search"):
        return Node.END
    return Node.AGENT


def route_after_tools(state: RealEstateAgentState) -> str:
    """Routes to context_update if any tool succeeded, safety if all failed or no results."""
    if _requires_human(state, "route_after_tools"):
        return Node.END

    tool_results = _parse_tool_messages(list(state["messages"]))

    if not tool_results:
        logger.warning("route_after_tools | no tool results → safety")
        return Node.SAFETY

    success_count = sum(1 for result in tool_results if result.get("success"))
    if success_count == 0:
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


def _parse_tool_messages(messages: list) -> list[dict]:
    results: list[dict] = []

    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            break

        if not isinstance(msg, ToolMessage):
            continue

        try:
            content = json.loads(msg.content)
            if isinstance(content, dict):
                content.setdefault("success", False)
                results.append(content)
            else:
                results.append({"success": False, "output": content})

        except (json.JSONDecodeError, TypeError):  # as exc:
            # logger.error("_parse_tool_messages | failed to parse ToolMessage: %s", exc)
            results.append({"success": False, "error": str(msg.content)})

    return results
