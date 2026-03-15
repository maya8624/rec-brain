"""
LangGraph node functions. Each node receives the full state,
does one focused job, and returns a PARTIAL state dict —
only the keys it actually changed.

Nodes in this graph:
    agent_node          — LLM decides: respond or call a tool
    context_update_node — extracts structured context from tool results
    safety_node         — checks error_count, sets requires_human if needed

LangGraph rule: nodes return dicts, not full state objects.
    correct:  return {"messages": [response]}
    wrong:    return state   (never mutate and return the whole state)
"""

import json
import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.state import (
    BookingContext,
    BookingStatus,
    PropertyContext,
    RealEstateAgentState,
    SearchContext,
    UserIntent,
)
from app.infrastructure.llm import get_llm
from app.prompts.system import REAL_ESTATE_AGENT_SYSTEM
from app.tools import get_all_tools

logger = logging.getLogger(__name__)

# Max consecutive tool errors before escalating to human
MAX_ERRORS_BEFORE_ESCALATION = 3

# ------------------------------------
# Intent classification)
# Keywords per intent — order matters, checked top to bottom.
# More specific intents listed first to avoid false matches.
# ------------------------------------
_INTENT_KEYWORDS: list[tuple[UserIntent, list[str]]] = [
    ("cancellation", [
        "cancel", "cancellation", "cancelled", "withdraw",
        "no longer", "don't want", "remove booking",
    ]),
    ("booking", [
        "book", "inspect", "inspection", "viewing", "view",
        "schedule", "arrange", "available", "availability",
        "when can i", "open for inspection", "open home",
    ]),
    ("document_query", [
        "lease", "contract", "strata", "terms", "clause",
        "bond", "deposit", "condition", "by-law", "bylaw",
        "pet policy", "break lease", "notice period",
        "landlord", "tenant", "agreement",
    ]),
    ("search", [
        "find", "search", "show", "list", "looking for",
        "properties", "house", "apartment", "unit", "townhouse",
        "bedroom", "bathroom", "suburb", "price", "budget",
        "under", "rent", "buy", "purchase",
    ]),
]


def _classify_intent(message: str) -> UserIntent:
    """
    Fast keyword-based intent classification. Zero LLM cost.

    Checks intents in priority order:
        cancellation -> booking -> document_query -> search -> general

    Cancellation checked first because "cancel my booking" contains
    booking keywords too — most-specific wins.

    Returns "general" for greetings, small talk, or anything unrecognised.
    The LLM still picks the actual tool — this is a hint for logging
    and future fast-path routing, not a hard constraint.
    """

    if not message:
        return "general"

    msg_lower = message.lower()

    for intent, keywords in _INTENT_KEYWORDS:
        if any(keyword in msg_lower for keyword in keywords):

            logger.debug(
                "_classify_intent | '%s...' -> %s",
                message[:40], intent
            )

            return intent

    return "general"


def _get_last_human_message(state: RealEstateAgentState) -> str:
    """
    Returns the content of the most recent HumanMessage in state.
    """
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else ""

    return ""


# ------------------------------------
# Node: agent
# ------------------------------------
def agent_node(state: RealEstateAgentState) -> dict:
    """
    Primary decision node — the LLM brain of the agent.

    Receives the full conversation history and decides one of:
        A) Call one or more tools  -> LangGraph routes to tool node
        B) Respond directly        -> LangGraph routes to END

    Also classifies user intent on every turn via _classify_intent()
    The LLM still picks the actual tool — intent is a hint, not a constraint.

    Returns partial state: messages + user_intent.
    """
    llm = get_llm()
    tools = get_all_tools()
    llm_with_tools = llm.bind_tools(tools)

    # Classify intent from latest user message — no LLM call, just keywords
    last_human_msg = _get_last_human_message(state)
    intent = _classify_intent(last_human_msg)

    # System prompt always prepended — never stored in state
    # (storing it would inflate the checkpointer DB unnecessarily)
    messages = [SystemMessage(
        content=REAL_ESTATE_AGENT_SYSTEM)] + list(state["messages"])

    logger.info(
        "agent_node | intent=%s | messages=%d | errors=%d",
        intent,
        len(state["messages"]),
        state.get("error_count", 0)
    )

    response = llm_with_tools.invoke(messages)

    # Log which tools the LLM decided to call (if any)
    if hasattr(response, "tool_calls") and response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        logger.info("agent_node | tool_calls=%s", tool_names)

    # Return both the LLM response AND the detected intent
    return {
        "messages": [response],
        "user_intent": intent,
    }


# ── Node: context update ───────────────────────────────────────────────────────

def context_update_node(state: RealEstateAgentState) -> dict:
    """
    Runs after the tool node. Inspects tool results and extracts
    structured context into the typed state fields.

    Why this exists:
        Tool results come back as raw JSON strings in ToolMessage.content.
        This node parses them and updates the strongly-typed context dicts
        so the router and subsequent agent turns have clean structured data.

    Returns only the context fields that actually changed.
    If no tool results found, returns empty dict (no-op).
    """

    messages = list(state["messages"])
    recent_tool_results = _extract_recent_tool_results(messages)

    if not recent_tool_results:
        logger.debug("context_update_node | no tool results found")
        return {}

    updates: dict[str, Any] = {}

    for tool_name, result in recent_tool_results:
        logger.info("context_update_node | processing tool=%s", tool_name)

        if tool_name == "search_listings":
            search_ctx = _extract_search_context(state, result)
            if search_ctx:
                updates["search_context"] = search_ctx

        elif tool_name == "get_property_details":
            prop_ctx = _extract_property_context(result)
            if prop_ctx:
                updates["property_context"] = prop_ctx

        elif tool_name == "check_inspection_availability":
            booking_ctx = _extract_availability_context(state, result)
            if booking_ctx:
                updates["booking_context"] = booking_ctx

        elif tool_name == "book_inspection":
            booking_ctx, booking_status = _extract_booking_confirmed(
                state, result)
            if booking_ctx:
                updates["booking_context"] = booking_ctx
            if booking_status:
                updates["booking_status"] = booking_status

        elif tool_name == "cancel_inspection":
            booking_status = _extract_cancellation_status(state, result)
            if booking_status:
                updates["booking_status"] = booking_status

    # Reset error_count on any successful tool result
    if any(r.get("success") for _, r in recent_tool_results):
        updates["error_count"] = 0

    return updates


# ── Node: safety ───────────────────────────────────────────────────────────────

def safety_node(state: RealEstateAgentState) -> dict:
    """
    Runs after tool failures accumulate. Checks whether we have hit
    the error threshold and should escalate to a human agent.

    Increments error_count each call. When threshold is reached,
    sets requires_human=True which the router uses to end the turn
    gracefully rather than looping.
    """

    current_errors = state.get("error_count", 0)
    new_error_count = current_errors + 1

    logger.warning(
        "safety_node | error_count=%d -> %d",
        current_errors, new_error_count,
    )

    if new_error_count >= MAX_ERRORS_BEFORE_ESCALATION:
        logger.error(
            "safety_node | escalating to human — %d consecutive errors",
            new_error_count,
        )
        return {
            "error_count": new_error_count,
            "requires_human": True,
        }

    return {"error_count": new_error_count}


# ── Private helpers ────────────────────────────────────────────────────────────

def _extract_recent_tool_results(messages: list) -> list[tuple[str, dict]]:
    """
    Finds the most recent batch of ToolMessages and parses their content.

    LangGraph tool node produces ToolMessages directly after an AIMessage
    with tool_calls. We walk back from the end of the message list to find
    the latest batch.

    Returns list of (tool_name, parsed_result_dict) tuples.
    Returns empty list if no ToolMessages found or parsing fails.
    """

    results = []

    for message in reversed(messages):
        if isinstance(message, AIMessage):
            break
        if isinstance(message, ToolMessage):
            try:
                content = json.loads(message.content) if isinstance(
                    message.content, str) else message.content

                results.append((message.name, content))

            except (json.JSONDecodeError, TypeError):
                results.append(
                    (message.name, {"output": message.content, "success": True}))

    return results


def _extract_search_context(state: RealEstateAgentState, result: dict) -> SearchContext | None:
    """Merge search result count into existing search_context."""

    if not result.get("success"):
        return None

    existing = dict(state.get("search_context", {}))
    existing["last_result_count"] = result.get("result_count", 0)
    return SearchContext(**existing)


def _extract_property_context(result: dict) -> PropertyContext | None:
    """
    Build PropertyContext from get_property_details tool result.
    The tool returns a structured dict matching PropertyContext fields.
    """
    if not result.get("success"):
        return None

    data = result.get("property", {})
    if not data:
        return None

    return PropertyContext(
        property_id=data.get("property_id", ""),
        address=data.get("address", ""),
        suburb=data.get("suburb", ""),
        price=data.get("price", 0.0),
        bedrooms=data.get("bedrooms", 0),
        bathrooms=data.get("bathrooms", 0),
        property_type=data.get("property_type", ""),
        agent_id=data.get("agent_id", ""),
        agent_name=data.get("agent_name", ""),
        agent_phone=data.get("agent_phone", ""),
    )


def _extract_availability_context(
    state: RealEstateAgentState,
    result: dict,
) -> BookingContext | None:
    """
    Merge available_slots from .NET availability API result
    into existing booking_context.
    """
    if not result.get("success"):
        return None

    slots = result.get("available_slots", [])
    if not slots:
        return None

    existing = dict(state.get("booking_context", {}))
    existing["available_slots"] = slots

    if result.get("property_id") and not existing.get("property_id"):
        existing["property_id"] = result["property_id"]

    return BookingContext(**existing)


def _extract_booking_confirmed(
    state: RealEstateAgentState,
    result: dict,
) -> tuple[BookingContext | None, BookingStatus | None]:
    """
    After a successful book_inspection tool call, update booking_context
    with the confirmation details from .NET and flip booking_status flags.
    """
    if not result.get("success"):
        return None, None

    existing = dict(state.get("booking_context", {}))
    existing["confirmation_id"] = result.get("confirmation_id", "")
    existing["confirmed_datetime"] = result.get("confirmed_datetime", "")

    booking_ctx = BookingContext(**existing)
    booking_status = BookingStatus(
        awaiting_confirmation=False,
        confirmed=True,
        cancelled=False,
    )

    return booking_ctx, booking_status


def _extract_cancellation_status(
    state: RealEstateAgentState,
    result: dict,
) -> BookingStatus | None:
    """After cancel_inspection succeeds, flip booking_status to cancelled."""
    if not result.get("success"):
        return None

    return BookingStatus(
        awaiting_confirmation=False,
        confirmed=False,
        cancelled=True,
    )
