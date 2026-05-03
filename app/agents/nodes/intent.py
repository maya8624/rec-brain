"""
intent_node — classifies user intent before routing through the graph.

Strategy: hybrid keyword + LLM

    Fast path (keyword, no LLM):
        - Obvious cancellation  → "cancellation"
        - Obvious booking only  → "booking"

    LLM path (everything else):
        - Ambiguous, follow-up, compound, search queries
        - Returns IntentClassification with intent + extracted entities
        - Entities written to state["search_context"] for downstream nodes
"""

import logging
import re
from typing import Any
from langchain_core.messages import HumanMessage, SystemMessage
from app.agents.nodes._base import last_human_message
from app.agents.state import IntentClassification, RealEstateAgentState, UserIntent
from app.core.constants import StateKeys
from app.infrastructure.llm import get_llm
from app.prompts.intent import INTENT_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)

# ------------------------------------
#  Keyword sets for the fast path
# ------------------------------------
_CANCELLATION_KEYWORDS = frozenset([
    "cancel", "cancellation", "cancelled", "withdraw",
    "remove booking", "no longer want to attend", "no longer available",
    "don't want to attend", "don't want the booking", "don't want the inspection",
])

_BOOKING_KEYWORDS = frozenset([
    "book", "viewing", "view the property",
    "schedule", "arrange", "open for inspection", "open home",
])

_LOOKUP_KEYWORDS = frozenset([
    "my booking", "my inspection", "check my booking", "check booking",
    "booking details", "booking status", "when is my inspection",
    "what time is my", "show my booking", "my confirmation",
    "look up my booking", "find my booking",
    "see my booking", "see my inspection", "view my booking", "view my inspection",
    "booked an inspection", "booked a viewing", "i booked",
])

_SEARCH_KEYWORDS = frozenset([
    "find", "search", "show", "list", "looking for",
    "properties", "house", "apartment", "unit", "townhouse",
    "bedroom", "bathroom", "suburb", "price", "budget",
    "under", "rent for", "for rent", "to rent", "buy", "purchase",
])

_LLM_HISTORY_LIMIT = 4


async def intent_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Classifies intent from the latest HumanMessage.
    """
    message = last_human_message(state).lower()
    if not message:
        return {StateKeys.USER_INTENT: "general"}

    obvious = _obvious_intent(message)
    if obvious:
        return {StateKeys.USER_INTENT: obvious}

    if _is_booking_continuation(state, message):
        return {StateKeys.USER_INTENT: "booking"}

    return await _classify_with_llm(state)


async def _classify_with_llm(state: RealEstateAgentState) -> dict[str, Any]:
    """Invoke the LLM to classify intent and extract search entities."""
    intent_completed = state.get(StateKeys.INTENT_COMPLETED, False)
    if intent_completed:
        history = state["messages"][-1:]
    else:
        history = [message for message in state["messages"] if isinstance(
            message, HumanMessage)][-_LLM_HISTORY_LIMIT:]

    state_hint = _build_state_hint(
        state.get(StateKeys.LAST_INTENT), intent_completed)

    prompt = [SystemMessage(
        content=INTENT_CLASSIFICATION_PROMPT + state_hint), *history]

    try:
        llm = get_llm().with_structured_output(IntentClassification)
        classification: IntentClassification = await llm.ainvoke(prompt)
    except Exception as exc:
        logger.error("intent_node | LLM classification failed: %s", exc)
        return {StateKeys.USER_INTENT: "general"}

    result = _apply_search_context(state, classification)
    return result


def _obvious_intent(msg_lower: str) -> UserIntent | None:
    """
    Returns a high-confidence intent without an LLM call, or None if ambiguous.

    Cancellation: very distinctive vocabulary, never overlaps with other intents.
    Booking:      only when no search keywords present — avoids missing search_then_book.
    """
    if (_matches_keywords(msg_lower, _CANCELLATION_KEYWORDS) and
            not _matches_keywords(msg_lower, _SEARCH_KEYWORDS)):
        return "cancellation"

    if (_matches_keywords(msg_lower, _LOOKUP_KEYWORDS) and
            not _matches_keywords(msg_lower, _CANCELLATION_KEYWORDS)):
        return "booking_lookup"

    if (_matches_keywords(msg_lower, _BOOKING_KEYWORDS) and
            not _matches_keywords(msg_lower, _SEARCH_KEYWORDS)):
        return "booking"

    return None


def _is_booking_continuation(state: RealEstateAgentState, message: str) -> bool:
    """True when slots are pending and the user is selecting one, not searching or cancelling."""
    booking_ctx = state.get(StateKeys.BOOKING_CONTEXT)
    if not booking_ctx or not booking_ctx.get("available_slots"):
        return False

    # Booking already completed or cancelled — slots are no longer pending
    booking_status = state.get(StateKeys.BOOKING_STATUS) or {}
    if booking_status.get("confirmed") or booking_status.get("cancelled"):
        return False

    return (
        not _matches_keywords(message, _SEARCH_KEYWORDS)
        and not _matches_keywords(message, _CANCELLATION_KEYWORDS)
    )


def _apply_search_context(
        state: RealEstateAgentState,
        result: IntentClassification) -> dict[str, Any]:
    """
    Build the state updates dict, 
    merging or replacing search_context based on extracted entities.
    """
    updates: dict[str, Any] = {
        StateKeys.USER_INTENT:      result.intent,
        StateKeys.EARLY_RESPONSE:   result.early_response,
        StateKeys.LAST_INTENT:      result.intent,
        StateKeys.INTENT_COMPLETED: False,
    }

    if result.intent == "general":
        return updates

    entities = _extract_entities(result)
    if not entities:
        return updates

    if entities.get("location"):
        updates[StateKeys.SEARCH_CONTEXT] = entities
    else:
        existing = dict(state.get(StateKeys.SEARCH_CONTEXT) or {})
        updates[StateKeys.SEARCH_CONTEXT] = {**existing, **entities}

    return updates


def _build_state_hint(last_intent: str | None, intent_completed: bool) -> str:
    """Return a [STATE] hint string for the LLM prompt, or '' if no prior intent."""
    if not last_intent:
        return ""
    return (
        f"\n[STATE] Previous intent: {last_intent}, "
        f"Completed: {intent_completed}. "
        f"If Completed is True, treat the current message as a fresh request."
    )


def _extract_entities(result: IntentClassification) -> dict:
    """Build a partial SearchContext dict from LLM-extracted entities."""
    entities = {}
    if result.location:
        entities["location"] = result.location
    if result.address:
        entities["address"] = result.address
    if result.listing_type:
        entities["listing_type"] = result.listing_type
    if result.property_type:
        entities["property_type"] = result.property_type
    if result.bedrooms is not None:
        entities["bedrooms"] = result.bedrooms
    if result.bathrooms is not None:
        entities["bathrooms"] = result.bathrooms
    if result.max_price is not None:
        entities["max_price"] = result.max_price
    if result.min_price is not None:
        entities["min_price"] = result.min_price
    if result.limit is not None:
        entities["limit"] = min(result.limit, 10)
    return entities


def _matches_keywords(msg_lower: str, keywords: frozenset[str]) -> bool:
    """Whole-word/phrase keyword match — prevents substring false positives."""
    for keyword in keywords:
        if re.search(rf'\b{re.escape(keyword)}\b', msg_lower):
            return True
    return False
