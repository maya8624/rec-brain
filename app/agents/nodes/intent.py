"""
intent_node — classifies user intent before routing through the graph.

Strategy: hybrid keyword + LLM

    Fast path (keyword, no LLM):
        - Obvious cancellation      → "cancellation"
        - Obvious booking lookup    → "booking_lookup"
        - Obvious booking only      → "booking"
        - Deposit + search keywords → "search_then_deposit"
        - Deposit only              → "deposit_payment"
        - Slot selection in context → "booking" (continuation)

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

# Cancellation vocabulary is highly distinctive — these phrases almost never appear
# in search or booking messages, so a match here is reliable without LLM confirmation.
_CANCELLATION_KEYWORDS = frozenset([
    "cancel", "cancellation", "cancelled", "withdraw",
    "remove booking", "no longer want to attend", "no longer available",
    "don't want to attend", "don't want the booking", "don't want the inspection",
])

# Booking keywords are only used as fast-path when NO search keywords are present
# (see _obvious_intent). This prevents "find a house and book it" from being
# misclassified as a standalone booking — it routes to listing_search_node first.
_BOOKING_KEYWORDS = frozenset([
    "book", "viewing", "view the property",
    "schedule", "arrange", "open for inspection", "open home",
])

# Lookup phrases indicate the user wants to retrieve an existing booking, not create one.
# Without this distinction the agent would try to create a duplicate booking or
# route to tool-calling mode with no useful tool to call.
_LOOKUP_KEYWORDS = frozenset([
    "my booking", "my inspection", "check my booking", "check booking",
    "booking details", "booking status", "when is my inspection",
    "what time is my", "show my booking", "my confirmation",
    "look up my booking", "find my booking",
    "see my booking", "see my inspection", "view my booking", "view my inspection",
    "booked an inspection", "booked a viewing", "i booked",
])

# Search keywords are used in two roles:
#   1. Suppress the booking fast-path so search+book flows go through listing_search_node.
#   2. Detect search_then_deposit when combined with deposit keywords.
_SEARCH_KEYWORDS = frozenset([
    "find", "search", "show", "list", "looking for",
    "properties", "house", "apartment", "unit", "townhouse",
    "bedroom", "bathroom", "suburb", "price", "budget",
    "under", "rent for", "for rent", "to rent", "buy", "purchase",
])

# Deposit keywords are specific enough to route directly without LLM classification.
# Combined with _SEARCH_KEYWORDS they produce "search_then_deposit", which lets
# listing_search_node run first so the agent has a property_id to attach the deposit to.
_DEPOSIT_KEYWORDS = frozenset([
    "pay deposit", "paying deposit", "holding deposit",
    "pay the deposit", "deposit payment", "pay my deposit",
])

# 4 messages of human history gives the LLM enough context to resolve follow-ups
# ("show me cheaper ones", "what about 3 bedrooms?") without inflating the prompt
# with irrelevant older turns that could distract or confuse the classifier.
_LLM_HISTORY_LIMIT = 4


async def intent_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Classifies intent from the latest HumanMessage.
    """
    message = last_human_message(state).lower()
    if not message:
        return {StateKeys.USER_INTENT: "general"}

    # Try the keyword fast path first — avoids an LLM call for unambiguous intents,
    # reducing latency and eliminating any risk of LLM misclassification on clear cases.
    obvious = _obvious_intent(message)
    if obvious:
        return {StateKeys.USER_INTENT: obvious}

    # Slot selection (e.g. "I'll take the 3pm one") can't be expressed as a keyword
    # match, but can be inferred from the presence of available_slots in state.
    # Catching it here prevents the LLM from misreading the follow-up as a new search.
    if _is_booking_continuation(state, message):
        return {StateKeys.USER_INTENT: "booking"}

    return await _classify_with_llm(state)


async def _classify_with_llm(state: RealEstateAgentState) -> dict[str, Any]:
    """Invoke the LLM to classify intent and extract search entities."""
    # After a booking/cancellation completes, intent_completed=True signals a fresh turn.
    # In that case we only pass the single latest message so the LLM doesn't interpret
    # prior booking messages as context for the new request.
    intent_completed = state.get(StateKeys.INTENT_COMPLETED, False)
    if intent_completed:
        history = state["messages"][-1:]
    else:
        # Only HumanMessages — ToolMessages and AIMessages contain intermediate
        # agent reasoning that the intent classifier doesn't need and could misread.
        history = [message for message in state["messages"] if isinstance(
            message, HumanMessage)][-_LLM_HISTORY_LIMIT:]

    # The state hint tells the LLM what was happening before this message, preventing
    # it from hallucinating a booking intent when the user is now asking something else.
    state_hint = _build_state_hint(
        state.get(StateKeys.LAST_INTENT), intent_completed)

    prompt = [SystemMessage(
        content=INTENT_CLASSIFICATION_PROMPT + state_hint), *history]

    try:
        # with_structured_output forces the LLM to return a validated Pydantic object,
        # eliminating free-form JSON parsing errors and hallucinated field names.
        llm = get_llm().with_structured_output(IntentClassification)
        classification: IntentClassification = await llm.ainvoke(prompt)
    except Exception as exc:
        logger.error("intent_node | LLM classification failed: %s", exc)
        # Fall back to "general" rather than raising — keeps the conversation alive
        # and lets agent_node handle the message as a plain question.
        return {StateKeys.USER_INTENT: "general"}

    return _build_state_update(state, classification)


def _obvious_intent(msg_lower: str) -> UserIntent | None:
    """
    Returns a high-confidence intent without an LLM call, or None if ambiguous.

    Cancellation: very distinctive vocabulary, never overlaps with other intents.
    Booking:      only when no search keywords present — avoids missing search_then_book.
    """
    # Require no search keywords alongside cancellation so that "find my booking and
    # cancel it" is not fast-pathed — it needs the LLM to understand the compound intent.
    if (_matches_keywords(msg_lower, _CANCELLATION_KEYWORDS) and
            not _matches_keywords(msg_lower, _SEARCH_KEYWORDS)):
        return "cancellation"

    # Lookup wins over cancellation ambiguity: "check my booking to cancel" should
    # first retrieve the booking so the user can confirm before we cancel anything.
    if (_matches_keywords(msg_lower, _LOOKUP_KEYWORDS) and
            not _matches_keywords(msg_lower, _CANCELLATION_KEYWORDS)):
        return "booking_lookup"

    # Only fast-path booking when there are no search keywords — compound messages like
    # "find a 2-bed flat and book a viewing" must go through listing_search_node first
    # so the agent has a property_id before attempting to book.
    if (_matches_keywords(msg_lower, _BOOKING_KEYWORDS) and
            not _matches_keywords(msg_lower, _SEARCH_KEYWORDS)):
        return "booking"

    if _matches_keywords(msg_lower, _DEPOSIT_KEYWORDS):
        # Deposit + search = user wants to find a property and then pay a deposit.
        # Routing to search_then_deposit runs listing_search_node first.
        if _matches_keywords(msg_lower, _SEARCH_KEYWORDS):
            return "search_then_deposit"
        return "deposit_payment"

    return None


def _is_booking_continuation(state: RealEstateAgentState, message: str) -> bool:
    """True when slots are pending and the user is selecting one, not searching or cancelling."""
    # Previous intent completed — never continue a finished flow
    if state.get(StateKeys.INTENT_COMPLETED):
        return False

    # No pending slots means there's nothing for the user to select — the message
    # is either a new request or something the LLM must interpret.
    booking_ctx = state.get(StateKeys.BOOKING_CONTEXT)
    if not booking_ctx or not booking_ctx.get("available_slots"):
        return False

    # Once a booking is confirmed or cancelled, slots are stale. Continuing the flow
    # here would send the user back to the booking tool with outdated slot data.
    booking_status = state.get(StateKeys.BOOKING_STATUS) or {}
    if booking_status.get("confirmed") or booking_status.get("cancelled"):
        return False

    # A slot-selection message should not contain search or cancellation vocabulary.
    # This prevents "actually, find me something cheaper" from being treated as a
    # slot selection and incorrectly continuing the booking flow.
    return (
        not _matches_keywords(message, _SEARCH_KEYWORDS)
        and not _matches_keywords(message, _CANCELLATION_KEYWORDS)
    )


def _build_state_update(
        state: RealEstateAgentState,
        result: IntentClassification) -> dict[str, Any]:
    """
    Build the full state update dict from an LLM classification result.
    Merges or replaces search_context depending on whether location changed.
    """
    updates: dict[str, Any] = {
        StateKeys.USER_INTENT:      result.intent,
        StateKeys.EARLY_RESPONSE:   result.early_response,
        # last_intent is written here (not only on completion) so the state hint
        # in the next turn always reflects the most recent routing decision.
        StateKeys.LAST_INTENT:      result.intent,
        # Reset on every new classification so _is_booking_continuation and
        # _classify_with_llm don't carry over a stale "completed" signal.
        StateKeys.INTENT_COMPLETED: False,
    }

    # "general" has no entities to extract — skip merge to avoid touching
    # search_context with empty data that could pollute a later search turn.
    if result.intent == "general":
        return updates

    entities = _extract_entities(result)
    if not entities:
        return updates

    if entities.get("location"):
        # A new location means the user is looking somewhere different — replace the
        # entire search_context so stale filters (bedrooms, price) from the previous
        # location don't silently carry over and produce wrong SQL.
        updates[StateKeys.SEARCH_CONTEXT] = entities
    else:
        # No location change: merge new entities into existing context so refinements
        # like "make it 3 bedrooms" layer on top of the current suburb and price.
        existing = dict(state.get(StateKeys.SEARCH_CONTEXT) or {})
        updates[StateKeys.SEARCH_CONTEXT] = {**existing, **entities}

    return updates


def _build_state_hint(last_intent: str | None, intent_completed: bool) -> str:
    """Return a [STATE] hint string for the LLM prompt, or '' if no prior intent."""
    if not last_intent:
        return ""
    # Without this hint the LLM can misread a follow-up ("book it") as a standalone
    # booking when there's no prior context, or mistake a new search as a continuation
    # of a just-finished booking flow. The hint anchors the LLM to actual graph state.
    return (
        f"\n[STATE] Previous intent: {last_intent}, "
        f"Completed: {intent_completed}. "
        f"If Completed is True, treat the current message as a fresh request."
    )


def _extract_entities(result: IntentClassification) -> dict:
    """Build a partial SearchContext dict from LLM-extracted entities."""
    # String fields: skip if falsy; numeric fields: skip if None
    _str_fields = ("location", "address", "listing_type", "property_type")
    _num_fields = ("bedrooms", "bathrooms", "max_price", "min_price", "limit")

    entities = {f: getattr(result, f) for f in _str_fields if getattr(result, f)}
    entities |= {f: getattr(result, f) for f in _num_fields if getattr(result, f) is not None}

    # Cap at 10 so a hallucinated large limit (e.g. 50) doesn't cause the SQL
    # service to return more rows than the frontend or LLM prompt can handle.
    if "limit" in entities:
        entities["limit"] = min(entities["limit"], 10)

    return entities


def _matches_keywords(msg_lower: str, keywords: frozenset[str]) -> bool:
    """Whole-word/phrase keyword match — prevents substring false positives."""
    for keyword in keywords:
        # \b word-boundary anchors prevent "cancel" matching inside "cancellation"
        # or "book" matching inside "Facebook", which would misroute innocent messages.
        if re.search(rf'\b{re.escape(keyword)}\b', msg_lower):
            return True
    return False
