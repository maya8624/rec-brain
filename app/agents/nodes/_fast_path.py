"""
Fast-path intent checks — keyword/state-based, no LLM call.

Called by intent_node before falling through to _classify_with_llm.
"""

import re
from app.agents.state import RealEstateAgentState, UserIntent
from app.core.constants import IntentConfig, StateKeys


def obvious_intent(msg_lower: str) -> UserIntent | None:
    """
    Returns a high-confidence intent without an LLM call, or None if ambiguous.
    """
    if (matches_keywords(msg_lower, IntentConfig.CANCELLATION_KEYWORDS) and
            not matches_keywords(msg_lower, IntentConfig.SEARCH_KEYWORDS)):
        return "cancellation"

    if (matches_keywords(msg_lower, IntentConfig.LOOKUP_KEYWORDS) and
            not matches_keywords(msg_lower, IntentConfig.CANCELLATION_KEYWORDS)):
        return "booking_lookup"

    if (matches_keywords(msg_lower, IntentConfig.BOOKING_KEYWORDS) and
            not matches_keywords(msg_lower, IntentConfig.SEARCH_KEYWORDS)):
        return "booking"

    if matches_keywords(msg_lower, IntentConfig.DEPOSIT_KEYWORDS):
        if matches_keywords(msg_lower, IntentConfig.SEARCH_KEYWORDS):
            return "search_then_deposit"
        return "deposit_payment"

    return None


def is_booking_continuation(state: RealEstateAgentState, message: str) -> bool:
    """True when slots are pending and the user is selecting one, not searching or cancelling."""
    if state.get(StateKeys.INTENT_COMPLETED):
        return False

    booking_ctx = state.get(StateKeys.BOOKING_CONTEXT)
    if not booking_ctx or not booking_ctx.get("available_slots"):
        return False

    booking_status = state.get(StateKeys.BOOKING_STATUS) or {}
    if booking_status.get("confirmed") or booking_status.get("cancelled"):
        return False

    no_search = not matches_keywords(message, IntentConfig.SEARCH_KEYWORDS)
    no_cancellation = not matches_keywords(
        message, IntentConfig.CANCELLATION_KEYWORDS)

    return no_search and no_cancellation

    # if (not matches_keywords(message, IntentConfig.SEARCH_KEYWORDS) and
    #         not matches_keywords(message, IntentConfig.CANCELLATION_KEYWORDS)):
    #     return True

    # return False


def matches_keywords(message: str, keywords: frozenset[str]) -> bool:
    """Whole-word/phrase keyword match — prevents substring false positives."""
    for keyword in keywords:
        if re.search(rf'\b{re.escape(keyword)}\b', message):
            return True
    return False
