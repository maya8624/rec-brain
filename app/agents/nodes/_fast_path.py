"""
Fast-path intent checks — keyword/state-based, no LLM call.

Called by intent_node before falling through to _classify_with_llm.
"""

import re
from app.agents.state import ConversationPhase, RealEstateAgentState, UserIntent
from app.core.constants import Intent, IntentConfig, StateKeys

# TODO: refactor


def fast_path_intent(message: str, state: RealEstateAgentState) -> UserIntent | None:
    """
    Returns a high-confidence intent without an LLM call, or None if ambiguous.
    """
    if (matches_keywords(message, IntentConfig.CANCELLATION_KEYWORDS) and
            not matches_keywords(message, IntentConfig.SEARCH_KEYWORDS)):
        return Intent.CANCELLATION

    if (matches_keywords(message, IntentConfig.LOOKUP_KEYWORDS) and
            not matches_keywords(message, IntentConfig.CANCELLATION_KEYWORDS)):
        return Intent.BOOKING_LOOKUP

    if (matches_keywords(message, IntentConfig.BOOKING_KEYWORDS) and
            not matches_keywords(message, IntentConfig.SEARCH_KEYWORDS)):
        return Intent.BOOKING

    if matches_keywords(message, IntentConfig.DEPOSIT_KEYWORDS):
        return Intent.DEPOSIT_PAYMENT

    if matches_keywords(message, IntentConfig.DOCUMENT_KEYWORDS):
        return Intent.DOCUMENT_QUERY

    return None


def is_booking_continuation(state: RealEstateAgentState, message: str) -> bool:
    """True when slots are pending and the user is selecting one, not searching or cancelling."""
    booking_ctx = state.get(StateKeys.BOOKING_CONTEXT)
    if not booking_ctx or not booking_ctx.get("available_slots"):
        return False

    no_search = not matches_keywords(message, IntentConfig.SEARCH_KEYWORDS)
    no_cancellation = not matches_keywords(
        message, IntentConfig.CANCELLATION_KEYWORDS)

    return no_search and no_cancellation


def is_cancellation_continuation(state: RealEstateAgentState, message: str) -> bool:
    """True when the user is confirming cancellation for a known booking."""
    booking_ctx = state.get(StateKeys.BOOKING_CONTEXT) or {}

    if not booking_ctx.get("confirmation_id"):
        return False

    phase = state.get(StateKeys.PHASE, ConversationPhase.IDLE)
    if phase not in {ConversationPhase.CANCELLATION_PENDING}:
        return False

    return matches_keywords(message, IntentConfig.CONFIRMATION_KEYWORDS)


def matches_keywords(message: str, keywords: frozenset[str]) -> bool:
    """Whole-word/phrase keyword match — prevents substring false positives."""
    for keyword in keywords:
        if re.search(rf'\b{re.escape(keyword)}\b', message):
            return True
    return False
