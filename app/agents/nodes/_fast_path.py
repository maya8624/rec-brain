"""
Fast-path intent checks — keyword/state-based, no LLM call.

Called by intent_node before falling through to _classify_with_llm.
"""

import re
from app.agents.state import RealEstateAgentState, UserIntent
from app.core.constants import IntentConfig, StateKeys


def fast_path_intent(message: str, state: RealEstateAgentState) -> UserIntent | None:
    """
    Returns a high-confidence intent without an LLM call, or None if ambiguous.
    """
    if (matches_keywords(message, IntentConfig.CANCELLATION_KEYWORDS) and
            not matches_keywords(message, IntentConfig.SEARCH_KEYWORDS)):
        return "cancellation"

    if (matches_keywords(message, IntentConfig.LOOKUP_KEYWORDS) and
            not matches_keywords(message, IntentConfig.CANCELLATION_KEYWORDS)):
        return "booking_lookup"

    if (matches_keywords(message, IntentConfig.BOOKING_KEYWORDS) and
            not matches_keywords(message, IntentConfig.SEARCH_KEYWORDS)):
        # not _is_search_then_book(state)):
        return "booking"

    if matches_keywords(message, IntentConfig.DEPOSIT_KEYWORDS):
        if matches_keywords(message, IntentConfig.SEARCH_KEYWORDS):
            return "search_then_deposit"
        return "deposit_payment"

    if matches_keywords(message, IntentConfig.DOCUMENT_KEYWORDS):
        return "document_query"

    return None


# def _is_search_then_book(state: RealEstateAgentState) -> bool:
#     """
#     Keep "booking" when the user can already resolve a property_id (prior search
#     results or an existing booking context). Otherwise return None to fall through
#     to the LLM classifier, which can detect the address and classify as
#     search_then_book so the property is looked up before check_availability runs.
#     """
#     search_results = state.get(StateKeys.SEARCH_RESULTS) or []
#     booking_ctx = state.get(StateKeys.BOOKING_CONTEXT) or {}
#     if search_results or booking_ctx.get("property_id"):
#         return False
#     return True


def is_booking_continuation(state: RealEstateAgentState, message: str) -> bool:
    """True when slots are pending and the user is selecting one, not searching or cancelling."""
    booking_ctx = state.get(StateKeys.BOOKING_CONTEXT)
    if not booking_ctx or not booking_ctx.get("available_slots"):
        return False

    # booking_status = state.get(StateKeys.BOOKING_STATUS) or {}
    # if booking_status.get("confirmed") or booking_status.get("cancelled"):
    #     return False

    no_search = not matches_keywords(message, IntentConfig.SEARCH_KEYWORDS)
    no_cancellation = not matches_keywords(
        message, IntentConfig.CANCELLATION_KEYWORDS)

    return no_search and no_cancellation


def is_cancellation_continuation(state: RealEstateAgentState, message: str) -> bool:
    """True when the user is confirming cancellation for a known booking."""
    booking_ctx = state.get(StateKeys.BOOKING_CONTEXT) or {}
    # booking_status = state.get(StateKeys.BOOKING_STATUS) or {}
    last_intent = state.get(StateKeys.LAST_INTENT)

    # if booking_status.get("cancelled"):
    #     return False

    if not booking_ctx.get("confirmation_id"):
        return False

    if last_intent not in {"cancellation", "booking_lookup"}:
        return False

    return matches_keywords(message, IntentConfig.CONFIRMATION_KEYWORDS)


def matches_keywords(message: str, keywords: frozenset[str]) -> bool:
    """Whole-word/phrase keyword match — prevents substring false positives."""
    for keyword in keywords:
        if re.search(rf'\b{re.escape(keyword)}\b', message):
            return True
    return False
