"""
intent_node — classifies user intent before routing through the graph.

Strategy: hybrid keyword + LLM (Option 3)

    Fast path (keyword, no LLM):
        - Obvious cancellation  → "cancellation"
        - Obvious booking only  → "booking"

    LLM path (everything else):
        - Ambiguous, follow-up, compound, search queries
        - Returns IntentClassification with intent + extracted entities
        - Entities written to state["search_context"] for downstream nodes

Why hybrid:
    Cancellation and standalone booking are unambiguous and need no entity
    extraction — keyword matching is faster and free. Everything else benefits
    from LLM context awareness (follow-ups, compound intents, entity extraction).
"""

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._base import last_human_message
from app.agents.state import IntentClassification, RealEstateAgentState, UserIntent
from app.infrastructure.llm import get_llm
from app.prompts.intent import INTENT_CLASSIFICATION_PROMPT

logger = logging.getLogger(__name__)

# ------------------------------------
#  Keyword sets for the fast path
# ------------------------------------
_CANCELLATION_KEYWORDS = frozenset([
    "cancel", "cancellation", "cancelled", "withdraw",
    "no longer", "don't want", "remove booking",
])

_BOOKING_KEYWORDS = frozenset([
    "book", "viewing", "view the property",
    "schedule", "arrange", "open for inspection", "open home",
])

_SEARCH_KEYWORDS = frozenset([
    "find", "search", "show", "list", "looking for",
    "properties", "house", "apartment", "unit", "townhouse",
    "bedroom", "bathroom", "suburb", "price", "budget",
    "under", "rent for", "for rent", "to rent", "buy", "purchase",
])

# History depth for LLM path — enough to resolve follow-up references
_LLM_HISTORY_LIMIT = 4


async def intent_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Classifies intent from the latest HumanMessage.

    Fast path: keyword match for obvious cancellation / booking — no LLM call.
    LLM path:  everything else — returns intent + extracted search entities.
    """
    message = last_human_message(state)
    if not message:
        return {"user_intent": "general"}

    #  Fast path: keyword-based intent classification
    obvious = _obvious_intent(message)
    if obvious:
        logger.info(
            "intent_node | fast-path | intent=%s | message=%.60s", obvious, message)
        return {"user_intent": obvious}

    #  LLM path: everything else — returns intent + extracted search entities.
    # Only HumanMessages are sent — AIMessages contain full listing/RAG responses
    # that add noise and tokens the intent classifier doesn't need.
    history = [m for m in state["messages"] if isinstance(m, HumanMessage)][-_LLM_HISTORY_LIMIT:]
    prompt = [SystemMessage(content=INTENT_CLASSIFICATION_PROMPT), *history]

    try:
        llm = get_llm().with_structured_output(IntentClassification)
        result: IntentClassification = await llm.ainvoke(prompt)
    except Exception as exc:
        logger.error("intent_node | LLM classification failed: %s", exc)
        return {"user_intent": "general"}

    logger.info(
        "intent_node | llm-path | intent=%s | early_response=%s | "
        "location=%s | property_type=%s | bedrooms=%s | max_price=%s | message=%.60s",
        result.intent,
        bool(result.early_response),
        result.location,
        result.property_type,
        result.bedrooms,
        result.max_price,
        message,
    )

    updates: dict[str, Any] = {
        "user_intent": result.intent,
        "early_response": result.early_response,
    }

    # Merge extracted entities into search_context (preserves previous filters)
    entities = _extract_entities(result)
    if entities:
        existing = dict(state.get("search_context") or {})
        updates["search_context"] = {**existing, **entities}

    return updates


def _obvious_intent(message: str) -> UserIntent | None:
    """
    Returns a high-confidence intent without an LLM call, or None if ambiguous.

    Cancellation: very distinctive vocabulary, never overlaps with other intents.
    Booking:      only when no search keywords present — avoids missing search_then_book.
    """
    msg_lower = message.lower()

    if (_matches_keywords(msg_lower, _CANCELLATION_KEYWORDS) and
            not _matches_keywords(msg_lower, _SEARCH_KEYWORDS)):
        return "cancellation"

    if (_matches_keywords(msg_lower, _BOOKING_KEYWORDS) and
            not _matches_keywords(msg_lower, _SEARCH_KEYWORDS)):
        return "booking"

    return None


def _extract_entities(result: IntentClassification) -> dict:
    """Build a partial SearchContext dict from LLM-extracted entities."""
    entities = {}
    if result.location:
        entities["location"] = result.location
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
    return entities


def _matches_keywords(msg_lower: str, keywords: frozenset[str]) -> bool:
    """Whole-word/phrase keyword match — prevents substring false positives."""
    for keyword in keywords:
        if re.search(rf'\b{re.escape(keyword)}\b', msg_lower):
            return True
    return False
