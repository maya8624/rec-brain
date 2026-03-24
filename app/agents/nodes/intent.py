"""
intent_node — classifies user intent before any LLM call.

Reads the latest HumanMessage, runs keyword classification,
writes user_intent into state — no LLM cost.

Intent values:
    "search"          → listing_search_node  (v_listings direct query)
    "booking"         → agent_node           (LLM calls check_availability/book_inspection)
    "cancellation"    → agent_node           (LLM calls cancel_inspection)
    "document_query"  → vector_search_node
    "general"         → agent_node           (LLM plain response)

Compound intents (e.g. search + booking) → "general"
    — LLM responds asking user to clarify one action at a time.
"""

import logging
from typing import Any
from langchain_core.messages import HumanMessage
from app.agents.state import RealEstateAgentState, UserIntent

logger = logging.getLogger(__name__)

_INTENT_KEYWORDS: dict[UserIntent, frozenset[str]] = {
    "cancellation": frozenset([
        "cancel", "cancellation", "cancelled", "withdraw",
        "no longer", "don't want", "remove booking",
    ]),
    "booking": frozenset([
        "book", "inspect", "inspection", "viewing", "view",
        "schedule", "arrange", "available", "availability",
        "when can i", "open for inspection", "open home",
    ]),
    "document_query": frozenset([
        "lease", "contract", "strata", "terms", "clause",
        "bond", "deposit", "condition", "by-law", "bylaw",
        "pet policy", "break lease", "notice period",
        "landlord", "tenant", "agreement",
    ]),
    "search": frozenset([
        "find", "search", "show", "list", "looking for",
        "properties", "house", "apartment", "unit", "townhouse",
        "bedroom", "bathroom", "suburb", "price", "budget",
        "under", "rent", "buy", "purchase",
    ]),
}

# If more than one of these match → compound intent → "general"
_COMPOUND_INTENTS = frozenset(["search", "booking", "cancellation"])


async def intent_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Classifies intent from the latest HumanMessage.
    Writes user_intent into state — no LLM call.
    """
    message = _get_last_human_message(state)
    intent = _classify_intent(message)

    logger.info(
        "intent_node | intent=%s | message=%.60s",
        intent,
        message,
    )

    return {"user_intent": intent}


def _classify_intent(message: str) -> UserIntent:
    """
    Keyword-based intent classification. Zero LLM cost.

    If multiple compound intents match (e.g. search + booking),
    falls through to "general" so LLM can ask user to clarify.
    """
    if not message:
        return "general"

    msg_lower = message.lower()

    matched_intents = [
        intent
        for intent, keywords in _INTENT_KEYWORDS.items()
        if any(keyword in msg_lower for keyword in keywords)
    ]

    compound_matches = [i for i in matched_intents if i in _COMPOUND_INTENTS]
    if len(compound_matches) > 1:
        logger.debug(
            "_classify_intent | compound=%s → general", compound_matches
        )

        return {
            "user_intent": "general",
            "early_response": "I can only do one thing at a time. "
                              "Would you like me to check availability first, or search for properties?"
        }

    if matched_intents:
        intent = matched_intents[0]
        logger.debug("_classify_intent | '%.40s' → %s", message, intent)

        return {"user_intent": intent}

    return {"user_intent": "general"}


def _get_last_human_message(state: RealEstateAgentState) -> str:
    """Return the content of the most recent HumanMessage in state."""
    for message in reversed(state["messages"]):
        if isinstance(message, HumanMessage):
            return message.content if isinstance(message.content, str) else ""

    return ""
