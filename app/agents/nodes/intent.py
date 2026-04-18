"""
intent_node — classifies user intent before any LLM call.

Reads the latest HumanMessage, runs keyword classification,
writes user_intent into state — no LLM cost.

Intent values:
    "search"          → listing_search_node  (v_listings direct query)
    "document_query"  → vector_search_node
    "hybrid_search"   → hybrid_search_node   (search + document_query together)
    "booking"         → agent_node           (LLM calls check_availability/book_inspection)
    "cancellation"    → agent_node           (LLM calls cancel_inspection)
    "general"         → agent_node           (LLM plain response)

Compound intents (e.g. search + booking) → "general"
    — LLM responds asking user to clarify one action at a time.
"""

import logging
import re
from typing import Any
from app.agents.nodes._base import last_human_message
from app.agents.state import RealEstateAgentState, UserIntent

logger = logging.getLogger(__name__)

_INTENT_KEYWORDS: dict[UserIntent, frozenset[str]] = {
    "cancellation": frozenset([
        "cancel", "cancellation", "cancelled", "withdraw",
        "no longer", "don't want", "remove booking",
    ]),
    "booking": frozenset([
        "book", "viewing", "view the property",
        "schedule", "arrange", "available", "availability",
        "when can i", "open for inspection", "open home",
    ]),
    "document_query": frozenset([
        # Legal / tenancy documents
        "lease", "contract", "strata", "terms", "clause",
        "bond", "deposit", "condition", "by-law", "bylaw",
        "pet policy", "break lease", "notice period",
        "landlord", "tenant", "agreement",
        # Agency info
        "address", "location", "where are you",
        "phone", "call you", "contact",
        "email",
        "website", "web site", "online",
        "hours", "trading hours", "office hours", "open", "opening hours", "when are you",
        "who is", "staff", "agent", "personnel", "team", "manager", "principal",
    ]),
    "search": frozenset([
        "find", "search", "show", "list", "looking for",
        "properties", "house", "apartment", "unit", "townhouse",
        "bedroom", "bathroom", "suburb", "price", "budget",
        "under", "rent for", "for rent", "to rent", "buy", "purchase",
    ]),
}

# If more than one of these match → compound intent → "general"
_COMPOUND_INTENTS = frozenset(["search", "booking", "cancellation"])

# Patterns that indicate a location was provided in a search query
_LOCATION_PATTERN = re.compile(
    r'\b(in|at|near|around|within)\s+\w+'       # "in Sydney", "near the CBD"
    r'|\b(sydney|melbourne|brisbane|perth|adelaide|hobart|darwin|canberra'
    r'|nsw|vic|qld|wa|sa|tas|nt|act)\b',         # city / state names
    re.IGNORECASE,
)


async def intent_node(state: RealEstateAgentState) -> dict[str, Any]:
    """
    Classifies intent from the latest HumanMessage.
    Writes user_intent into state — no LLM call.
    """
    message = last_human_message(state)
    intent = _classify_intent(message)

    logger.info(
        "intent_node | intent=%s | message=%.60s",
        intent,
        message,
    )

    # Vague search — no location provided
    # TODO: Option 3 — promote to a dedicated clarify_node between intent_node
    #   and listing_search_node. It would check query completeness (location,
    #   price range, property type) and route to END with a clarifying question
    #   if required criteria are missing, or forward to listing_search_node if
    #   sufficient. Cleaner separation of concerns and easier to extend with
    #   more validation rules without growing intent_node further.
    if intent == "search" and not _has_location(message):
        return {
            "user_intent": "search",
            "early_response": "Which suburb or area are you looking in?",
        }

    # Compound intent handling
    if intent == "general" and _is_compound(message):
        # search + booking: run search first, agent prompts user to pick a property
        if _is_search_and_book(message):
            return {"user_intent": "search_then_book"}
        # all other compounds: ask user to clarify one request at a time
        return {
            "user_intent": "general",
            "early_response": "I can only handle one request at a time. "
                              "Would you like to search for properties, or book an inspection?"
        }

    return {"user_intent": intent}


def _classify_intent(message: str) -> UserIntent:
    """
    Keyword-based intent classification. Zero LLM cost.
    Returns a plain string intent — never a dict.
    """
    if not message:
        return "general"

    msg_lower = message.lower()

    matched_intents = [
        intent
        for intent, keywords in _INTENT_KEYWORDS.items()
        if _matches_keywords(msg_lower, keywords)
    ]

    # Hybrid: search + document_query together → use both sources
    if "search" in matched_intents and "document_query" in matched_intents:
        logger.debug("_classify_intent | search+document_query → hybrid_search")
        return "hybrid_search"

    compound_matches = [i for i in matched_intents if i in _COMPOUND_INTENTS]
    if len(compound_matches) > 1:
        logger.debug(
            "_classify_intent | compound=%s → general", compound_matches
        )
        return "general"  # ← plain string, early_response handled in intent_node

    if matched_intents:
        intent = matched_intents[0]
        logger.debug("_classify_intent | '%.40s' → %s", message, intent)
        return intent  # ← plain string

    return "general"  # ← plain string


def _is_compound(message: str) -> bool:
    """Returns True if multiple compound intents match the message."""
    msg_lower = message.lower()
    compound_matches = [
        intent for intent in _COMPOUND_INTENTS
        if _matches_keywords(msg_lower, _INTENT_KEYWORDS[intent])
    ]
    return len(compound_matches) > 1


def _is_search_and_book(message: str) -> bool:
    """True only when the compound is exactly search + booking (not cancellation)."""
    msg_lower = message.lower()
    return (
        _matches_keywords(msg_lower, _INTENT_KEYWORDS["search"]) and
        _matches_keywords(msg_lower, _INTENT_KEYWORDS["booking"])
    )


def _has_location(message: str) -> bool:
    """Returns True if the message contains a recognisable location indicator."""
    return bool(_LOCATION_PATTERN.search(message))


def _matches_keywords(msg_lower: str, keywords: frozenset[str]) -> bool:
    """
    Match keywords as whole words/phrases using word boundary matching.
    Prevents substring false positives e.g. "rent" matching "rental".
    """
    for keyword in keywords:
        if re.search(rf'\b{re.escape(keyword)}\b', msg_lower):
            return True
    return False
