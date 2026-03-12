from __future__ import annotations
import re

from typing import Any
from src.models.routing_models import RouteType, RoutingDecision, UserIntent
from src.models.state_models import OrchestrationState


class HybridRouter:
    """
    Decides how a user request should be handled.

    Responsibilities:
    - detect high-level user intent (e.g., information retrieval, task execution, etc.)
    - choose the execution route
    - decide whether SQL, vector search, and/or tools are needed
    - extract simple entities for downstream planning

    This is first version uses deterministic rules.
    Later, it can be enhanced with an LLM-assisted classifier
    """

    TOOL_KEYWORDS = {
        "check_availability": [
            "availability",
            "available",
            "can i inspect",
            "inspection time",
            "viewing time",
            "check inspection",
        ],
        "schedule_inspection": [
            "book inspection",
            "schedule inspection",
            "book viewing",
            "schedule viewing",
            "arrange inspection",
            "arrange viewing",
        ],
        "cancel_viewing": [
            "cancel inspection",
            "cancel viewing",
            "cancel booking",
            "remove inspection",
        ],
    }

    PROPERTY_SEARCH_KEYWORDS = [
        "show me",
        "find me",
        "properties",
        "houses",
        "apartments",
        "units",
        "bedroom",
        "under",
        "budget",
        "suburb",
        "in sydney",
    ]

    MARKET_INFO_KEYWORDS = [
        "market",
        "trend",
        "growth",
        "growing",
        "suburb insight",
        "median price",
        "investment",
        "forecast",
    ]

    PROPERTY_DETAIL_KEYWORDS = [
        "tell me about",
        "property details",
        "property information",
        "more about",
        "listing",
        "what about",
        "details of",
        "information on",
    ]

    def route(self, state: OrchestrationState) -> RoutingDecision:
        """
        Build a routing decision from the current orchestration state.
        """
        message = state.user_message.strip()
        normalized = self._normalize(message)
        intent = self._detect_intent(normalized)
        entities = self._extract_entities(message)
        candidate_tools = self._detect_candidate_tools(message, intent)
        route = self._select_route(intent, candidate_tools, entities)

        confidence = self._estimate_confidence(
            intent,
            route,
            candidate_tools,
            entities)

        reasoning = self._build_reasoning(intent, route, candidate_tools)

    def _estimate_confidence(
        self,
        intent: UserIntent,
        route: RouteType,
        candidate_tools: list[str],
        entities: dict[str, Any],
    ) -> float:
        """
        Return a simple confidence score between 0 and 1.
        """
        if intent == UserIntent.GENERAL_CHAT and route == RouteType.GENERAL:
            return 0.60

        if intent in {
            UserIntent.CHECK_AVAILABILITY,
            UserIntent.SCHEDULE_VIEWING,
            UserIntent.CANCEL_VIEWING,
        }:
            return 0.90 if candidate_tools else 0.70

        if intent == UserIntent.PROPERTY_SEARCH:
            return 0.92 if entities else 0.80

        if intent == UserIntent.MARKET_INFO:
            return 0.88

        if intent == UserIntent.PROPERTY_DETAILS:
            return 0.85

        return 0.65

    def _build_reasoning(
        self,
        intent: UserIntent,
        route: RouteType,
        candidate_tools: list[str],
    ) -> str:
        """
        Build a short explanation for observability and debugging.
        """
        if candidate_tools:
            return (
                f"Detected intent '{intent.value}' and selected route '{route.value}' "
                f"with candidate tools: {', '.join(candidate_tools)}."
            )

        return f"Detected intent '{intent.value}' and selected route '{route.value}'."

    def _detect_candidate_tools(
        self,
        message: str,
        intent: UserIntent,
    ) -> list[str]:
        """
        Return likely tools for the request.
        """
        tools: list[str] = []

        if intent == UserIntent.CHECK_AVAILABILITY:
            tools.append("check_availability")

        elif intent == UserIntent.SCHEDULE_VIEWING:
            tools.extend(["check_availability", "schedule_viewing"])

        elif intent == UserIntent.CANCEL_VIEWING:
            tools.append("cancel_viewing")

        return tools

    def _select_route(
        self,
        intent: UserIntent,
        candidate_tools: list[str],
        entities: dict[str, Any],
    ) -> RouteType:
        """
        Map intent + context to an execution route.
        """
        if intent == UserIntent.PROPERTY_SEARCH:
            return RouteType.SQL_ONLY

        if intent == UserIntent.MARKET_INFO:
            return RouteType.VECTOR_ONLY

        if intent == UserIntent.PROPERTY_DETAILS:
            has_listing_identifier = any(
                key in entities for key in ("listing_id", "property_id", "address")
            )
            return RouteType.VECTOR_SQL if has_listing_identifier else RouteType.VECTOR_ONLY

        if intent == UserIntent.CHECK_AVAILABILITY:
            return RouteType.SQL_TOOL

        if intent == UserIntent.SCHEDULE_VIEWING:
            return RouteType.SQL_TOOL

        if intent == UserIntent.CANCEL_VIEWING:
            return RouteType.TOOL_ONLY

        if candidate_tools:
            return RouteType.TOOL_ONLY

        return RouteType.GENERAL

    def _detect_intent(self, message: str) -> UserIntent:
        """
        Detect the usr's high-level intent using simple rules.
        """
        if self._contains_any(message, self.TOOL_KEYWORDS["cancel_viewing"]):
            return UserIntent.CANCEL_VIEWING

        if self._contains_any(message, self.TOOL_KEYWORDS["schedule_viewing"]):
            return UserIntent.SCHEDULE_VIEWING

        if self._contains_any(message, self.TOOL_KEYWORDS["check_availability"]):
            return UserIntent.CHECK_AVAILABILITY

        if self._contains_any(message, self.MARKET_INFO_KEYWORDS):
            return UserIntent.MARKET_INFO

        if self._contains_any(message, self.PROPERTY_DETAIL_KEYWORDS):
            return UserIntent.PROPERTY_DETAILS

        if self._contains_any(message, self.PROPERTY_SEARCH_KEYWORDS):
            return UserIntent.PROPERTY_SEARCH

        return UserIntent.GENERAL_CHAT

    def _extract_entities(self, message: str) -> dict[str, Any]:
        """
        Extract lightweight structured entities.

        This is intentionally simple for now.
        Later, replace or augment with LLM/entity extraction.
        """
        message_lower = message.lower()
        entities: dict[str, Any] = {}

        budget = self._extract_budget(message_lower)
        if budget is not None:
            entities["max_price"] = budget

        bedrooms = self._extract_bedrooms(message_lower)
        if bedrooms is not None:
            entities["bedrooms"] = bedrooms

        if "sydney" in message_lower:
            entities["city"] = "Sydney"

        for property_type in ("apartment", "apartments", "unit", "units", "house", "houses"):
            if property_type in message_lower:
                normalized_type = property_type.rstrip("s")
                entities["property_type"] = normalized_type
                break

        for time_hint in ("today", "tomorrow", "this saturday", "saturday", "this sunday", "sunday", "morning", "afternoon"):
            if time_hint in message_lower:
                entities["requested_time"] = time_hint
                break

        address = self._extract_address(message)
        if address:
            entities["address"] = address

        return entities

    @staticmethod
    def _normalize(message: str) -> str:
        return " ".join(message.lower().split())

    @staticmethod
    def _contains_any(message: str, keywords: list[str]) -> bool:
        return any(keyword in message for keyword in keywords)

    @staticmethod
    def _extract_budget(message: str) -> int | None:
        """
        Examples handled:
        - under 900k
        - under 1m
        - budget 750k
        """

        pattern = r"(?:under|budget|max|below|less than)\s*\$?\s*(\d+(?:\.\d+)?)\s*(k|m)?"

        match = re.search(pattern, message.lower())
        if not match:
            return None

        value = float(match.group(1))
        suffix = match.group(2)

        if suffix == "m":
            return int(value * 1_000_000)

        if suffix == "k":
            return int(value * 1_000)

        return int(value)

    @staticmethod
    def _extract_bedrooms(message: str) -> int | None:
        """
        Examples handled:
        - 2 bedroom
        - 3 bedrooms
        - 4 bed
        """
        match = re.search(
            r"\b(\d+)\s*(?:bed|beds|bedroom|bedrooms)\b", message)
        return int(match.group(1)) if match else None

    @staticmethod
    def _extract_address(message: str) -> str | None:
        """
        Lightweight address extraction.

        Example:
        - 25 George St
        - 10 Pitt Street
        """

        match = re.search(
            r"\b\d+\s+[A-Z][a-zA-Z]+\s+(?:St|Street|Rd|Road|Ave|Avenue|Dr|Drive|Ln|Lane)\b",
            message,
        )
        return match.group(0) if match else None
