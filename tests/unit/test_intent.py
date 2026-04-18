"""
Unit tests for intent_node and its helpers.

Fast path (_obvious_intent): pure keyword matching — no DB or LLM required.
LLM path (intent_node):      patched LLM — tests state mutations and entity extraction.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from app.agents.nodes.intent import _obvious_intent, _matches_keywords, intent_node
from app.agents.state import IntentClassification


# ── _obvious_intent ────────────────────────────────────────────────────────────

class TestObviousIntent:
    def test_cancel_keyword(self):
        assert _obvious_intent("I want to cancel my inspection") == "cancellation"

    def test_cancellation_keyword(self):
        assert _obvious_intent("I need a cancellation") == "cancellation"

    def test_withdraw_keyword(self):
        assert _obvious_intent("I'd like to withdraw my booking") == "cancellation"

    def test_no_longer_keyword(self):
        assert _obvious_intent("I no longer want to inspect this property") == "cancellation"

    def test_book_keyword_alone(self):
        assert _obvious_intent("I'd like to book an inspection") == "booking"

    def test_schedule_keyword_alone(self):
        assert _obvious_intent("Can we schedule a viewing?") == "booking"

    def test_open_home_keyword(self):
        assert _obvious_intent("When is the next open home?") == "booking"

    def test_book_with_search_returns_none(self):
        """search + booking → not obvious, needs LLM to detect search_then_book."""
        assert _obvious_intent("Find houses in Sydney and book an inspection") is None

    def test_cancel_with_search_returns_none(self):
        """search + cancellation → compound, needs LLM."""
        assert _obvious_intent("Show me apartments and cancel my booking") is None

    def test_search_returns_none(self):
        """Search always goes to LLM for entity extraction."""
        assert _obvious_intent("Show me 3 bedroom houses in Sydney") is None

    def test_general_returns_none(self):
        assert _obvious_intent("Hello, how are you?") is None

    def test_empty_message_returns_none(self):
        assert _obvious_intent("") is None

    def test_follow_up_returns_none(self):
        assert _obvious_intent("what about his number?") is None


# ── intent_node fast path ──────────────────────────────────────────────────────

class TestIntentNodeFastPath:
    async def test_cancellation_skips_llm(self):
        state = {"messages": [HumanMessage(content="cancel my inspection")]}
        with patch("app.agents.nodes.intent.get_llm") as mock_get_llm:
            result = await intent_node(state)
            mock_get_llm.assert_not_called()
        assert result["user_intent"] == "cancellation"

    async def test_booking_skips_llm(self):
        state = {"messages": [HumanMessage(content="I'd like to book a viewing")]}
        with patch("app.agents.nodes.intent.get_llm") as mock_get_llm:
            result = await intent_node(state)
            mock_get_llm.assert_not_called()
        assert result["user_intent"] == "booking"

    async def test_empty_messages_returns_general(self):
        state = {"messages": []}
        result = await intent_node(state)
        assert result["user_intent"] == "general"


# ── intent_node LLM path ───────────────────────────────────────────────────────

def _make_llm_mock(classification: IntentClassification):
    """Returns a mock get_llm() that produces the given IntentClassification."""
    structured = AsyncMock(return_value=classification)
    llm = MagicMock()
    llm.with_structured_output.return_value = MagicMock(ainvoke=structured)
    return llm


@patch("app.agents.nodes.intent.get_llm")
class TestIntentNodeLLMPath:
    async def test_search_intent_with_entities(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock(IntentClassification(
            intent="search",
            location="Sydney",
            property_type="House",
            bedrooms=3,
            max_price=800000,
        ))
        state = {"messages": [HumanMessage(content="Show me 3 bedroom houses in Sydney under $800k")]}
        result = await intent_node(state)

        assert result["user_intent"] == "search"
        assert result["search_context"]["location"] == "Sydney"
        assert result["search_context"]["property_type"] == "House"
        assert result["search_context"]["bedrooms"] == 3
        assert result["search_context"]["max_price"] == 800000

    async def test_search_without_location_sets_early_response(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock(IntentClassification(
            intent="search",
            early_response="Which suburb or area are you looking in?",
        ))
        state = {"messages": [HumanMessage(content="Find apartments with 2 bathrooms for rent")]}
        result = await intent_node(state)

        assert result["user_intent"] == "search"
        assert result["early_response"]

    async def test_follow_up_resolved_from_history(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock(IntentClassification(
            intent="document_query",
        ))
        state = {"messages": [
            HumanMessage(content="Who is the principal agent?"),
            AIMessage(content="Sam Jones is the principal agent."),
            HumanMessage(content="what is his number?"),
        ]}
        result = await intent_node(state)
        assert result["user_intent"] == "document_query"

    async def test_search_then_book_intent(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock(IntentClassification(
            intent="search_then_book",
            location="Sydney",
        ))
        state = {"messages": [HumanMessage(
            content="Find me houses in Sydney and book an inspection"
        )]}
        result = await intent_node(state)
        assert result["user_intent"] == "search_then_book"

    async def test_compound_sets_early_response(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock(IntentClassification(
            intent="general",
            early_response="I can only handle one request at a time. "
                           "Would you like to search or cancel a booking?",
        ))
        state = {"messages": [HumanMessage(
            content="Show me apartments and cancel my booking"
        )]}
        result = await intent_node(state)
        assert result["user_intent"] == "general"
        assert result["early_response"]

    async def test_entities_merged_with_existing_search_context(self, mock_get_llm):
        """New entities merge with existing context — previous filters preserved."""
        mock_get_llm.return_value = _make_llm_mock(IntentClassification(
            intent="search",
            max_price=600000,
        ))
        state = {
            "messages": [HumanMessage(content="actually make it under $600k")],
            "search_context": {"location": "Sydney", "bedrooms": 3},
        }
        result = await intent_node(state)

        ctx = result["search_context"]
        assert ctx["location"] == "Sydney"    # preserved from previous turn
        assert ctx["bedrooms"] == 3           # preserved
        assert ctx["max_price"] == 600000     # new from this turn

    async def test_no_entities_does_not_write_search_context(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock(IntentClassification(
            intent="general",
        ))
        state = {"messages": [HumanMessage(content="Hello!")]}
        result = await intent_node(state)
        assert "search_context" not in result

    async def test_llm_failure_falls_back_to_general(self, mock_get_llm):
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock(
            ainvoke=AsyncMock(side_effect=RuntimeError("Groq down"))
        )
        mock_get_llm.return_value = llm
        state = {"messages": [HumanMessage(content="show me something nice")]}
        result = await intent_node(state)
        assert result["user_intent"] == "general"
