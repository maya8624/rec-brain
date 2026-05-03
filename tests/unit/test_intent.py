"""
Unit tests for intent_node and its helpers.

Fast path (_obvious_intent): pure keyword matching — no DB or LLM required.
LLM path (intent_node):      patched LLM — tests state mutations and entity extraction.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.messages import HumanMessage, AIMessage

from app.agents.nodes.intent import (
    _is_booking_continuation,
    _matches_keywords,
    _obvious_intent,
    intent_node,
)
from app.agents.state import IntentClassification


# ── _obvious_intent ────────────────────────────────────────────────────────────

class TestObviousIntent:
    def test_cancel_keyword(self):
        assert _obvious_intent("I want to cancel my inspection") == "cancellation"

    def test_cancellation_keyword(self):
        assert _obvious_intent("I need a cancellation") == "cancellation"

    def test_withdraw_keyword(self):
        assert _obvious_intent("I'd like to withdraw my booking") == "cancellation"

    def test_no_longer_available_keyword(self):
        assert _obvious_intent("I'm no longer available for the inspection") == "cancellation"

    def test_dont_want_to_attend_keyword(self):
        assert _obvious_intent("I don't want to attend the inspection") == "cancellation"

    def test_no_longer_want_not_cancellation(self):
        """'no longer want' alone is too vague — must fall through to LLM."""
        assert _obvious_intent("I no longer want 3 bedrooms") is None

    def test_dont_want_not_cancellation(self):
        """'don't want' alone is too vague — must fall through to LLM."""
        assert _obvious_intent("I don't want a property near a highway") is None

    def test_book_keyword_alone(self):
        assert _obvious_intent("I'd like to book an inspection") == "booking"

    def test_schedule_keyword_alone(self):
        assert _obvious_intent("Can we schedule a viewing?") == "booking"

    def test_open_home_keyword(self):
        assert _obvious_intent("When is the next open home?") == "booking"

    def test_book_with_search_returns_none(self):
        """search + booking → not obvious, needs LLM to detect search_then_book."""
        assert _obvious_intent("find houses in sydney and book an inspection") is None

    def test_cancel_with_search_returns_none(self):
        """search + cancellation → compound, needs LLM."""
        assert _obvious_intent("show me apartments and cancel my booking") is None

    def test_search_returns_none(self):
        """Search always goes to LLM for entity extraction."""
        assert _obvious_intent("Show me 3 bedroom houses in Sydney") is None

    def test_general_returns_none(self):
        assert _obvious_intent("Hello, how are you?") is None

    def test_empty_message_returns_none(self):
        assert _obvious_intent("") is None

    def test_follow_up_returns_none(self):
        assert _obvious_intent("what about his number?") is None


# ── _obvious_intent: lookup ────────────────────────────────────────────────────

class TestObviousIntentLookup:
    def test_my_booking_keyword(self):
        assert _obvious_intent("show me my booking") == "booking_lookup"

    def test_booking_status_keyword(self):
        assert _obvious_intent("what's my booking status?") == "booking_lookup"

    def test_check_my_booking_keyword(self):
        assert _obvious_intent("can you check my booking?") == "booking_lookup"

    def test_i_booked_keyword(self):
        assert _obvious_intent("I booked an inspection yesterday") == "booking_lookup"


# ── _is_booking_continuation ───────────────────────────────────────────────────

def _state_with_slots(**extra):
    return {
        "messages": [HumanMessage(content="ok")],
        "booking_context": {"available_slots": ["Mon 10am", "Tue 2pm"]},
        **extra,
    }


class TestIsBookingContinuation:
    def test_returns_true_for_neutral_slot_selection(self):
        assert _is_booking_continuation(_state_with_slots(), "the 10am one") is True

    def test_returns_true_for_option_number(self):
        assert _is_booking_continuation(_state_with_slots(), "option 2 please") is True

    def test_returns_false_when_no_booking_context(self):
        state = {"messages": [HumanMessage(content="ok")]}
        assert _is_booking_continuation(state, "the 10am one") is False

    def test_returns_false_when_slots_empty(self):
        state = {
            "messages": [HumanMessage(content="ok")],
            "booking_context": {"available_slots": []},
        }
        assert _is_booking_continuation(state, "the 10am one") is False

    def test_returns_false_when_message_has_search_keywords(self):
        assert _is_booking_continuation(_state_with_slots(), "show me cheaper properties") is False

    def test_returns_false_when_message_has_cancellation_keywords(self):
        assert _is_booking_continuation(_state_with_slots(), "cancel") is False

    def test_returns_false_when_booking_confirmed(self):
        state = _state_with_slots(booking_status={"confirmed": True, "cancelled": False, "awaiting_confirmation": False})
        assert _is_booking_continuation(state, "what are the trading hours") is False

    def test_returns_false_when_booking_cancelled(self):
        state = _state_with_slots(booking_status={"confirmed": False, "cancelled": True, "awaiting_confirmation": False})
        assert _is_booking_continuation(state, "show me more properties") is False


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

    async def test_booking_lookup_skips_llm(self):
        state = {"messages": [HumanMessage(content="show me my booking")]}
        with patch("app.agents.nodes.intent.get_llm") as mock_get_llm:
            result = await intent_node(state)
            mock_get_llm.assert_not_called()
        assert result["user_intent"] == "booking_lookup"

    async def test_booking_continuation_skips_llm(self):
        """Slot selection mid-flow returns booking without LLM call."""
        state = {
            "messages": [HumanMessage(content="the 10am one")],
            "booking_context": {"available_slots": ["Mon 10am", "Tue 2pm"]},
        }
        with patch("app.agents.nodes.intent.get_llm") as mock_get_llm:
            result = await intent_node(state)
            mock_get_llm.assert_not_called()
        assert result["user_intent"] == "booking"

    @pytest.mark.xfail(reason="fast path fires before slot check — fix pending decision")
    async def test_cancellation_mid_slot_goes_to_llm(self):
        """Cancel while slots are pending — no confirmed booking, so LLM handles it."""
        state = {
            "messages": [HumanMessage(content="cancel")],
            "booking_context": {"available_slots": ["Mon 10am", "Tue 2pm"]},
        }
        with patch("app.agents.nodes.intent.get_llm") as mock_get_llm:
            mock_get_llm.return_value = _make_llm_mock(IntentClassification(
                intent="general",
                early_response="No problem, I've cancelled the booking process.",
            ))
            result = await intent_node(state)
            mock_get_llm.assert_called()
        assert result["user_intent"] == "general"

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

    async def test_explicit_count_stored_as_limit(self, mock_get_llm):
        """'show me 3 properties' — limit=3 must flow into search_context."""
        mock_get_llm.return_value = _make_llm_mock(IntentClassification(
            intent="search",
            location="Castle Hill",
            listing_type="Rent",
            limit=3,
        ))
        state = {"messages": [HumanMessage(
            content="Show me 3 properties for rent in Castle Hill"
        )]}
        result = await intent_node(state)

        assert result["search_context"]["limit"] == 3

    async def test_limit_capped_at_10(self, mock_get_llm):
        """Even if LLM hallucinates limit=50, it must be capped at 10."""
        mock_get_llm.return_value = _make_llm_mock(IntentClassification(
            intent="search",
            location="Sydney",
            limit=50,
        ))
        state = {"messages": [HumanMessage(content="Show me 50 properties in Sydney")]}
        result = await intent_node(state)

        assert result["search_context"]["limit"] == 10
