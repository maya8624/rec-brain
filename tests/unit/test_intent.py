"""
Unit tests for intent_node and _classify_intent.

_classify_intent: pure keyword matching — no DB or LLM required.
intent_node:      async node wrapper — tests state mutation (early_response).
"""
import pytest

from app.agents.nodes.intent import _classify_intent, intent_node
from langchain_core.messages import HumanMessage


# ── _classify_intent ───────────────────────────────────────────────────────────

class TestSearchIntent:
    def test_show_keyword(self):
        assert _classify_intent(
            "Show me 3 bedroom houses in Sydney") == "search"

    def test_find_keyword(self):
        assert _classify_intent("Find apartments in Melbourne") == "search"

    def test_looking_for(self):
        assert _classify_intent(
            "I'm looking for a unit in Parramatta") == "search"

    def test_list_keyword(self):
        assert _classify_intent("List properties under $500k") == "search"

    def test_buy_keyword(self):
        assert _classify_intent(
            "I want to buy a house in Brisbane") == "search"

    def test_rent_keyword(self):
        assert _classify_intent("Show me rentals in Sydney CBD") == "search"

    def test_bedroom_keyword(self):
        assert _classify_intent("3 bedroom apartment") == "search"

    def test_price_keyword(self):
        assert _classify_intent("Properties under $800k") == "search"


class TestBookingIntent:
    def test_book_keyword(self):
        assert _classify_intent("I'd like to book an inspection") == "booking"

    def test_inspection_keyword(self):
        assert _classify_intent("Can I arrange an inspection?") == "booking"

    def test_viewing_keyword(self):
        assert _classify_intent("I'd like a viewing please") == "booking"

    def test_availability_keyword(self):
        assert _classify_intent("Is this property available?") == "booking"

    def test_open_home_keyword(self):
        assert _classify_intent("When is the next open home?") == "booking"

    def test_schedule_keyword(self):
        assert _classify_intent("Can we schedule a viewing?") == "booking"


class TestCancellationIntent:
    def test_cancel_keyword(self):
        assert _classify_intent(
            "I want to cancel my inspection") == "cancellation"

    def test_cancellation_keyword(self):
        assert _classify_intent("I need a cancellation") == "cancellation"

    def test_no_longer_keyword(self):
        assert _classify_intent(
            "I no longer want to inspect this property") == "cancellation"

    def test_withdraw_keyword(self):
        assert _classify_intent(
            "I'd like to withdraw my booking") == "cancellation"


class TestDocumentQueryIntent:
    def test_lease_keyword(self):
        assert _classify_intent(
            "What are the lease conditions?") == "document_query"

    def test_strata_keyword(self):
        assert _classify_intent(
            "Can you explain the strata report?") == "document_query"

    def test_contract_keyword(self):
        assert _classify_intent(
            "Tell me about the contract") == "document_query"

    def test_bond_keyword(self):
        assert _classify_intent(
            "What are the bond requirements?") == "document_query"

    def test_pet_policy_keyword(self):
        assert _classify_intent("What is the pet policy?") == "document_query"

    def test_break_lease_keyword(self):
        assert _classify_intent("How do I break my lease?") == "document_query"


class TestHybridSearchIntent:
    """search + document_query together → hybrid_search (not general)."""

    def test_search_and_lease(self):
        assert _classify_intent(
            "Show me 3 bedroom apartments and what are the lease terms?"
        ) == "hybrid_search"

    def test_find_and_contract(self):
        assert _classify_intent(
            "Find houses in Sydney and explain the contract"
        ) == "hybrid_search"

    def test_properties_and_strata(self):
        assert _classify_intent(
            "List properties in Parramatta and tell me about the strata"
        ) == "hybrid_search"


class TestGeneralIntent:
    def test_office_hours(self):
        # "hours" is a document_query keyword — routes to vector search for agency info
        assert _classify_intent("What are your office hours?") == "document_query"

    def test_greeting(self):
        assert _classify_intent("Hello, how are you?") == "general"

    def test_empty_message(self):
        assert _classify_intent("") == "general"

    def test_whitespace_only(self):
        assert _classify_intent("   ") == "general"

    def test_process_question(self):
        assert _classify_intent(
            "How does the rental process work?") == "general"


class TestSearchThenBookIntent:
    """search + booking compound → 'search_then_book' (not early_response refusal)."""

    def test_find_and_book(self):
        assert _classify_intent(
            "Find me houses in Sydney and book an inspection"
        ) == "general"  # _classify_intent returns general; intent_node upgrades to search_then_book

    def test_show_and_schedule(self):
        assert _classify_intent(
            "Show me apartments in Melbourne and schedule a viewing"
        ) == "general"


class TestCompoundIntent:
    """Non search+book compounds → 'general' (user must clarify)."""

    def test_search_and_cancel(self):
        assert _classify_intent(
            "Show me apartments and cancel my booking"
        ) == "general"

    def test_book_and_cancel(self):
        assert _classify_intent(
            "I want to book but also cancel my existing inspection"
        ) == "general"

    def test_search_book_cancel(self):
        assert _classify_intent(
            "Find properties, book a viewing, and cancel my old booking"
        ) == "general"


class TestIntentNode:
    """Tests for the async intent_node wrapper — verifies state mutations."""

    async def test_simple_intent_sets_user_intent(self):
        state = {"messages": [HumanMessage(
            content="Show me houses in Sydney")]}
        result = await intent_node(state)
        assert result["user_intent"] == "search"

    async def test_simple_intent_does_not_set_early_response(self):
        state = {"messages": [HumanMessage(
            content="Show me houses in Sydney")]}
        result = await intent_node(state)
        assert "early_response" not in result or result.get(
            "early_response") is None

    async def test_search_and_book_sets_search_then_book(self):
        state = {"messages": [HumanMessage(
            content="Find me houses in Sydney and book an inspection"
        )]}
        result = await intent_node(state)
        assert result["user_intent"] == "search_then_book"
        assert not result.get("early_response")

    async def test_search_and_book_does_not_set_early_response(self):
        state = {"messages": [HumanMessage(
            content="Show me apartments in Parramatta and schedule a viewing"
        )]}
        result = await intent_node(state)
        assert result["user_intent"] == "search_then_book"
        assert not result.get("early_response")

    async def test_search_and_cancel_still_sets_early_response(self):
        """search + cancellation is not search_then_book — user must clarify."""
        state = {"messages": [HumanMessage(
            content="Show me apartments and cancel my booking"
        )]}
        result = await intent_node(state)
        assert result["user_intent"] == "general"
        assert result.get("early_response")

    async def test_booking_intent_no_early_response(self):
        state = {"messages": [HumanMessage(
            content="I'd like to book an inspection")]}
        result = await intent_node(state)
        assert result["user_intent"] == "booking"
        assert not result.get("early_response")

    async def test_empty_state_messages(self):
        state = {"messages": []}
        result = await intent_node(state)
        assert result["user_intent"] == "general"

    async def test_vague_search_without_location_sets_early_response(self):
        """Search with no location → ask for suburb before running SQL."""
        state = {"messages": [HumanMessage(content="Find apartments with 2 bathrooms for rent")]}
        result = await intent_node(state)
        assert result["user_intent"] == "search"
        assert result.get("early_response")

    async def test_search_with_location_does_not_set_early_response(self):
        state = {"messages": [HumanMessage(content="Find apartments in Melbourne")]}
        result = await intent_node(state)
        assert result["user_intent"] == "search"
        assert not result.get("early_response")

    async def test_search_with_preposition_location_passes(self):
        """'near the CBD' matches the preposition pattern."""
        state = {"messages": [HumanMessage(content="Show me houses near the CBD")]}
        result = await intent_node(state)
        assert not result.get("early_response")

    async def test_search_with_state_abbreviation_passes(self):
        state = {"messages": [HumanMessage(content="Show me houses in NSW")]}
        result = await intent_node(state)
        assert not result.get("early_response")
