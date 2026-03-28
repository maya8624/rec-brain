"""
Tests for intent_node classification.
No DB or LLM required — pure keyword matching.
"""
from app.agents.nodes.intent import _classify_intent


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


class TestGeneralIntent:
    def test_office_hours(self):
        assert _classify_intent("What are your office hours?") == "general"

    def test_greeting(self):
        assert _classify_intent("Hello, how are you?") == "general"

    def test_empty_message(self):
        assert _classify_intent("") == "general"

    def test_whitespace_only(self):
        assert _classify_intent("   ") == "general"

    def test_process_question(self):
        assert _classify_intent(
            "How does the rental process work?") == "general"


class TestCompoundIntent:
    """Compound intents should fall through to 'general'."""

    def test_search_and_book(self):
        assert _classify_intent(
            "Find me houses in Sydney and book an inspection"
        ) == "general"

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
