"""
Shared fixtures available to every test in the suite.

All factories follow the "fixture-as-factory" pattern:
    the fixture returns a factory function so tests can call it
    with custom arguments while still getting pytest's dependency injection.
"""
import json

from unittest.mock import AsyncMock, MagicMock
import pytest
from langchain_core.messages import HumanMessage


@pytest.fixture
def make_node():
    """Factory for mock LlamaIndex NodeWithScore objects."""
    def _factory(text: str, score: float = 0.85, metadata: dict | None = None):
        node = MagicMock()
        node.get_content.return_value = text
        node.metadata = metadata or {}
        return MagicMock(node=node, score=score)
    return _factory


@pytest.fixture
def make_rag_retriever(make_node):
    """Factory for a mock RagRetriever (LlamaIndex BaseRetriever)."""
    def _factory(nodes: list | None = None, raise_error: Exception | None = None):
        mock = AsyncMock()
        if raise_error:
            mock.aretrieve.side_effect = raise_error
        else:
            mock.aretrieve.return_value = nodes if nodes is not None else [
                make_node(
                    "A lease agreement requires 4 weeks bond.",
                    score=0.92,
                    metadata={"property_id": "prop_1", "doc_type": "lease"},
                ),
            ]
        return mock
    return _factory


# ── SQL / search factories ─────────────────────────────────────────────────────

@pytest.fixture
def make_sql_service():
    """Factory for a mock SqlViewService."""
    def _factory(result: dict | None = None, raise_error: Exception | None = None):
        mock = AsyncMock()
        default_result = result or {
            "success": True,
            "output": [
                {"address": "12 Park Ave, Sydney",
                    "price": 750_000, "bedrooms": 3},
            ],
            "result_count": 1,
            "sql_used": "SELECT * FROM v_listings WHERE suburb = 'Sydney'",
        }
        if raise_error:
            mock.search_listings.side_effect = raise_error
            mock.search_from_context.side_effect = raise_error
        else:
            mock.search_listings.return_value = default_result
            mock.search_from_context.return_value = default_result
        return mock
    return _factory


# ── Booking factories ──────────────────────────────────────────────────────────

@pytest.fixture
def make_booking_service():
    """Factory for a mock BookingService."""
    from app.services.booking_service import BookingService
    from app.schemas.booking import AvailabilityResult, AvailableSlot, CancellationConfirmation

    def _factory(
        availability: AvailabilityResult | None = None,
        booking_result: dict | None = None,
        raise_error: Exception | None = None,
    ):
        mock = AsyncMock(spec=BookingService)
        if raise_error:
            mock.get_availability.side_effect = raise_error
            mock.book.side_effect = raise_error
            mock.cancel.side_effect = raise_error
        else:
            mock.get_availability.return_value = availability or AvailabilityResult(
                success=True,
                property_id="prop_123",
                available_slots=[
                    AvailableSlot(agent_name="Jane Smith", startAtUtc="2026-04-12T10:00:00Z", status="open", capacity=1),
                    AvailableSlot(agent_name="Jane Smith", startAtUtc="2026-04-12T14:00:00Z", status="open", capacity=1),
                ],
                slot_count=2,
            )
            mock.book.return_value = booking_result or {
                "confirmation_id": "CONF-12345",
                "property_id": "prop_123",
                "agent_first_name": "Jane",
                "agent_last_name": "Smith",
                "agent_phone": "0412 345 678",
            }
            mock.cancel.return_value = CancellationConfirmation(
                id="CONF-12345",
                success=True,
            )
        return mock
    return _factory


# ── LangGraph config / state factories ────────────────────────────────────────

@pytest.fixture
def make_config():
    """Factory for a LangGraph RunnableConfig with services in configurable directly."""
    def _factory(
        sql_service=None,
        rag_retriever=None,
        booking_service=None,
        thread_id: str = "test-thread",
    ) -> dict:
        return {
            "configurable": {
                "thread_id":        thread_id,
                "sql_view_service": sql_service,
                "rag_retriever":    rag_retriever,
                "booking_service":  booking_service,
            }
        }
    return _factory


@pytest.fixture
def make_state():
    """Factory for a minimal RealEstateAgentState dict."""
    def _factory(
        question: str = "How can I help you?",
        intent: str = "general",
        messages: list | None = None,
        error_count: int = 0,
        requires_human: bool = False,
        **kwargs,
    ) -> dict:
        state: dict = {
            "messages": messages if messages is not None else [HumanMessage(content=question)],
            "user_intent": intent,
            "error_count": error_count,
            "requires_human": requires_human,
        }
        state.update(kwargs)
        return state
    return _factory


def parsed(result: dict) -> dict:
    """Parse the JSON content of the first SystemMessage in a node result dict."""
    return json.loads(result["messages"][0].content)
