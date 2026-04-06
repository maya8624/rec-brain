"""
Shared fixtures available to every test in the suite.

All factories follow the "fixture-as-factory" pattern:
    the fixture returns a factory function so tests can call it
    with custom arguments while still getting pytest's dependency injection.
"""
import json

import pytest
from langchain_core.messages import HumanMessage
from unittest.mock import AsyncMock, MagicMock


# ── Vector / RAG factories ─────────────────────────────────────────────────────

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
        if raise_error:
            mock.search_listings.side_effect = raise_error
        else:
            mock.search_listings.return_value = result or {
                "success": True,
                "output": [
                    {"address": "12 Park Ave, Sydney", "price": 750_000, "bedrooms": 3},
                ],
                "result_count": 1,
                "sql_used": "SELECT * FROM v_listings WHERE suburb = 'Sydney'",
            }
        return mock
    return _factory


# ── Booking factories ──────────────────────────────────────────────────────────

@pytest.fixture
def make_booking_service():
    """
    Factory for a mock BookingService.

    get_availability returns a list of slot dicts — this matches what
    the check_availability tool iterates over (the tool wraps them into
    AvailableSlot objects itself).
    """
    from app.services.booking_service import BookingService

    def _factory(
        availability: list | None = None,
        booking_result: dict | None = None,
        raise_error: Exception | None = None,
    ):
        mock = AsyncMock(spec=BookingService)
        if raise_error:
            mock.get_availability.side_effect = raise_error
            mock.book.side_effect = raise_error
            mock.cancel.side_effect = raise_error
        else:
            mock.get_availability.return_value = (
                availability if availability is not None else [
                    {"datetime": "2026-04-12 10:00", "agent_name": "Jane Smith", "available": True},
                    {"datetime": "2026-04-12 14:00", "agent_name": "Jane Smith", "available": True},
                ]
            )
            mock.book.return_value = booking_result or {
                "confirmation_id": "CONF-12345",
                "property_address": "123 Main St, Sydney NSW 2000",
                "confirmed_datetime": "2026-04-12 10:00",
                "agent_name": "Jane Smith",
                "agent_phone": "0412 345 678",
            }
            mock.cancel.return_value = {
                "confirmation_id": "CONF-12345",
                "message": "Booking successfully cancelled",
            }
        return mock
    return _factory


# ── LangGraph config / state factories ────────────────────────────────────────

@pytest.fixture
def make_config():
    """Factory for a LangGraph RunnableConfig with services on app.state."""
    def _factory(
        sql_service=None,
        rag_retriever=None,
        booking_service=None,
        thread_id: str = "test-thread",
    ) -> dict:
        request = MagicMock()
        request.app.state.sql_view_service = sql_service
        request.app.state.rag_retriever = rag_retriever
        request.app.state.booking_service = booking_service
        return {"configurable": {"thread_id": thread_id, "request": request}}
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


# ── Parsing helpers ────────────────────────────────────────────────────────────

def parsed(result: dict) -> dict:
    """Parse the JSON content of the first SystemMessage in a node result dict."""
    return json.loads(result["messages"][0].content)
