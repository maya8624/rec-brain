"""
LangGraph flow tests.

Runs the compiled graph with mocked services where possible.
Skip with: pytest -m unit
Run with:  pytest -m integration
"""
from unittest.mock import AsyncMock
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from tests.integration.conftest import skip_if_no_env

from app.agents.graph import build_graph
from app.infrastructure.checkpointer import PostgresCheckpointer
from app.infrastructure.llm import get_llm
from app.services.sql_service import SqlViewService
from app.services.rag_service import RagRetriever
from app.infrastructure.pgvector_store import PgVectorStoreService
from app.infrastructure.embedding import EmbeddingService

pytestmark = [pytest.mark.integration, skip_if_no_env]


@pytest.fixture(scope="module")
async def graph():
    checkpointer = await PostgresCheckpointer.create()
    yield build_graph(checkpointer.instance)
    await checkpointer.close()


def make_booking_service():
    from app.schemas.booking import AvailabilityResult, AvailableSlot, BookingConfirmation, CancellationConfirmation

    mock = AsyncMock()
    mock.check_availability.return_value = AvailabilityResult(
        success=True,
        property_id="prop_123",
        available_slots=[
            AvailableSlot(startAtUtc="2027-04-12T10:00:00Z", endAtUtc="2027-04-12T10:30:00Z", status="open", capacity=1),
            AvailableSlot(startAtUtc="2027-04-12T14:00:00Z", endAtUtc="2027-04-12T14:30:00Z", status="open", capacity=1),
        ],
        slot_count=2,
    )
    mock.book.return_value = BookingConfirmation(
        confirmation_id="CONF-12345",
        property_id="prop_123",
        agent_first_name="Jane",
        agent_last_name="Smith",
        agent_phone="0412 345 678",
    )
    mock.cancel.return_value = CancellationConfirmation(
        id="CONF-12345",
        success=True,
    )
    return mock


def make_request_mock(booking_service=None, sql_view_service=None):
    mock = AsyncMock()
    mock.app.state.sql_view_service = sql_view_service
    mock.app.state.booking_service = booking_service
    mock.app.state.rag_retriever = None
    return mock


def get_config(request, thread_id: str = "integ-thread") -> dict:
    return {
        "configurable": {
            "thread_id":        thread_id,
            "booking_service":  request.app.state.booking_service,
            "sql_view_service": request.app.state.sql_view_service,
            "rag_retriever":    request.app.state.rag_retriever,
        }
    }


class TestGraphFlows:
    async def test_general_intent_flow(self, graph):
        request = make_request_mock()
        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="Hi, can you help me?")]},
            config=get_config(request, "integ-general"),
        )

        assert result["user_intent"] == "general"
        assert len(result["messages"]) > 0

    async def test_agency_info_intent_flow(self, graph):
        rag_retriever = RagRetriever(
            vector_store_service=PgVectorStoreService(),
            embedding_service=EmbeddingService(),
        )
        request = make_request_mock()
        request.app.state.rag_retriever = rag_retriever

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What are your office hours?")]
            },
            config=get_config(request, "integ-agency-info"),
        )

        assert result["user_intent"] == "document_query"
        assert len(result["messages"]) > 0

        last_message = result["messages"][-1]
        assert last_message.content
        assert len(last_message.content) > 20

    async def test_booking_intent_flow(self, graph):
        request = make_request_mock(booking_service=make_booking_service())
        result = await graph.ainvoke(
            {"messages": [HumanMessage(
                content="I'd like to book an inspection for property 08d1202e-cd7e-d6cc-f2b3-c309f377d123")]
             },
            config=get_config(request, "integ-booking"),
        )

        assert result["user_intent"] == "booking"

    async def test_cancellation_intent_flow(self, graph):
        request = make_request_mock(booking_service=make_booking_service())
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="I want to cancel my inspection booking CONF-12345")]
            },
            config=get_config(request, "integ-cancel"),
        )

        assert result["user_intent"] == "cancellation"

    async def test_search_then_book_intent_flow(self, graph):
        sql_service = SqlViewService(llm=get_llm())
        request = make_request_mock(sql_view_service=sql_service)

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(
                    content="Find me 2 bedroom apartments in Sydney and book an inspection"
                )]
            },
            config=get_config(request, "integ-search-then-book"),
        )

        # Turn 1: only search runs — no booking, no early_response refusal
        assert result["user_intent"] == "search_then_book"
        assert not result.get("early_response")
        assert len(result["messages"]) > 0

    async def test_search_then_book_full_flow(self, graph):
        """Two-turn flow: Turn 1 searches, Turn 2 books a specific property."""
        sql_service = SqlViewService(llm=get_llm())
        booking_service = make_booking_service()
        request = make_request_mock(
            booking_service=booking_service,
            sql_view_service=sql_service,
        )
        config = get_config(request, "integ-search-then-book-full")

        # Turn 1 — search runs, agent prompts user to pick a property
        result1 = await graph.ainvoke(
            {"messages": [HumanMessage(
                content="Find me 2 bedroom apartments in Sydney and book an inspection")]},
            config=config,
        )

        assert result1["user_intent"] == "search_then_book"
        assert not result1.get("early_response")
        assert len(result1["messages"]) > 0

        # Turn 2 — user picks a specific property to book
        result2 = await graph.ainvoke(
            {"messages": [HumanMessage(
                content="Book an inspection for property 08d1202e-cd7e-d6cc-f2b3-c309f377d123")]},
            config=config,
        )

        assert result2["user_intent"] == "booking"
        # Conversation history from Turn 1 is preserved
        assert len(result2["messages"]) > len(result1["messages"])

        # check_availability ran — slots should be in booking_context
        booking_ctx = result2.get("booking_context", {})
        assert booking_ctx.get(
            "available_slots"), "expected available_slots after check_availability"

        last_message = result2["messages"][-1]
        assert last_message.content

    async def test_compound_intent_sets_early_response(self, graph):
        request = make_request_mock()
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Book an inspection and cancel my booking CONF-12345")]
            },
            config=get_config(request, "integ-compound"),
        )

        assert result["user_intent"] == "general"
        assert result.get("early_response")
        assert "one request at a time" in result["early_response"].lower()
        # Graph must end without calling the LLM — no AIMessage should be added
        assert not any(isinstance(m, AIMessage) for m in result["messages"])

    async def test_search_intent_flow(self, graph):
        sql_service = SqlViewService(llm=get_llm())
        request = make_request_mock(sql_view_service=sql_service)

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Show me 3 bedroom houses in Sydney under $800k")]
            },
            config=get_config(request, "integ-search"),
        )

        assert result["user_intent"] == "search"

    async def test_document_query_intent_flow(self, graph):
        rag_retriever = RagRetriever(
            vector_store_service=PgVectorStoreService(),
            embedding_service=EmbeddingService(),
        )
        request = make_request_mock()
        request.app.state.rag_retriever = rag_retriever

        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="What are the break lease conditions?")]
            },
            config=get_config(request, "integ-doc-query"),
        )

        assert result["user_intent"] == "document_query"
        assert len(result["messages"]) > 0

        last_message = result["messages"][-1]
        assert last_message.content
        assert len(last_message.content) > 50

        # Second turn — same thread_id continues the conversation
        result2 = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Tell me about the buying process")]
            },
            config=get_config(request, "integ-doc-query"),
        )

        assert result2["user_intent"] == "general"
        assert len(result2["messages"]) > 2  # history from turn 1 is preserved

    async def test_result_always_has_messages(self, graph):
        request = make_request_mock()

        result = await graph.ainvoke(
            {"messages": [HumanMessage(content="hello")]},
            config=get_config(request, "integ-messages-check"),
        )

        assert "messages" in result
        assert len(result["messages"]) > 0
