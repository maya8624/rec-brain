"""
LangGraph flow tests.

Runs the compiled graph with mocked services where possible.
Skip with: pytest -m unit
Run with:  pytest -m integration
"""
from unittest.mock import AsyncMock
import pytest
from langchain_core.messages import HumanMessage

from tests.integration.conftest import skip_if_no_env

from app.infrastructure.llm import get_llm
from app.services.sql_service import SqlViewService
from app.services.rag_service import RagRetriever
from app.infrastructure.pgvector_store import PgVectorStoreService
from app.infrastructure.embedding import EmbeddingService

pytestmark = [pytest.mark.integration, skip_if_no_env]


@pytest.fixture(scope="module")
def graph():
    from app.agents.graph import build_graph
    return build_graph()


def make_booking_service():
    mock = AsyncMock()
    mock.get_availability.return_value = [
        {
            "datetime": "2027-04-12 10:00",
            "agent_name": "Jane Smith",
            "available": True
        },
        {
            "datetime": "2027-04-12 14:00",
            "agent_name": "Jane Smith",
            "available": True
        },
    ]

    mock.book.return_value = {
        "confirmation_id": "CONF-12345",
        "property_address": "123 Main St, Sydney NSW 2000",
        "confirmed_datetime": "2027-04-12 10:00",
        "agent_name": "Jane Smith",
        "agent_phone": "0412 345 678",
    }

    mock.cancel.return_value = {"success": True}

    return mock


def make_request_mock(booking_service=None, sql_view_service=None):
    mock = AsyncMock()
    mock.app.state.sql_view_service = sql_view_service
    mock.app.state.booking_service = booking_service
    mock.app.state.rag_retriever = None
    return mock


def get_config(request, thread_id: str = "integ-thread") -> dict:
    return {"configurable": {"thread_id": thread_id, "request": request}}


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
            {
                "messages": [HumanMessage(
                    content="I'd like to book an inspection for property prop_123"
                )]
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

    async def test_compound_intent_sets_early_response(self, graph):
        request = make_request_mock()
        result = await graph.ainvoke(
            {
                "messages": [HumanMessage(content="Find me houses in Sydney and book an inspection")]
            },
            config=get_config(request, "integ-compound"),
        )

        assert result["user_intent"] == "general"
        assert result.get("early_response")

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
