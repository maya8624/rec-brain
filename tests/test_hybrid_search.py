"""
Tests for hybrid_search_node — concurrent SQL + vector search.
No real DB, LLM, or embedding model required — services are mocked.

Usage:
    pytest tests/test_hybrid_search.py
    pytest tests/test_hybrid_search.py -v
    pytest tests/test_hybrid_search.py -k both_fail
"""
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes.hybrid import hybrid_search_node


# ── Factories ──────────────────────────────────────────────────────────────────

def make_node(text: str, score: float = 0.85, metadata: dict | None = None):
    """Build a mock NodeWithScore — avoids importing LlamaIndex schema."""
    node = MagicMock()
    node.get_content.return_value = text
    node.metadata = metadata or {}
    return MagicMock(node=node, score=score)


def make_sql_service(result: dict | None = None, raise_error: Exception | None = None):
    """
    Factory for a mock SqlViewService.
    Pass raise_error to simulate search_listings failures.
    """
    mock = AsyncMock()
    if raise_error:
        mock.search_listings.side_effect = raise_error
    else:
        mock.search_listings.return_value = result or {
            "success": True,
            "output": [
                {"address": "12 Park Ave, Sydney", "price": 750000, "bedrooms": 3},
                {"address": "5 Elm St, Sydney", "price": 620000, "bedrooms": 2},
            ],
            "result_count": 2,
            "sql_used": "SELECT * FROM v_listings WHERE suburb = 'Sydney'",
        }
    return mock


def make_rag_retriever(nodes: list | None = None, raise_error: Exception | None = None):
    """
    Factory for a mock RagRetriever.
    Pass raise_error to simulate aretrieve failures.
    """
    mock = AsyncMock()
    if raise_error:
        mock.aretrieve.side_effect = raise_error
    else:
        mock.aretrieve.return_value = nodes if nodes is not None else [
            make_node("Lease requires 4 weeks bond.", score=0.91,
                      metadata={"property_id": "prop_1", "doc_type": "lease"}),
        ]
    return mock


def make_config(sql_service=None, rag_retriever=None):
    """Build a RunnableConfig with both services on app.state."""
    request = MagicMock()
    request.app.state.sql_view_service = sql_service
    request.app.state.rag_retriever = rag_retriever
    return {"configurable": {"request": request}}


def make_state(question: str = "Show 2 bedroom apartments in Sydney and explain the lease") -> dict:
    return {"messages": [HumanMessage(content=question)]}


def parsed(result: dict) -> dict:
    """Parse the SystemMessage content from a node result."""
    return json.loads(result["messages"][0].content)


# ── Success paths ──────────────────────────────────────────────────────────────

class TestHybridSearchSuccess:
    async def test_returns_system_message(self):
        """Node returns a SystemMessage when both services succeed."""
        config = make_config(make_sql_service(), make_rag_retriever())

        result = await hybrid_search_node(make_state(), config)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)

    async def test_result_contains_both_sections(self):
        """SystemMessage content contains sql_results and vector_results."""
        config = make_config(make_sql_service(), make_rag_retriever())

        result = await hybrid_search_node(make_state(), config)
        content = parsed(result)

        assert "sql_results" in content
        assert "vector_results" in content

    async def test_sql_results_structure(self):
        """sql_results reflects what SqlViewService returned."""
        config = make_config(make_sql_service(), make_rag_retriever())

        result = await hybrid_search_node(make_state(), config)
        sql = parsed(result)["sql_results"]

        assert sql["success"] is True
        assert sql["result_count"] == 2

    async def test_vector_results_structure(self):
        """vector_results contains success flag, result_count, and results list."""
        config = make_config(make_sql_service(), make_rag_retriever())

        result = await hybrid_search_node(make_state(), config)
        vector = parsed(result)["vector_results"]

        assert vector["success"] is True
        assert vector["result_count"] == 1
        assert len(vector["results"]) == 1

    async def test_vector_result_item_shape(self):
        """Each vector result item has text, score, and metadata."""
        config = make_config(make_sql_service(), make_rag_retriever())

        result = await hybrid_search_node(make_state(), config)
        item = parsed(result)["vector_results"]["results"][0]

        assert "text" in item
        assert "score" in item
        assert "metadata" in item

    async def test_calls_both_services_with_question(self):
        """Both services receive the exact question from HumanMessage."""
        sql = make_sql_service()
        rag = make_rag_retriever()
        question = "Show apartments near the CBD and their lease terms"
        config = make_config(sql, rag)

        await hybrid_search_node(make_state(question), config)

        sql.search_listings.assert_called_once_with(question)
        rag.aretrieve.assert_called_once_with(question)

    async def test_empty_sql_and_vector_results(self):
        """Returns SystemMessage even when both services return empty results."""
        sql = make_sql_service(result={
            "success": True, "output": [], "result_count": 0, "sql_used": ""
        })
        rag = make_rag_retriever(nodes=[])
        config = make_config(sql, rag)

        result = await hybrid_search_node(make_state(), config)
        content = parsed(result)

        assert content["sql_results"]["result_count"] == 0
        assert content["vector_results"]["result_count"] == 0


# ── Partial failure paths ──────────────────────────────────────────────────────

class TestHybridSearchPartialFailure:
    async def test_sql_fails_vector_succeeds(self):
        """SQL exception is captured; vector results still returned."""
        sql = make_sql_service(raise_error=RuntimeError("DB connection lost"))
        config = make_config(sql, make_rag_retriever())

        result = await hybrid_search_node(make_state(), config)
        content = parsed(result)

        assert content["sql_results"]["success"] is False
        assert "error" in content["sql_results"]
        assert content["vector_results"]["success"] is True

    async def test_vector_fails_sql_succeeds(self):
        """Vector exception is captured; SQL results still returned."""
        rag = make_rag_retriever(
            raise_error=RuntimeError("pgvector unreachable"))
        config = make_config(make_sql_service(), rag)

        result = await hybrid_search_node(make_state(), config)
        content = parsed(result)

        assert content["sql_results"]["success"] is True
        assert content["vector_results"]["success"] is False
        assert "error" in content["vector_results"]

    async def test_both_fail_still_returns_message(self):
        """Both exceptions captured; SystemMessage returned with both errors."""
        sql = make_sql_service(raise_error=RuntimeError("SQL down"))
        rag = make_rag_retriever(raise_error=RuntimeError("Vector down"))
        config = make_config(sql, rag)

        result = await hybrid_search_node(make_state(), config)
        content = parsed(result)

        assert content["sql_results"]["success"] is False
        assert content["vector_results"]["success"] is False
        assert "error" in content["sql_results"]
        assert "error" in content["vector_results"]

    async def test_error_message_is_captured(self):
        """The exception message is included in the error field."""
        sql = make_sql_service(raise_error=RuntimeError("timeout after 30s"))
        config = make_config(sql, make_rag_retriever())

        result = await hybrid_search_node(make_state(), config)
        content = parsed(result)

        assert "timeout after 30s" in content["sql_results"]["error"]


# ── Guard paths ────────────────────────────────────────────────────────────────

class TestHybridSearchGuards:
    async def test_no_human_message_returns_empty(self):
        """Returns {} without calling either service when state has no HumanMessage."""
        sql = make_sql_service()
        rag = make_rag_retriever()
        config = make_config(sql, rag)

        result = await hybrid_search_node({"messages": []}, config)

        assert result == {}
        sql.search_listings.assert_not_called()
        rag.aretrieve.assert_not_called()

    async def test_missing_sql_service_returns_empty(self):
        """Returns {} when sql_view_service is None on app.state."""
        config = make_config(
            sql_service=None, rag_retriever=make_rag_retriever())

        result = await hybrid_search_node(make_state(), config)

        assert result == {}

    async def test_missing_rag_retriever_returns_empty(self):
        """Returns {} when rag_retriever is None on app.state."""
        config = make_config(
            sql_service=make_sql_service(), rag_retriever=None)

        result = await hybrid_search_node(make_state(), config)

        assert result == {}

    async def test_missing_request_returns_empty(self):
        """Returns {} when request is absent from configurable."""
        result = await hybrid_search_node(make_state(), {"configurable": {}})

        assert result == {}
