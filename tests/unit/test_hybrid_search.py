"""
Unit tests for hybrid_search_node — concurrent SQL + vector search.
Uses make_sql_service, make_rag_service, and make_config from conftest.
"""
import pytest
from langchain_core.messages import HumanMessage

from app.agents.nodes.hybrid import hybrid_search_node

_QUESTION = "Show 2 bedroom apartments in Sydney and explain the lease"


class TestHybridSearchSuccess:
    async def test_returns_search_results(self, make_sql_service, make_rag_service, make_config):
        config = make_config(make_sql_service(), make_rag_service())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "search_results" in result

    async def test_returns_retrieved_docs(self, make_sql_service, make_rag_service, make_config):
        config = make_config(make_sql_service(), make_rag_service())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], dict)
        assert "docs" in result["retrieved_docs"]

    async def test_search_results_has_one_item(self, make_sql_service, make_rag_service, make_config):
        config = make_config(make_sql_service(), make_rag_service())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert len(result["search_results"]) == 1

    async def test_retrieved_docs_contains_excerpt(self, make_sql_service, make_rag_service, make_config):
        """Vector results appear as [excerpt N] blocks in retrieved_docs["docs"]."""
        config = make_config(make_sql_service(), make_rag_service())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "[excerpt 1]" in result["retrieved_docs"]["docs"]

    async def test_calls_both_services_with_question(
        self, make_sql_service, make_rag_service, make_config
    ):
        sql = make_sql_service()
        rag = make_rag_service()
        question = "Show apartments near the CBD and their lease terms"
        await hybrid_search_node(
            {"messages": [HumanMessage(content=question)]},
            make_config(sql, rag),
        )
        sql.search_listings.assert_called_once_with(question)
        rag.aretrieve.assert_called_once_with(question, property_id=None)

    async def test_uses_context_path_when_location_set(
        self, make_sql_service, make_rag_service, make_config
    ):
        """When search_context has a location, use search_from_context (no extra LLM call)."""
        sql = make_sql_service()
        rag = make_rag_service()
        ctx = {"location": "Parramatta", "listing_type": "Rent", "limit": 3}
        await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)], "search_context": ctx},
            make_config(sql, rag),
        )
        sql.search_from_context.assert_called_once_with(ctx)
        sql.search_listings.assert_not_called()

    async def test_uses_llm_path_when_no_location(
        self, make_sql_service, make_rag_service, make_config
    ):
        """When search_context has no location, fall back to LLM SQL generation."""
        sql = make_sql_service()
        rag = make_rag_service()
        question = "Show apartments near the CBD and their lease terms"
        await hybrid_search_node(
            {"messages": [HumanMessage(content=question)], "search_context": {}},
            make_config(sql, rag),
        )
        sql.search_listings.assert_called_once_with(question)
        sql.search_from_context.assert_not_called()

    async def test_empty_sql_results(
        self, make_sql_service, make_rag_service, make_config
    ):
        """Empty SQL results → search_results is an empty list."""
        sql = make_sql_service(
            result={"success": True, "output": [], "result_count": 0})
        rag = make_rag_service()
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            make_config(sql, rag),
        )
        assert result["search_results"] == []

    async def test_empty_vector_results(
        self, make_sql_service, make_rag_service, make_config
    ):
        """Empty vector results → retrieved_docs is None (no excerpts to show)."""
        sql = make_sql_service()
        rag = make_rag_service(nodes=[])
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            make_config(sql, rag),
        )
        assert result["retrieved_docs"] is None


class TestHybridSearchPartialFailure:
    async def test_sql_fails_vector_succeeds(self, make_sql_service, make_rag_service, make_config):
        """SQL failure → empty search_results; vector still returns retrieved_docs."""
        config = make_config(
            make_sql_service(raise_error=RuntimeError("DB connection lost")),
            make_rag_service(),
        )
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert result["search_results"] == []
        assert result["retrieved_docs"] is not None

    async def test_vector_fails_sql_succeeds(self, make_sql_service, make_rag_service, make_config):
        """Vector failure → retrieved_docs is None; SQL still returns search_results."""
        config = make_config(
            make_sql_service(),
            make_rag_service(raise_error=RuntimeError("pgvector unreachable")),
        )
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert len(result["search_results"]) == 1
        assert result["retrieved_docs"] is None

    async def test_both_fail_returns_empty_state(
        self, make_sql_service, make_rag_service, make_config
    ):
        """Both services fail → empty search_results and retrieved_docs is None."""
        config = make_config(
            make_sql_service(raise_error=RuntimeError("SQL down")),
            make_rag_service(raise_error=RuntimeError("Vector down")),
        )
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert result["search_results"] == []
        assert result["retrieved_docs"] is None


class TestHybridSearchGuards:
    async def test_no_human_message_returns_empty(
        self, make_sql_service, make_rag_service, make_config
    ):
        sql = make_sql_service()
        rag = make_rag_service()
        result = await hybrid_search_node({"messages": []}, make_config(sql, rag))
        assert result == {}
        sql.search_listings.assert_not_called()
        rag.aretrieve.assert_not_called()

    async def test_missing_sql_service_raises(self, make_rag_service, make_config):
        with pytest.raises(RuntimeError):
            await hybrid_search_node(
                {"messages": [HumanMessage(content=_QUESTION)]},
                make_config(sql_service=None, rag_service=make_rag_service()),
            )

    async def test_missing_rag_retriever_raises(self, make_sql_service, make_config):
        with pytest.raises(RuntimeError):
            await hybrid_search_node(
                {"messages": [HumanMessage(content=_QUESTION)]},
                make_config(sql_service=make_sql_service(), rag_service=None),
            )

    async def test_missing_request_raises(self):
        with pytest.raises(RuntimeError):
            await hybrid_search_node(
                {"messages": [HumanMessage(content=_QUESTION)]},
                {"configurable": {}},
            )
