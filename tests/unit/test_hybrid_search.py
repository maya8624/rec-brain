"""
Unit tests for hybrid_search_node — concurrent SQL + vector search.
Uses make_sql_service, make_rag_service, and make_config from conftest.
"""
import json

from langchain_core.messages import HumanMessage

from app.agents.nodes.hybrid import hybrid_search_node

_QUESTION = "Show 2 bedroom apartments in Sydney and explain the lease"


def _parse_vector(result: dict) -> dict:
    """Extract the vector JSON payload from the DOCUMENTS section of retrieved_docs."""
    docs_section = result["retrieved_docs"].split("DOCUMENTS:\n", 1)[1]
    return json.loads(docs_section)["vector_results"]


class TestHybridSearchSuccess:
    async def test_returns_retrieved_docs(self, make_sql_service, make_rag_service, make_config):
        config = make_config(make_sql_service(), make_rag_service())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], str)

    async def test_retrieved_docs_contains_both_sections(self, make_sql_service, make_rag_service, make_config):
        config = make_config(make_sql_service(), make_rag_service())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "LISTINGS:" in result["retrieved_docs"]
        assert "DOCUMENTS:" in result["retrieved_docs"]

    async def test_search_results_returned(self, make_sql_service, make_rag_service, make_config):
        config = make_config(make_sql_service(), make_rag_service())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "search_results" in result
        assert len(result["search_results"]) == 1

    async def test_vector_results_structure(self, make_sql_service, make_rag_service, make_config):
        config = make_config(make_sql_service(), make_rag_service())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        vector = _parse_vector(result)
        assert vector["success"] is True
        assert vector["result_count"] == 1
        assert len(vector["results"]) == 1

    async def test_vector_result_item_shape(self, make_sql_service, make_rag_service, make_config):
        config = make_config(make_sql_service(), make_rag_service())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        item = _parse_vector(result)["results"][0]
        assert "text" in item
        assert "score" in item
        assert "metadata" in item

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
        rag.aretrieve.assert_called_once_with(question)

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

    async def test_empty_results_from_both_services(
        self, make_sql_service, make_rag_service, make_config
    ):
        sql = make_sql_service(
            result={"success": True, "output": [], "result_count": 0, "sql_used": ""})
        rag = make_rag_service(nodes=[])
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            make_config(sql, rag),
        )
        assert "No listings found." in result["retrieved_docs"]
        assert _parse_vector(result)["result_count"] == 0


class TestHybridSearchPartialFailure:
    async def test_sql_fails_vector_succeeds(self, make_sql_service, make_rag_service, make_config):
        config = make_config(
            make_sql_service(raise_error=RuntimeError("DB connection lost")),
            make_rag_service(),
        )
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "retrieved_docs" in result
        vector = _parse_vector(result)
        assert vector["success"] is True

    async def test_vector_fails_sql_succeeds(self, make_sql_service, make_rag_service, make_config):
        config = make_config(
            make_sql_service(),
            make_rag_service(raise_error=RuntimeError("pgvector unreachable")),
        )
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "retrieved_docs" in result
        assert len(result["search_results"]) == 1
        vector = _parse_vector(result)
        assert vector["success"] is False

    async def test_both_fail_still_returns_retrieved_docs(
        self, make_sql_service, make_rag_service, make_config
    ):
        config = make_config(
            make_sql_service(raise_error=RuntimeError("SQL down")),
            make_rag_service(raise_error=RuntimeError("Vector down")),
        )
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "retrieved_docs" in result
        vector = _parse_vector(result)
        assert vector["success"] is False

    async def test_error_message_captured_in_vector_result(
        self, make_sql_service, make_rag_service, make_config
    ):
        config = make_config(
            make_sql_service(raise_error=RuntimeError("timeout after 30s")),
            make_rag_service(),
        )
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        # SQL error surfaces in retrieved_docs (no listings found)
        assert "No listings found." in result["retrieved_docs"]


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

    async def test_missing_sql_service_returns_empty(self, make_rag_service, make_config):
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            make_config(sql_service=None, rag_service=make_rag_service()),
        )
        assert result == {}

    async def test_missing_rag_retriever_returns_empty(self, make_sql_service, make_config):
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            make_config(sql_service=make_sql_service(), rag_service=None),
        )
        assert result == {}

    async def test_missing_request_returns_empty(self):
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            {"configurable": {}},
        )
        assert result == {}
