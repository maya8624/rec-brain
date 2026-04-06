"""
Unit tests for hybrid_search_node — concurrent SQL + vector search.
Uses make_sql_service, make_rag_retriever, and make_config from conftest.
"""
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes.hybrid import hybrid_search_node
from tests.conftest import parsed

_QUESTION = "Show 2 bedroom apartments in Sydney and explain the lease"


class TestHybridSearchSuccess:
    async def test_returns_system_message(self, make_sql_service, make_rag_retriever, make_config):
        config = make_config(make_sql_service(), make_rag_retriever())
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        )
        assert "messages" in result
        assert isinstance(result["messages"][0], SystemMessage)

    async def test_result_contains_both_sections(self, make_sql_service, make_rag_retriever, make_config):
        config = make_config(make_sql_service(), make_rag_retriever())
        content = parsed(await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        ))
        assert "sql_results" in content
        assert "vector_results" in content

    async def test_sql_results_structure(self, make_sql_service, make_rag_retriever, make_config):
        config = make_config(make_sql_service(), make_rag_retriever())
        sql = parsed(await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        ))["sql_results"]
        assert sql["success"] is True
        assert sql["result_count"] == 1

    async def test_vector_results_structure(self, make_sql_service, make_rag_retriever, make_config):
        config = make_config(make_sql_service(), make_rag_retriever())
        vector = parsed(await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        ))["vector_results"]
        assert vector["success"] is True
        assert vector["result_count"] == 1
        assert len(vector["results"]) == 1

    async def test_vector_result_item_shape(self, make_sql_service, make_rag_retriever, make_config):
        config = make_config(make_sql_service(), make_rag_retriever())
        item = parsed(await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        ))["vector_results"]["results"][0]
        assert "text" in item
        assert "score" in item
        assert "metadata" in item

    async def test_calls_both_services_with_question(
        self, make_sql_service, make_rag_retriever, make_config
    ):
        sql = make_sql_service()
        rag = make_rag_retriever()
        question = "Show apartments near the CBD and their lease terms"
        await hybrid_search_node(
            {"messages": [HumanMessage(content=question)]},
            make_config(sql, rag),
        )
        sql.search_listings.assert_called_once_with(question)
        rag.aretrieve.assert_called_once_with(question)

    async def test_empty_results_from_both_services(
        self, make_sql_service, make_rag_retriever, make_config
    ):
        sql = make_sql_service(
            result={"success": True, "output": [], "result_count": 0, "sql_used": ""})
        rag = make_rag_retriever(nodes=[])
        content = parsed(await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            make_config(sql, rag),
        ))
        assert content["sql_results"]["result_count"] == 0
        assert content["vector_results"]["result_count"] == 0


class TestHybridSearchPartialFailure:
    async def test_sql_fails_vector_succeeds(self, make_sql_service, make_rag_retriever, make_config):
        config = make_config(
            make_sql_service(raise_error=RuntimeError("DB connection lost")),
            make_rag_retriever(),
        )
        content = parsed(await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        ))
        assert content["sql_results"]["success"] is False
        assert "error" in content["sql_results"]
        assert content["vector_results"]["success"] is True

    async def test_vector_fails_sql_succeeds(self, make_sql_service, make_rag_retriever, make_config):
        config = make_config(
            make_sql_service(),
            make_rag_retriever(
                raise_error=RuntimeError("pgvector unreachable")),
        )
        content = parsed(await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        ))
        assert content["sql_results"]["success"] is True
        assert content["vector_results"]["success"] is False

    async def test_both_fail_still_returns_message(
        self, make_sql_service, make_rag_retriever, make_config
    ):
        config = make_config(
            make_sql_service(raise_error=RuntimeError("SQL down")),
            make_rag_retriever(raise_error=RuntimeError("Vector down")),
        )
        content = parsed(await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        ))
        assert content["sql_results"]["success"] is False
        assert content["vector_results"]["success"] is False

    async def test_error_message_captured_in_result(
        self, make_sql_service, make_rag_retriever, make_config
    ):
        config = make_config(
            make_sql_service(raise_error=RuntimeError("timeout after 30s")),
            make_rag_retriever(),
        )
        content = parsed(await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]}, config
        ))
        assert "timeout after 30s" in content["sql_results"]["error"]


class TestHybridSearchGuards:
    async def test_no_human_message_returns_empty(
        self, make_sql_service, make_rag_retriever, make_config
    ):
        sql = make_sql_service()
        rag = make_rag_retriever()
        result = await hybrid_search_node({"messages": []}, make_config(sql, rag))
        assert result == {}
        sql.search_listings.assert_not_called()
        rag.aretrieve.assert_not_called()

    async def test_missing_sql_service_returns_empty(self, make_rag_retriever, make_config):
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            make_config(sql_service=None, rag_retriever=make_rag_retriever()),
        )
        assert result == {}

    async def test_missing_rag_retriever_returns_empty(self, make_sql_service, make_config):
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            make_config(sql_service=make_sql_service(), rag_retriever=None),
        )
        assert result == {}

    async def test_missing_request_returns_empty(self):
        result = await hybrid_search_node(
            {"messages": [HumanMessage(content=_QUESTION)]},
            {"configurable": {}},
        )
        assert result == {}
