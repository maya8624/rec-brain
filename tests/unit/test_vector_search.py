"""
Unit tests for vector_search_node.
Uses make_rag_retriever and make_config fixtures from tests/conftest.py.
"""
import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes.vector import vector_search_node
from tests.conftest import parsed


# ── Success paths ──────────────────────────────────────────────────────────────

class TestVectorSearchSuccess:
    async def test_returns_system_message(self, make_rag_retriever, make_config):
        result = await vector_search_node(
            {"messages": [HumanMessage(content="What are the lease conditions?")]},
            make_config(rag_retriever=make_rag_retriever()),
        )
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)

    async def test_result_structure(self, make_rag_retriever, make_config):
        result = await vector_search_node(
            {"messages": [HumanMessage(content="What are the lease conditions?")]},
            make_config(rag_retriever=make_rag_retriever()),
        )
        content = parsed(result)
        assert content["source"] == "vector_db"
        assert "result_count" in content
        assert "results" in content

    async def test_result_item_shape(self, make_rag_retriever, make_config, make_node):
        nodes = [make_node("Bond is 4 weeks rent.", score=0.91)]
        result = await vector_search_node(
            {"messages": [HumanMessage(content="What is the bond?")]},
            make_config(rag_retriever=make_rag_retriever(nodes=nodes)),
        )
        item = parsed(result)["results"][0]
        assert "text" in item
        assert "score" in item
        assert "metadata" in item

    async def test_text_and_score_values(self, make_rag_retriever, make_config, make_node):
        nodes = [make_node("Bond is 4 weeks rent.", score=0.91)]
        result = await vector_search_node(
            {"messages": [HumanMessage(content="What is the bond?")]},
            make_config(rag_retriever=make_rag_retriever(nodes=nodes)),
        )
        item = parsed(result)["results"][0]
        assert item["text"] == "Bond is 4 weeks rent."
        assert item["score"] == 0.91

    async def test_metadata_passed_through(self, make_rag_retriever, make_config, make_node):
        meta = {"property_id": "prop_42", "doc_type": "strata"}
        nodes = [make_node("Strata levy is $800/quarter.", metadata=meta)]
        result = await vector_search_node(
            {"messages": [HumanMessage(content="strata?")]},
            make_config(rag_retriever=make_rag_retriever(nodes=nodes)),
        )
        item = parsed(result)["results"][0]
        assert item["metadata"]["property_id"] == "prop_42"
        assert item["metadata"]["doc_type"] == "strata"

    async def test_empty_results(self, make_rag_retriever, make_config):
        result = await vector_search_node(
            {"messages": [HumanMessage(content="Very obscure question")]},
            make_config(rag_retriever=make_rag_retriever(nodes=[])),
        )
        content = parsed(result)
        assert content["result_count"] == 0
        assert content["results"] == []

    async def test_calls_retrieve_with_exact_question(self, make_rag_retriever, make_config):
        rag = make_rag_retriever()
        question = "Explain the strata by-laws"
        await vector_search_node(
            {"messages": [HumanMessage(content=question)]},
            make_config(rag_retriever=rag),
        )
        rag.aretrieve.assert_called_once_with(question)


# ── Guard paths ────────────────────────────────────────────────────────────────

class TestVectorSearchGuards:
    async def test_no_human_message_returns_empty(self, make_rag_retriever, make_config):
        rag = make_rag_retriever()
        result = await vector_search_node(
            {"messages": []},
            make_config(rag_retriever=rag),
        )
        assert result == {}
        rag.aretrieve.assert_not_called()

    async def test_missing_service_returns_empty(self, make_config):
        result = await vector_search_node(
            {"messages": [HumanMessage(content="lease?")]},
            make_config(rag_retriever=None),
        )
        assert result == {}

    async def test_missing_request_returns_empty(self):
        result = await vector_search_node(
            {"messages": [HumanMessage(content="lease?")]},
            {"configurable": {}},
        )
        assert result == {}

    async def test_retrieve_exception_returns_empty(self, make_rag_retriever, make_config):
        rag = make_rag_retriever(raise_error=RuntimeError("pgvector connection lost"))
        result = await vector_search_node(
            {"messages": [HumanMessage(content="lease?")]},
            make_config(rag_retriever=rag),
        )
        assert result == {}
