"""
Unit tests for vector_search_node.
Uses make_rag_retriever and make_config fixtures from tests/conftest.py.
"""
import json

from langchain_core.messages import HumanMessage

from app.agents.nodes.vector import vector_search_node


def _parse_docs(result: dict) -> dict:
    """Extract the JSON payload from retrieved_docs (after the header line)."""
    return json.loads(result["retrieved_docs"].split("]\n", 1)[1])


class TestVectorSearchSuccess:
    async def test_returns_retrieved_docs(self, make_rag_retriever, make_config):
        result = await vector_search_node(
            {"messages": [HumanMessage(content="What are the lease conditions?")]},
            make_config(rag_retriever=make_rag_retriever()),
        )
        assert "retrieved_docs" in result
        assert isinstance(result["retrieved_docs"], str)

    async def test_retrieved_docs_contains_header(self, make_rag_retriever, make_config):
        result = await vector_search_node(
            {"messages": [HumanMessage(content="What are the lease conditions?")]},
            make_config(rag_retriever=make_rag_retriever()),
        )
        assert "[DOCUMENT SEARCH RESULTS" in result["retrieved_docs"]

    async def test_result_structure(self, make_rag_retriever, make_config):
        result = await vector_search_node(
            {"messages": [HumanMessage(content="What are the lease conditions?")]},
            make_config(rag_retriever=make_rag_retriever()),
        )
        content = _parse_docs(result)
        assert content["source"] == "vector_db"
        assert "result_count" in content
        assert "results" in content

    async def test_result_item_shape(self, make_rag_retriever, make_config, make_node):
        nodes = [make_node("Bond is 4 weeks rent.", score=0.91)]
        result = await vector_search_node(
            {"messages": [HumanMessage(content="What is the bond?")]},
            make_config(rag_retriever=make_rag_retriever(nodes=nodes)),
        )
        item = _parse_docs(result)["results"][0]
        assert "text" in item
        assert "score" in item
        assert "metadata" in item

    async def test_text_and_score_values(self, make_rag_retriever, make_config, make_node):
        nodes = [make_node("Bond is 4 weeks rent.", score=0.91)]
        result = await vector_search_node(
            {"messages": [HumanMessage(content="What is the bond?")]},
            make_config(rag_retriever=make_rag_retriever(nodes=nodes)),
        )
        item = _parse_docs(result)["results"][0]
        assert item["text"] == "Bond is 4 weeks rent."
        assert item["score"] == 0.91

    async def test_metadata_passed_through(self, make_rag_retriever, make_config, make_node):
        meta = {"property_id": "prop_42", "doc_type": "strata"}
        nodes = [make_node("Strata levy is $800/quarter.", metadata=meta)]
        result = await vector_search_node(
            {"messages": [HumanMessage(content="strata?")]},
            make_config(rag_retriever=make_rag_retriever(nodes=nodes)),
        )
        item = _parse_docs(result)["results"][0]
        assert item["metadata"]["property_id"] == "prop_42"
        assert item["metadata"]["doc_type"] == "strata"

    async def test_empty_results(self, make_rag_retriever, make_config):
        result = await vector_search_node(
            {"messages": [HumanMessage(content="Very obscure question")]},
            make_config(rag_retriever=make_rag_retriever(nodes=[])),
        )
        content = _parse_docs(result)
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
        rag = make_rag_retriever(
            raise_error=RuntimeError("pgvector connection lost"))
        result = await vector_search_node(
            {"messages": [HumanMessage(content="lease?")]},
            make_config(rag_retriever=rag),
        )
        assert result == {}
