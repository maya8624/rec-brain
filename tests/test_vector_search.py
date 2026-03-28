"""
Tests for vector_search_node — RagRetriever via pgvector.
No real DB or embedding model required — services are mocked.

Usage:
    pytest tests/test_vector_search.py
    pytest tests/test_vector_search.py -v
    pytest tests/test_vector_search.py -k empty
"""
import json
from unittest.mock import AsyncMock, MagicMock

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes.vector import vector_search_node


# ── Factories ──────────────────────────────────────────────────────────────────

def make_node(text: str, score: float = 0.85, metadata: dict | None = None):
    """Build a mock NodeWithScore — avoids importing LlamaIndex schema."""
    node = MagicMock()
    node.get_content.return_value = text
    node.metadata = metadata or {}
    return MagicMock(node=node, score=score)


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
            make_node("A lease agreement requires 4 weeks bond.", score=0.92,
                      metadata={"property_id": "prop_1", "doc_type": "lease"}),
            make_node("Pet policy: small pets allowed with written approval.", score=0.85,
                      metadata={"property_id": "prop_1", "doc_type": "lease"}),
        ]
    return mock


def make_config(rag_retriever=None):
    """Build a RunnableConfig with rag_retriever on app.state."""
    request = MagicMock()
    request.app.state.rag_retriever = rag_retriever
    return {"configurable": {"request": request}}


def make_state(question: str = "What are the lease conditions?") -> dict:
    return {"messages": [HumanMessage(content=question)]}


def parsed(result: dict) -> dict:
    """Parse the SystemMessage content from a node result."""
    return json.loads(result["messages"][0].content)


# ── Success paths ──────────────────────────────────────────────────────────────

class TestVectorSearchSuccess:
    async def test_returns_system_message(self):
        """Node returns a SystemMessage when retrieval succeeds."""
        result = await vector_search_node(
            make_state(), make_config(make_rag_retriever())
        )

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], SystemMessage)

    async def test_result_structure(self):
        """SystemMessage contains result_count, source, and results list."""
        result = await vector_search_node(
            make_state(), make_config(make_rag_retriever())
        )
        content = parsed(result)

        assert content["result_count"] == 2
        assert content["source"] == "vector_db"
        assert len(content["results"]) == 2

    async def test_result_item_shape(self):
        """Each result item has text, score, and metadata keys."""
        result = await vector_search_node(
            make_state(), make_config(make_rag_retriever())
        )
        item = parsed(result)["results"][0]

        assert "text" in item
        assert "score" in item
        assert "metadata" in item

    async def test_text_and_score_values(self):
        """Text and score match the mock node values."""
        nodes = [make_node("Bond is 4 weeks rent.", score=0.91)]
        result = await vector_search_node(
            make_state("What is the bond?"), make_config(make_rag_retriever(nodes))
        )
        item = parsed(result)["results"][0]

        assert item["text"] == "Bond is 4 weeks rent."
        assert item["score"] == 0.91

    async def test_metadata_passed_through(self):
        """Node metadata is preserved in the result."""
        meta = {"property_id": "prop_42", "doc_type": "strata"}
        nodes = [make_node("Strata levy is $800/quarter.", metadata=meta)]
        result = await vector_search_node(
            make_state(), make_config(make_rag_retriever(nodes))
        )
        item = parsed(result)["results"][0]

        assert item["metadata"]["property_id"] == "prop_42"
        assert item["metadata"]["doc_type"] == "strata"

    async def test_empty_results(self):
        """Returns SystemMessage with empty list when no nodes match."""
        result = await vector_search_node(
            make_state("Very obscure question"), make_config(make_rag_retriever(nodes=[]))
        )
        content = parsed(result)

        assert content["result_count"] == 0
        assert content["results"] == []

    async def test_calls_retrieve_with_exact_question(self):
        """aretrieve is called once with the exact human message text."""
        rag = make_rag_retriever()
        question = "Explain the strata by-laws"

        await vector_search_node(make_state(question), make_config(rag))

        rag.aretrieve.assert_called_once_with(question)


# ── Guard paths ────────────────────────────────────────────────────────────────

class TestVectorSearchGuards:
    async def test_no_human_message_returns_empty(self):
        """Returns {} without calling aretrieve when state has no HumanMessage."""
        rag = make_rag_retriever()
        state = {"messages": []}

        result = await vector_search_node(state, make_config(rag))

        assert result == {}
        rag.aretrieve.assert_not_called()

    async def test_missing_service_returns_empty(self):
        """Returns {} when rag_retriever is None on app.state."""
        result = await vector_search_node(make_state(), make_config(rag_retriever=None))

        assert result == {}

    async def test_missing_request_returns_empty(self):
        """Returns {} when request is absent from configurable."""
        result = await vector_search_node(make_state(), {"configurable": {}})

        assert result == {}

    async def test_retrieve_exception_returns_empty(self):
        """Returns {} when aretrieve raises an unexpected exception."""
        rag = make_rag_retriever(raise_error=RuntimeError("pgvector connection lost"))

        result = await vector_search_node(make_state(), make_config(rag))

        assert result == {}
