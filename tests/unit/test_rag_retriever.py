"""
Unit tests for RagRetriever.
VectorStoreIndex is patched so no real PG connection is needed.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.types import VectorStoreQueryMode

from app.services.rag_service import RagRetriever

pytestmark = pytest.mark.unit


@pytest.fixture
def mock_index():
    """Patches VectorStoreIndex so RagRetriever can be instantiated without a real PG connection."""
    with patch("app.services.rag_service.VectorStoreIndex") as mock_cls:
        index = MagicMock()
        mock_cls.from_vector_store.return_value = index
        yield index


def make_retriever(similarity_top_k: int = 5, similarity_cutoff: float = 0.4) -> RagRetriever:
    return RagRetriever(
        vector_store_service=MagicMock(),
        embedding_service=MagicMock(),
        similarity_top_k=similarity_top_k,
        similarity_cutoff=similarity_cutoff,
    )


def make_node(score: float) -> NodeWithScore:
    node = MagicMock(spec=NodeWithScore)
    node.score = score
    return node


# ── Init / defaults ───────────────────────────────────────────────────────────

class TestRagRetrieverInit:
    def test_default_similarity_cutoff(self, mock_index):
        assert make_retriever()._similarity_cutoff == 0.4

    def test_custom_similarity_cutoff(self, mock_index):
        assert make_retriever(similarity_cutoff=0.6)._similarity_cutoff == 0.6

    def test_default_similarity_top_k(self, mock_index):
        assert make_retriever()._similarity_top_k == 5

    def test_custom_similarity_top_k(self, mock_index):
        assert make_retriever(similarity_top_k=3)._similarity_top_k == 3


# ── _build_retriever ──────────────────────────────────────────────────────────

class TestBuildRetriever:
    def test_default_mode_always_set(self, mock_index):
        make_retriever()._build_retriever()
        kwargs = mock_index.as_retriever.call_args.kwargs
        assert kwargs["vector_store_query_mode"] == VectorStoreQueryMode.DEFAULT

    def test_similarity_top_k_forwarded(self, mock_index):
        make_retriever(similarity_top_k=3)._build_retriever()
        kwargs = mock_index.as_retriever.call_args.kwargs
        assert kwargs["similarity_top_k"] == 3

    def test_filters_none_when_not_provided(self, mock_index):
        make_retriever()._build_retriever()
        kwargs = mock_index.as_retriever.call_args.kwargs
        assert kwargs["filters"] is None

    def test_filters_forwarded_when_provided(self, mock_index):
        filters = MetadataFilters(filters=[MetadataFilter(key="doc_type", value="guide")])
        make_retriever()._build_retriever(filters=filters)
        kwargs = mock_index.as_retriever.call_args.kwargs
        assert kwargs["filters"] is filters


# ── aretrieve ─────────────────────────────────────────────────────────────────

class TestAretrieve:
    async def test_empty_query_raises_value_error(self, mock_index):
        rag = make_retriever()
        with pytest.raises(ValueError, match="empty"):
            await rag.aretrieve("   ")

    async def test_no_doc_type_passes_no_filter(self, mock_index):
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=[]))
        mock_index.as_retriever.return_value = retriever
        await make_retriever().aretrieve("suburb query")
        kwargs = mock_index.as_retriever.call_args.kwargs
        assert kwargs["filters"] is None

    async def test_doc_type_builds_metadata_filter(self, mock_index):
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=[]))
        mock_index.as_retriever.return_value = retriever
        await make_retriever().aretrieve("suburb query", doc_type="guide")
        filters = mock_index.as_retriever.call_args.kwargs["filters"]
        assert filters is not None
        assert filters.filters[0].key == "doc_type"
        assert filters.filters[0].value == "guide"

    async def test_doc_types_builds_or_filter(self, mock_index):
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=[]))
        mock_index.as_retriever.return_value = retriever
        await make_retriever().aretrieve("water query", doc_types=frozenset(["lease", "water_bill"]))
        filters = mock_index.as_retriever.call_args.kwargs["filters"]
        assert filters is not None
        keys = {f.value for f in filters.filters}
        assert keys == {"lease", "water_bill"}

    async def test_retriever_called_with_query(self, mock_index):
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=[]))
        mock_index.as_retriever.return_value = retriever
        await make_retriever().aretrieve("lease conditions")
        retriever.aretrieve.assert_called_once_with("lease conditions")

    async def test_nodes_above_cutoff_returned(self, mock_index):
        nodes = [make_node(0.8), make_node(0.5), make_node(0.4)]
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=nodes))
        mock_index.as_retriever.return_value = retriever
        result = await make_retriever(similarity_cutoff=0.4).aretrieve("query")
        assert result == nodes  # all at or above 0.4

    async def test_nodes_below_cutoff_filtered_out(self, mock_index):
        nodes = [make_node(0.8), make_node(0.3), make_node(0.1)]
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=nodes))
        mock_index.as_retriever.return_value = retriever
        result = await make_retriever(similarity_cutoff=0.4).aretrieve("query")
        assert len(result) == 1
        assert result[0].score == 0.8

    async def test_nodes_with_none_score_filtered_out(self, mock_index):
        nodes = [make_node(0.8), make_node(None)]
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=nodes))
        mock_index.as_retriever.return_value = retriever
        result = await make_retriever(similarity_cutoff=0.4).aretrieve("query")
        assert len(result) == 1
        assert result[0].score == 0.8

    async def test_all_nodes_filtered_returns_empty(self, mock_index):
        nodes = [make_node(0.1), make_node(0.2)]
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=nodes))
        mock_index.as_retriever.return_value = retriever
        result = await make_retriever(similarity_cutoff=0.4).aretrieve("hello")
        assert result == []
