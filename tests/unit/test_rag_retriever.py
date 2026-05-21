"""
Unit tests for RagRetriever.
VectorStoreIndex is patched so no real PG connection is needed.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

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


def make_retriever(similarity_top_k: int = 3, mmr_threshold: float = 0.7) -> RagRetriever:
    return RagRetriever(
        vector_store_service=MagicMock(),
        embedding_service=MagicMock(),
        similarity_top_k=similarity_top_k,
        mmr_threshold=mmr_threshold,
    )


# ── Init / defaults ───────────────────────────────────────────────────────────

class TestRagRetrieverInit:
    def test_default_mmr_threshold(self, mock_index):
        assert make_retriever()._mmr_threshold == 0.7

    def test_custom_mmr_threshold(self, mock_index):
        assert make_retriever(mmr_threshold=0.5)._mmr_threshold == 0.5

    def test_default_similarity_top_k(self, mock_index):
        assert make_retriever()._similarity_top_k == 3

    def test_custom_similarity_top_k(self, mock_index):
        assert make_retriever(similarity_top_k=5)._similarity_top_k == 5


# ── _build_retriever ──────────────────────────────────────────────────────────

class TestBuildRetriever:
    def test_mmr_mode_always_set(self, mock_index):
        make_retriever()._build_retriever()
        kwargs = mock_index.as_retriever.call_args.kwargs
        assert kwargs["vector_store_query_mode"] == VectorStoreQueryMode.MMR

    def test_mmr_threshold_forwarded(self, mock_index):
        make_retriever(mmr_threshold=0.55)._build_retriever()
        kwargs = mock_index.as_retriever.call_args.kwargs
        assert kwargs["mmr_threshold"] == 0.55

    def test_similarity_top_k_forwarded(self, mock_index):
        make_retriever(similarity_top_k=5)._build_retriever()
        kwargs = mock_index.as_retriever.call_args.kwargs
        assert kwargs["similarity_top_k"] == 5

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

    async def test_returns_retriever_results(self, mock_index):
        node = MagicMock()
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=[node]))
        mock_index.as_retriever.return_value = retriever
        result = await make_retriever().aretrieve("suburb query")
        assert result == [node]

    async def test_retriever_called_with_query(self, mock_index):
        retriever = AsyncMock(aretrieve=AsyncMock(return_value=[]))
        mock_index.as_retriever.return_value = retriever
        await make_retriever().aretrieve("lease conditions")
        retriever.aretrieve.assert_called_once_with("lease conditions")
