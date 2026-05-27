import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.types import FilterCondition, VectorStoreQueryMode

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.pgvector_store import PgVectorStoreService

logger = logging.getLogger(__name__)


class RagRetriever:
    """
    Thin retrieval layer over the existing pgvector-backed LlamaIndex store.
    """

    def __init__(
        self,
        vector_store_service: PgVectorStoreService,
        embedding_service: EmbeddingService,
        similarity_top_k: int = 3,
        mmr_threshold: float = 0.7
    ) -> None:
        self._vector_store = vector_store_service.create_vector_store()
        self._embed_model = embedding_service.model
        self._similarity_top_k = similarity_top_k
        self._mmr_threshold = mmr_threshold

        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )

    def _build_retriever(self, filters: MetadataFilters | None = None):
        return self._index.as_retriever(
            similarity_top_k=self._similarity_top_k,
            filters=filters,
            vector_store_query_mode=VectorStoreQueryMode.MMR,
            mmr_threshold=self._mmr_threshold,
        )

    async def aretrieve(
        self,
        query: str,
        doc_type: str | None = None,
        doc_types: frozenset[str] | None = None,
        file_name: str | None = None,
    ) -> list[NodeWithScore]:
        """Retrieve nodes matching the query.

        Args:
            query:     The search query.
            doc_type:  Filter by a single doc_type (AND with file_name if provided).
            doc_types: Filter by multiple doc_types joined with OR. Takes precedence
                       over doc_type when provided. file_name is ignored in this path.
            file_name: Filter by file name (used with doc_type, not doc_types).
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if doc_types:
            filters = MetadataFilters(
                filters=[MetadataFilter(key="doc_type", value=dt) for dt in doc_types],
                condition=FilterCondition.OR,
            )
        else:
            filter_list = []
            if doc_type:
                filter_list.append(MetadataFilter(key="doc_type", value=doc_type))
            if file_name:
                filter_list.append(MetadataFilter(key="file_name", value=file_name))
            filters = MetadataFilters(filters=filter_list) if filter_list else None

        retriever = self._build_retriever(filters=filters)
        return await retriever.aretrieve(query)
