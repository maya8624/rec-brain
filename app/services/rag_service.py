import logging

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.vector_stores.types import FilterCondition, VectorStoreQueryMode

from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.pgvector_store import PgVectorStoreService

logger = logging.getLogger(__name__)

_GLOBAL_DOC_TYPES = frozenset(["legislation", "policy", "guide", "faq"])


class RagRetriever:
    """
    Thin retrieval layer over the existing pgvector-backed LlamaIndex store.
    """

    def __init__(
        self,
        vector_store_service: PgVectorStoreService,
        embedding_service: EmbeddingService,
        similarity_top_k: int = 5,
        similarity_cutoff: float = 0.5,
    ) -> None:
        self._vector_store = vector_store_service.create_vector_store()
        self._embed_model = embedding_service.model
        self._similarity_top_k = similarity_top_k
        self._similarity_cutoff = similarity_cutoff

        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self._embed_model,
        )

    def _build_retriever(self, filters: MetadataFilters | None = None):
        return self._index.as_retriever(
            similarity_top_k=self._similarity_top_k,
            filters=filters,
            vector_store_query_mode=VectorStoreQueryMode.DEFAULT,
        )

    async def aretrieve(
        self,
        query: str,
        doc_type: str | None = None,
        doc_types: frozenset[str] | None = None,
        file_name: str | None = None,
        property_id: str | None = None,
    ) -> list[NodeWithScore]:
        """Retrieve nodes matching the query.

        Args:
            query:       The search query.
            doc_type:    Filter by a single doc_type (AND with file_name if provided).
            doc_types:   Filter by multiple doc_types joined with OR. Takes precedence
                         over doc_type when provided. file_name is ignored in this path.
            file_name:   Filter by file name (used with doc_type, not doc_types).
            property_id: When provided with doc_types, builds a compound filter:
                         (property_id = X AND doc_type IN [property types])
                         OR (doc_type = "legislation")
        """
        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if doc_types:
            filters = self._build_doc_types_filter(doc_types, property_id)
        else:
            filter_list = []
            if doc_type:
                filter_list.append(MetadataFilter(key="doc_type", value=doc_type))
            if file_name:
                filter_list.append(MetadataFilter(key="file_name", value=file_name))
            filters = MetadataFilters(filters=filter_list) if filter_list else None

        retriever = self._build_retriever(filters=filters)
        nodes = await retriever.aretrieve(query)
        filtered = [node for node in nodes if node.score is not None and node.score >= self._similarity_cutoff]
        return filtered

    def _build_doc_types_filter(
        self,
        doc_types: frozenset[str],
        property_id: str | None,
    ) -> MetadataFilters:
        """
        Builds a compound filter when property_id is provided:
          (doc_type IN [property_types] AND property_id = X)
          OR
          (doc_type IN [global_types])   ← legislation, policy, guide, faq — never property-scoped

        Falls back to a flat OR filter when no property_id is given.
        """
        if not property_id:
            return MetadataFilters(
                filters=[MetadataFilter(key="doc_type", value=dt) for dt in doc_types],
                condition=FilterCondition.OR,
            )

        property_types = doc_types - _GLOBAL_DOC_TYPES
        global_types = doc_types & _GLOBAL_DOC_TYPES

        branches = []

        if property_types:
            branches.append(
                MetadataFilters(
                    filters=[
                        MetadataFilter(key="property_id", value=property_id),
                        MetadataFilters(
                            filters=[MetadataFilter(key="doc_type", value=dt) for dt in property_types],
                            condition=FilterCondition.OR,
                        ),
                    ],
                    condition=FilterCondition.AND,
                )
            )

        if global_types:
            branches.append(
                MetadataFilters(
                    filters=[MetadataFilter(key="doc_type", value=dt) for dt in global_types],
                    condition=FilterCondition.OR,
                )
            )

        return MetadataFilters(filters=branches, condition=FilterCondition.OR)
