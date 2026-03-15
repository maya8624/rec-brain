# """
# app/services/rag_service.py

# Document retrieval service using LlamaIndex + ChromaDB.

# Why LlamaIndex over plain LangChain:
#     - Hierarchical chunking keeps lease clauses intact
#     - QueryFusionRetriever generates query variations for better recall
#     - Hybrid search (vector + keyword) for legal/contract terminology
#     - Source citation comes back structured — page numbers, filenames, scores
#     - Built-in evaluation pipeline (scripts/eval_rag.py)

# Vector store: Chroma now, pgvector later.
# Migration: change VECTOR_STORE=pgvector in .env — no code changes needed.
# """
# from __future__ import annotations

# import logging
# from typing import Optional

# import structlog
# from llama_index.core import VectorStoreIndex, StorageContext, Settings as LlamaSettings
# from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes
# from llama_index.core.retrievers import VectorIndexRetriever, QueryFusionRetriever
# from llama_index.core.query_engine import RetrieverQueryEngine
# from llama_index.core.postprocessor import SimilarityPostprocessor

# from app.core.config import settings

# logger = structlog.get_logger(__name__)

# # Singletons
# _vector_store = None
# _index: Optional[VectorStoreIndex] = None
# _embedding_model = None


# # ── Embedding model ────────────────────────────────────────────────────────────

# def get_embedding_model():
#     """
#     Lazy singleton embedding model.
#     OpenAI text-embedding-3-small if OPENAI_API_KEY is set (better quality).
#     Falls back to HuggingFace EMBEDDING_MODEL (free, local).
#     """
#     global _embedding_model
#     if _embedding_model is None:
#         if settings.OPENAI_API_KEY:
#             from llama_index.embeddings.openai import OpenAIEmbedding
#             logger.info("embedding_model", type="openai",
#                         model="text-embedding-3-small")
#             _embedding_model = OpenAIEmbedding(
#                 model="text-embedding-3-small",
#                 api_key=settings.OPENAI_API_KEY,
#                 embed_batch_size=100,
#                 dimensions=1536,
#             )
#         else:
#             from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#             logger.info("embedding_model", type="huggingface",
#                         model=settings.EMBEDDING_MODEL)
#             _embedding_model = HuggingFaceEmbedding(
#                 model_name=settings.EMBEDDING_MODEL,
#                 embed_batch_size=32,
#             )

#         # Register globally so LlamaIndex uses it everywhere
#         LlamaSettings.embed_model = _embedding_model

#     return _embedding_model


# # ── Vector store ───────────────────────────────────────────────────────────────

# def get_vector_store():
#     """
#     Returns the configured vector store singleton.
#     VECTOR_STORE=chroma (default) or VECTOR_STORE=pgvector in .env.
#     """
#     global _vector_store
#     if _vector_store is None:
#         if settings.VECTOR_STORE == "pgvector":
#             _vector_store = _build_pgvector_store()
#         else:
#             _vector_store = _build_chroma_store()
#     return _vector_store


# def _build_chroma_store():
#     """ChromaDB — local file, zero infrastructure, works out of the box."""
#     import chromadb
#     from llama_index.vector_stores.chroma import ChromaVectorStore

#     logger.info("vector_store_init", backend="chroma",
#                 path=settings.CHROMA_PATH)

#     client = chromadb.PersistentClient(path=settings.CHROMA_PATH)
#     collection = client.get_or_create_collection("real-estate-brain")

#     return ChromaVectorStore(chroma_collection=collection)


# def _build_pgvector_store():
#     """pgvector — activated by VECTOR_STORE=pgvector in .env."""
#     from llama_index.vector_stores.postgres import PGVectorStore

#     logger.info(
#         "vector_store_init",
#         backend="pgvector",
#         table=settings.PGVECTOR_TABLE,
#         dim=settings.PGVECTOR_EMBED_DIM,
#     )

#     return PGVectorStore.from_params(
#         database=settings.DB_NAME,
#         host=settings.DB_HOST,
#         password=settings.DB_PASSWORD,
#         port=int(settings.DB_PORT),
#         user=settings.DB_USER,
#         table_name=settings.PGVECTOR_TABLE,
#         embed_dim=settings.PGVECTOR_EMBED_DIM,
#         hybrid_search=True,
#         text_search_config="english",
#     )


# # ── Index ──────────────────────────────────────────────────────────────────────

# def get_index() -> VectorStoreIndex:
#     """Lazy singleton index over the vector store."""
#     global _index
#     if _index is None:
#         # Ensure embedding model is registered before building index
#         get_embedding_model()

#         storage_context = StorageContext.from_defaults(
#             vector_store=get_vector_store()
#         )
#         _index = VectorStoreIndex.from_vector_store(
#             vector_store=get_vector_store(),
#             storage_context=storage_context,
#         )
#         logger.info("vector_index_ready", backend=settings.VECTOR_STORE)

#     return _index


# # ── Public interface ───────────────────────────────────────────────────────────

# def perform_vector_search(
#     query: str,
#     property_id: Optional[str] = None,
#     top_k: int = None,
# ) -> dict:
#     """
#     Search property documents and return a synthesized answer with sources.

#     Args:
#         query:       Natural language question
#         property_id: Optional — filter to one property's documents
#         top_k:       Number of chunks to retrieve (defaults to TOP_K_RESULTS)

#     Returns:
#         {success, answer, sources}
#         sources: [{document, doc_type, property_id, page, relevance_score}]
#     """
#     top_k = top_k or settings.TOP_K_RESULTS

#     logger.info("vector_search",
#                 query=query[:60], property_id=property_id, top_k=top_k)

#     try:
#         query_engine = _build_query_engine(
#             property_id=property_id, top_k=top_k)
#         response = query_engine.query(query)
#         sources = _extract_sources(response)

#         logger.info("vector_search_complete", sources=len(sources))

#         return {
#             "success": True,
#             "answer": str(response),
#             "sources": sources,
#         }

#     except Exception as e:
#         logger.exception("vector_search_failed",
#                          query=query[:60], error=str(e))
#         return {
#             "success": False,
#             "answer": None,
#             "sources": [],
#             "error": "Document search is temporarily unavailable.",
#         }


# def ingest_document(file_path: str, metadata: dict) -> int:
#     """
#     Ingest a document into the vector store.
#     Called by DocumentLoaderService — not during request handling.

#     Returns number of chunks stored.
#     """
#     from llama_index.core import SimpleDirectoryReader

#     get_embedding_model()  # ensure registered before ingestion

#     logger.info("ingest_start", file=file_path)

#     documents = SimpleDirectoryReader(input_files=[file_path]).load_data()

#     for doc in documents:
#         doc.metadata.update(metadata)

#     # Hierarchical chunking — large chunks preserve context,
#     # small chunks enable precise retrieval of specific clauses
#     parser = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])
#     nodes = parser.get_nodes_from_documents(documents)
#     leaf_nodes = get_leaf_nodes(nodes)

#     index = get_index()
#     index.insert_nodes(nodes)

#     logger.info("ingest_complete", nodes=len(nodes),
#                 leaf=len(leaf_nodes), file=file_path)

#     return len(leaf_nodes)


# # ── Private helpers ────────────────────────────────────────────────────────────

# def _build_query_engine(
#     property_id: Optional[str],
#     top_k: int,
# ) -> RetrieverQueryEngine:
#     """Build retriever with optional property filter and query fusion."""
#     index = get_index()

#     # Metadata filter — pgvector only (Chroma uses different filtering)
#     filters = None
#     if property_id and settings.VECTOR_STORE == "pgvector":
#         from llama_index.core.vector_stores.types import MetadataFilters, ExactMatchFilter
#         filters = MetadataFilters(
#             filters=[ExactMatchFilter(key="property_id", value=property_id)]
#         )

#     # Base vector retriever
#     vector_retriever = VectorIndexRetriever(
#         index=index,
#         similarity_top_k=top_k,
#         filters=filters,
#     )

#     # QueryFusionRetriever — generates query variations to improve recall
#     # eg "break lease fee" also searches "early termination penalty"
#     retriever = QueryFusionRetriever(
#         retrievers=[vector_retriever],
#         similarity_top_k=top_k,
#         num_queries=3,
#         mode="reciprocal_rerank",
#         use_async=False,
#     )

#     # Filter out low-relevance chunks
#     postprocessor = SimilarityPostprocessor(
#         similarity_cutoff=settings.SIMILARITY_THRESHOLD
#     )

#     return RetrieverQueryEngine.from_args(
#         retriever=retriever,
#         node_postprocessors=[postprocessor],
#         response_mode="compact",
#     )


# def _extract_sources(response) -> list[dict]:
#     """Extract and deduplicate source document metadata from LlamaIndex response."""
#     sources = []
#     seen = set()

#     for node in response.source_nodes:
#         filename = node.metadata.get("filename", "Unknown document")
#         if filename in seen:
#             continue
#         seen.add(filename)
#         sources.append({
#             "document": filename,
#             "doc_type": node.metadata.get("doc_type", ""),
#             "property_id": node.metadata.get("property_id", ""),
#             "page": str(node.metadata.get("page_label", "")),
#             "relevance_score": round(node.score or 0.0, 3),
#         })

#     return sorted(sources, key=lambda x: x["relevance_score"], reverse=True)


# class RagServiceError(Exception):
#     """Raised when RAG retrieval fails unrecoverably."""
#     pass
