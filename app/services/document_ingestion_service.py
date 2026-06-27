import uuid

import structlog
from langchain_openai import ChatOpenAI
from llama_index.core.ingestion import DocstoreStrategy, IngestionPipeline
from llama_index.core.node_parser import MarkdownNodeParser, SentenceSplitter

from app.core.config import settings
from app.core.doc_constants import (
    ALLOWED_DOC_TYPES,
    CLASSIFICATION_RULES,
    DOC_TYPE_GENERIC,
)
from app.infrastructure.azure_di_parser import AzureDocumentIntelligenceParser
from app.infrastructure.embedding import EmbeddingService
from app.infrastructure.pgvector_store import PgVectorStoreService

logger = structlog.get_logger(__name__)

_MAX_CLASSIFICATION_CHARS = 2000


class DocumentIngestionError(Exception):
    pass


def _classify_by_rules(text: str) -> str | None:
    text_lower = text.lower()
    for doc_type, keywords, match_all in CLASSIFICATION_RULES:
        check = all if match_all else any
        if check(kw in text_lower for kw in keywords):
            return doc_type
    return None


async def _classify_with_llm(llm: ChatOpenAI, text: str, filename: str) -> str:
    filename_hint = f"\nFilename: {filename}" if filename else ""
    prompt = (
        f"Classify this real estate document into one of: {', '.join(sorted(ALLOWED_DOC_TYPES))}\n"
        f"{filename_hint}\nDocument excerpt:\n{text[:_MAX_CLASSIFICATION_CHARS]}\n\n"
        "Respond with ONLY the type name."
    )
    try:
        response = await llm.ainvoke(prompt)
        result = response.content.strip().lower()
        return result if result in ALLOWED_DOC_TYPES else DOC_TYPE_GENERIC
    except Exception:
        logger.warning("llm_classification_failed", filename=filename)
        return DOC_TYPE_GENERIC


class DocumentIngestionService:
    """
    Orchestrates: Azure DI parse → classify → chunk → embed → pgvector upsert.
    File bytes are passed in directly — caller (Azure Function → endpoint) owns fetching.
    """

    def __init__(
        self,
        di_parser: AzureDocumentIntelligenceParser,
        embedding_service: EmbeddingService,
        vector_store_service: PgVectorStoreService,
    ) -> None:
        self._di_parser = di_parser
        self._llm = ChatOpenAI(
            model=settings.OPENAI_MODEL_NAME,
            api_key=settings.OPENAI_API_KEY,
            temperature=0,
        )
        self._pipeline = IngestionPipeline(
            transformations=[
                MarkdownNodeParser(),
                SentenceSplitter(chunk_size=512, chunk_overlap=50),
                embedding_service.model,
            ],
            vector_store=vector_store_service.create_vector_store(),
            docstore_strategy=DocstoreStrategy.UPSERTS,
        )

    async def ingest(
        self,
        content: bytes,
        filename: str,
        property_id: str = "",
        doc_type: str = "",
    ) -> dict:
        documents = await self._di_parser.parse(content, filename)
        if not documents:
            raise DocumentIngestionError(f"No content extracted from {filename}")

        doc_text = documents[0].get_content()
        if doc_type.strip() in ALLOWED_DOC_TYPES:
            resolved_type = doc_type.strip()
        else:
            resolved_type = _classify_by_rules(doc_text) or await _classify_with_llm(
                self._llm, doc_text, filename
            )

        for doc in documents:
            doc.metadata.update({
                "property_id": property_id,
                "doc_type": resolved_type,
                "file_name": filename,
                "doc_id": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{property_id}:{filename}")),
            })

        nodes = await self._pipeline.arun(documents=documents)

        logger.info(
            "ingest_complete",
            filename=filename,
            property_id=property_id,
            doc_type=resolved_type,
            chunks=len(nodes),
        )

        return {"doc_type": resolved_type, "chunk_count": len(nodes)}
