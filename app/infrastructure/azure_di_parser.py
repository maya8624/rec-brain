import asyncio
import io
from pathlib import Path

import structlog
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from llama_index.core import Document

from app.core.config import settings
from app.core.doc_constants import SUPPORTED_EXTENSIONS

logger = structlog.get_logger(__name__)


class AzureDocumentIntelligenceParser:
    """
    Parses document bytes using Azure Document Intelligence (prebuilt-layout).
    Returns LlamaIndex Document objects with markdown content.
    """

    def __init__(self) -> None:
        self._client = DocumentIntelligenceClient(
            endpoint=settings.AZURE_DOC_INTEL_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_DOC_INTEL_KEY),
        )

    async def parse(self, content: bytes, filename: str) -> list[Document]:
        suffix = Path(filename).suffix.lower()

        if suffix in SUPPORTED_EXTENSIONS:
            text = await asyncio.to_thread(self._analyze, content)
        else:
            text = content.decode("utf-8", errors="replace")

        if not text or not text.strip():
            logger.warning("no_content_extracted", filename=filename)
            return []

        return [
            Document(
                text=text,
                metadata={
                    "file_name": filename,
                    "file_type": suffix,
                },
            )
        ]

    def _analyze(self, content: bytes) -> str:
        poller = self._client.begin_analyze_document(
            "prebuilt-layout",
            body=io.BytesIO(content),
            content_type="application/octet-stream",
            output_content_format="markdown",
        )
        result = poller.result()
        return result.content or ""
