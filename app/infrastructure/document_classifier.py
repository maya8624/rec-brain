from __future__ import annotations

import asyncio
import io
import re
from typing import Literal, Protocol

import structlog
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel

from app.core.config import settings
from app.core.constants import DocumentClassifierConfig, PromptLabels
from app.infrastructure.llm import get_llm
from app.prompts.document_classifier import DOCUMENT_TYPE_CLASSIFICATION_PROMPT

logger = structlog.get_logger(__name__)

DocType = Literal["invoice", "receipt"]


class DocumentTypeClassification(BaseModel):
    doc_type: DocType


class DocumentClassifierProtocol(Protocol):
    async def classify(self, content: bytes, filename: str) -> DocType: ...


class DocumentTypeClassifier:
    def __init__(self) -> None:
        self._client = DocumentIntelligenceClient(
            endpoint=settings.AZURE_DOC_INTEL_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_DOC_INTEL_KEY),
        )

    async def classify(self, content: bytes, filename: str) -> DocType:
        text = await asyncio.to_thread(self._read_text, content)
        result = self._classify_from_text(text.lower())

        if result is not None:
            logger.info("document_classifier_fast_path", filename=filename, doc_type=result)
            return result

        logger.info("document_classifier_llm_fallback", filename=filename, text_length=len(text))
        try:
            prompt = [
                SystemMessage(
                    content=DOCUMENT_TYPE_CLASSIFICATION_PROMPT + PromptLabels.DOCUMENT_TYPE_CLASSIFIER
                ),
                HumanMessage(content=text[: DocumentClassifierConfig.LLM_TEXT_LIMIT]),
            ]
            llm = get_llm().with_structured_output(DocumentTypeClassification)
            classification: DocumentTypeClassification = await llm.ainvoke(prompt)
            return classification.doc_type
        except Exception as exc:
            logger.warning("document_classifier_llm_failed", filename=filename, error=str(exc))
            return "invoice"

    def _read_text(self, content: bytes) -> str:
        poller = self._client.begin_analyze_document(
            "prebuilt-read",
            body=io.BytesIO(content),
            content_type="application/octet-stream",
        )
        return poller.result().content or ""

    @staticmethod
    def _classify_from_text(text: str) -> DocType | None:
        has_receipt = any(
            DocumentTypeClassifier._match_keyword(text, kw)
            for kw in DocumentClassifierConfig.RECEIPT_KEYWORDS
        )
        has_invoice = any(
            DocumentTypeClassifier._match_keyword(text, kw)
            for kw in DocumentClassifierConfig.INVOICE_KEYWORDS
        )

        if has_receipt and not has_invoice:
            return "receipt"
        if has_invoice and not has_receipt:
            return "invoice"
        return None

    @staticmethod
    def _match_keyword(text: str, keyword: str) -> bool:
        trailing = r"\b" if (keyword[-1].isalnum() or keyword[-1] == "_") else ""
        return bool(re.search(rf"\b{re.escape(keyword)}{trailing}", text))
