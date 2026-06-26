from __future__ import annotations

import structlog

from app.infrastructure.document_classifier import DocumentClassifierProtocol
from app.infrastructure.invoice_parser import InvoiceParserProtocol
from app.schemas.invoice import InvoiceData

logger = structlog.get_logger(__name__)


class InvoiceExtractionError(Exception):
    pass


class InvoiceExtractionService:
    """
    Orchestrates invoice extraction: classifies document type, delegates to the
    appropriate parser, wraps infrastructure errors into a domain exception so
    the route layer never sees Azure SDK types.
    """

    def __init__(
        self,
        parser: InvoiceParserProtocol,
        receipt_parser: InvoiceParserProtocol,
        classifier: DocumentClassifierProtocol,
    ) -> None:
        self._parser = parser
        self._receipt_parser = receipt_parser
        self._classifier = classifier

    async def extract(
        self,
        content: bytes,
        filename: str,
        property_id: str = "",
    ) -> InvoiceData:
        logger.info("invoice_extract_start", filename=filename, property_id=property_id)
        try:
            doc_type = await self._classifier.classify(content, filename)
            logger.info("invoice_doc_type_resolved", filename=filename, doc_type=doc_type)
            parser = self._receipt_parser if doc_type == "receipt" else self._parser
            data = await parser.parse(content, filename)
        except Exception as exc:
            logger.exception("invoice_extract_failed", filename=filename, error=str(exc))
            raise InvoiceExtractionError(f"Failed to extract invoice data from {filename}") from exc

        logger.info(
            "invoice_extract_complete",
            filename=filename,
            vendor=data.vendor_name,
            total=data.total,
            line_items=len(data.line_items),
            confidence=data.confidence,
        )
        return data
