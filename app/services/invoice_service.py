from __future__ import annotations

import structlog

from app.infrastructure.invoice_parser import InvoiceParserProtocol
from app.schemas.invoice import InvoiceData

logger = structlog.get_logger(__name__)


class InvoiceExtractionError(Exception):
    pass


class InvoiceExtractionService:
    """
    Orchestrates invoice extraction: delegates to the parser, wraps infrastructure
    errors into a domain exception so the route layer never sees Azure SDK types.
    """

    def __init__(self, parser: InvoiceParserProtocol) -> None:
        self._parser = parser

    async def extract(
        self,
        content: bytes,
        filename: str,
        property_id: str = "",
    ) -> InvoiceData:
        logger.info("invoice_extract_start", filename=filename, property_id=property_id)
        try:
            data = await self._parser.parse(content, filename)
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
