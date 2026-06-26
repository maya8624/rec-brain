from __future__ import annotations

import asyncio
import io
from typing import Protocol

import structlog
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import DocumentField
from azure.core.credentials import AzureKeyCredential

from app.core.config import settings
from app.schemas.invoice import InvoiceData, LineItem

logger = structlog.get_logger(__name__)


# ------------------------------------
# Protocol — the only thing the service layer knows about
# ------------------------------------

class InvoiceParserProtocol(Protocol):
    async def parse(self, content: bytes, filename: str) -> InvoiceData: ...


# ------------------------------------
# Azure DI field extractors (private)
# All guard against absent fields and None values at every level.
# ------------------------------------

def _str(fields: dict[str, DocumentField], key: str) -> str | None:
    f = fields.get(key)
    return f.value_string if f else None


def _address(fields: dict[str, DocumentField], key: str) -> str | None:
    """Address fields expose raw text via .content rather than a scalar value."""
    f = fields.get(key)
    return f.content if f else None


def _date(fields: dict[str, DocumentField], key: str):
    f = fields.get(key)
    return f.value_date if f else None


def _amount(fields: dict[str, DocumentField], key: str) -> float | None:
    f = fields.get(key)
    if not f or not f.value_currency:
        return None
    return f.value_currency.amount


def _currency_symbol(fields: dict[str, DocumentField], key: str) -> str | None:
    f = fields.get(key)
    if not f or not f.value_currency:
        return None
    return f.value_currency.currency_code


def _line_items(fields: dict[str, DocumentField]) -> list[LineItem]:
    items_field = fields.get("Items")
    if not items_field or not items_field.value_array:
        return []

    rows: list[LineItem] = []
    for item in items_field.value_array:
        obj = item.value_object or {}
        desc_f = obj.get("Description")
        qty_f = obj.get("Quantity")
        unit_f = obj.get("UnitPrice")
        amt_f = obj.get("Amount")
        rows.append(
            LineItem(
                description=desc_f.value_string if desc_f else None,
                quantity=qty_f.value_number if qty_f else None,
                unit_price=unit_f.value_currency.amount if (unit_f and unit_f.value_currency) else None,
                amount=amt_f.value_currency.amount if (amt_f and amt_f.value_currency) else None,
            )
        )
    return rows


def _receipt_line_items(fields: dict[str, DocumentField]) -> list[LineItem]:
    items_field = fields.get("Items")
    if not items_field or not items_field.value_array:
        return []

    rows: list[LineItem] = []
    for item in items_field.value_array:
        obj = item.value_object or {}
        desc_f  = obj.get("Description")
        qty_f   = obj.get("Quantity")
        price_f = obj.get("Price")       # receipt uses "Price" not "UnitPrice"
        total_f = obj.get("TotalPrice")  # receipt uses "TotalPrice" not "Amount"
        rows.append(
            LineItem(
                description=desc_f.value_string if desc_f else None,
                quantity=qty_f.value_number if qty_f else None,
                unit_price=price_f.value_currency.amount if (price_f and price_f.value_currency) else None,
                amount=total_f.value_currency.amount if (total_f and total_f.value_currency) else None,
            )
        )
    return rows


# ------------------------------------
# Concrete Azure DI implementation
# ------------------------------------

class AzureInvoiceParser:
    def __init__(self) -> None:
        self._client = DocumentIntelligenceClient(
            endpoint=settings.AZURE_DOC_INTEL_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_DOC_INTEL_KEY),
        )

    async def parse(self, content: bytes, filename: str) -> InvoiceData:
        return await asyncio.to_thread(self._extract, content, filename)

    def _extract(self, content: bytes, filename: str) -> InvoiceData:
        poller = self._client.begin_analyze_document(
            "prebuilt-invoice",
            body=io.BytesIO(content),
            content_type="application/octet-stream",
        )
        result = poller.result()

        if not result.documents:
            logger.warning("invoice_no_documents_detected", filename=filename)
            return InvoiceData(doc_type="invoice")

        doc = result.documents[0]
        fields = doc.fields or {}
        confidence = doc.confidence or 0.0

        if confidence < 0.8:
            logger.warning(
                "invoice_low_confidence",
                filename=filename,
                confidence=round(confidence, 3),
            )

        return InvoiceData(
            doc_type="invoice",
            vendor_name=_str(fields, "VendorName"),
            vendor_address=_address(fields, "VendorAddress"),
            customer_name=_str(fields, "CustomerName"),
            invoice_id=_str(fields, "InvoiceId"),
            invoice_date=_date(fields, "InvoiceDate"),
            due_date=_date(fields, "DueDate"),
            subtotal=_amount(fields, "SubTotal"),
            tax=_amount(fields, "TotalTax"),
            total=_amount(fields, "InvoiceTotal"),
            currency=_currency_symbol(fields, "InvoiceTotal"),
            line_items=_line_items(fields),
            confidence=confidence,
        )


# ------------------------------------
# Receipt parser (prebuilt-receipt)
# ------------------------------------

class AzureReceiptParser:
    def __init__(self) -> None:
        self._client = DocumentIntelligenceClient(
            endpoint=settings.AZURE_DOC_INTEL_ENDPOINT,
            credential=AzureKeyCredential(settings.AZURE_DOC_INTEL_KEY),
        )

    async def parse(self, content: bytes, filename: str) -> InvoiceData:
        return await asyncio.to_thread(self._extract, content, filename)

    def _extract(self, content: bytes, filename: str) -> InvoiceData:
        poller = self._client.begin_analyze_document(
            "prebuilt-receipt",
            body=io.BytesIO(content),
            content_type="application/octet-stream",
        )
        result = poller.result()

        if not result.documents:
            logger.warning("receipt_no_documents_detected", filename=filename)
            return InvoiceData(doc_type="receipt")

        doc = result.documents[0]
        fields = doc.fields or {}
        confidence = doc.confidence or 0.0

        if confidence < 0.8:
            logger.warning(
                "receipt_low_confidence",
                filename=filename,
                confidence=round(confidence, 3),
            )

        return InvoiceData(
            doc_type="receipt",
            vendor_name=_str(fields, "MerchantName"),
            vendor_address=_address(fields, "MerchantAddress"),
            customer_name=None,
            invoice_id=None,
            invoice_date=_date(fields, "TransactionDate"),
            due_date=None,
            subtotal=_amount(fields, "Subtotal"),
            tax=_amount(fields, "Tax"),
            total=_amount(fields, "Total"),
            currency=_currency_symbol(fields, "Total"),
            line_items=_receipt_line_items(fields),
            confidence=confidence,
        )
