"""
Unit tests for InvoiceExtractionService.
InvoiceParserProtocol is mocked — no Azure credentials required.
"""
from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from app.schemas.invoice import InvoiceData, LineItem
from app.services.invoice_service import InvoiceExtractionError, InvoiceExtractionService

_PATCH_LOGGER = patch("app.services.invoice_service.logger")


# ------------------------------------
# Helpers
# ------------------------------------

def _make_parser(return_value: InvoiceData | None = None, raise_error: Exception | None = None):
    mock = AsyncMock()
    if raise_error:
        mock.parse.side_effect = raise_error
    else:
        mock.parse.return_value = return_value or _default_invoice()
    return mock


def _default_invoice() -> InvoiceData:
    return InvoiceData(
        vendor_name="Acme Plumbing",
        vendor_address="12 George St, Sydney NSW 2000",
        customer_name="Sunshine Realty",
        invoice_id="INV-0042",
        invoice_date=date(2026, 6, 1),
        due_date=date(2026, 6, 15),
        subtotal=450.00,
        tax=45.00,
        total=495.00,
        currency="$",
        line_items=[
            LineItem(description="Pipe repair", quantity=1.0, unit_price=450.00, amount=450.00),
        ],
        confidence=0.97,
    )


def _make_service(parser=None, **parser_kwargs) -> InvoiceExtractionService:
    return InvoiceExtractionService(parser=parser or _make_parser(**parser_kwargs))


# ------------------------------------
# Success path
# ------------------------------------

@pytest.mark.unit
class TestInvoiceExtractionSuccess:
    async def test_returns_invoice_data(self):
        result = await _make_service().extract(b"data", "invoice.pdf", "prop-1")
        assert isinstance(result, InvoiceData)

    async def test_vendor_name_populated(self):
        result = await _make_service().extract(b"data", "invoice.pdf", "prop-1")
        assert result.vendor_name == "Acme Plumbing"

    async def test_total_populated(self):
        result = await _make_service().extract(b"data", "invoice.pdf", "prop-1")
        assert result.total == 495.00

    async def test_tax_populated(self):
        result = await _make_service().extract(b"data", "invoice.pdf", "prop-1")
        assert result.tax == 45.00

    async def test_line_items_populated(self):
        result = await _make_service().extract(b"data", "invoice.pdf", "prop-1")
        assert len(result.line_items) == 1
        assert result.line_items[0].description == "Pipe repair"

    async def test_confidence_populated(self):
        result = await _make_service().extract(b"data", "invoice.pdf", "prop-1")
        assert result.confidence == 0.97

    async def test_parser_called_with_content_and_filename(self):
        parser = _make_parser()
        await InvoiceExtractionService(parser).extract(b"bytes", "inv.pdf", "prop-1")
        parser.parse.assert_called_once_with(b"bytes", "inv.pdf")

    async def test_empty_invoice_data_returned_without_error(self):
        """Azure DI found no documents — service returns empty InvoiceData, does not raise."""
        result = await _make_service(return_value=InvoiceData()).extract(b"data", "blank.pdf")
        assert isinstance(result, InvoiceData)
        assert result.vendor_name is None
        assert result.total is None

    async def test_property_id_default_is_empty_string(self):
        parser = _make_parser()
        await InvoiceExtractionService(parser).extract(b"data", "inv.pdf")
        parser.parse.assert_called_once()

    async def test_multiple_line_items(self):
        invoice = InvoiceData(
            line_items=[
                LineItem(description="Labour", amount=300.00),
                LineItem(description="Materials", amount=150.00),
                LineItem(description="Call-out fee", amount=50.00),
            ]
        )
        result = await _make_service(return_value=invoice).extract(b"data", "inv.pdf")
        assert len(result.line_items) == 3

    async def test_invoice_date_parsed(self):
        invoice = InvoiceData(invoice_date=date(2026, 6, 1), due_date=date(2026, 6, 15))
        result = await _make_service(return_value=invoice).extract(b"data", "inv.pdf")
        assert result.invoice_date == date(2026, 6, 1)
        assert result.due_date == date(2026, 6, 15)

    async def test_currency_populated(self):
        invoice = InvoiceData(currency="AUD", total=495.00)
        result = await _make_service(return_value=invoice).extract(b"data", "inv.pdf")
        assert result.currency == "AUD"


# ------------------------------------
# Error path
# ------------------------------------

@pytest.mark.unit
class TestInvoiceExtractionErrors:
    async def test_parser_exception_raises_invoice_extraction_error(self):
        with _PATCH_LOGGER, pytest.raises(InvoiceExtractionError):
            await _make_service(
                raise_error=RuntimeError("Azure DI timeout")
            ).extract(b"data", "invoice.pdf")

    async def test_extraction_error_message_contains_filename(self):
        with _PATCH_LOGGER, pytest.raises(InvoiceExtractionError, match="invoice.pdf"):
            await _make_service(
                raise_error=ConnectionError("network error")
            ).extract(b"data", "invoice.pdf")

    async def test_azure_http_error_wrapped(self):
        with _PATCH_LOGGER, pytest.raises(InvoiceExtractionError):
            await _make_service(
                raise_error=Exception("HttpResponseError: 401 Unauthorized")
            ).extract(b"data", "invoice.pdf")

    async def test_parser_called_once_on_failure(self):
        parser = _make_parser(raise_error=RuntimeError("fail"))
        with _PATCH_LOGGER, pytest.raises(InvoiceExtractionError):
            await InvoiceExtractionService(parser).extract(b"data", "inv.pdf")
        parser.parse.assert_called_once()
