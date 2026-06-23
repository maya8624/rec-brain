from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.core.constants import InvoiceToolNames


class LineItem(BaseModel):
    description: str | None = None
    quantity: float | None = None
    unit_price: float | None = None
    amount: float | None = None


class InvoiceData(BaseModel):
    vendor_name: str | None = None
    vendor_address: str | None = None
    customer_name: str | None = None
    invoice_id: str | None = None
    invoice_date: date | None = None
    due_date: date | None = None
    subtotal: float | None = None
    tax: float | None = None
    total: float | None = None
    currency: str | None = None
    line_items: list[LineItem] = []
    confidence: float = 0.0


class InvoiceExtractionResponse(BaseModel):
    tool_name: Literal["save_invoice"] = InvoiceToolNames.SAVE_INVOICE
    success: bool
    filename: str
    property_id: str
    data: InvoiceData
