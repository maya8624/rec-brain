"""
Unit tests for AzureInvoiceParser field extractors.
All tests use MagicMock DocumentField objects — no Azure credentials required.
"""
from datetime import date
from unittest.mock import MagicMock

import pytest

from app.infrastructure.invoice_parser import (
    _address,
    _amount,
    _currency_symbol,
    _date,
    _line_items,
    _str,
)


# ------------------------------------
# Mock builders
# ------------------------------------

def _field(
    value_string=None,
    value_date=None,
    value_currency=None,
    value_number=None,
    value_array=None,
    value_object=None,
    content=None,
) -> MagicMock:
    f = MagicMock()
    f.value_string = value_string
    f.value_date = value_date
    f.value_currency = value_currency
    f.value_number = value_number
    f.value_array = value_array
    f.value_object = value_object
    f.content = content
    return f


def _currency(amount: float, symbol: str = "$") -> MagicMock:
    cv = MagicMock()
    cv.amount = amount
    cv.symbol = symbol
    return cv


def _item_field(
    description: str | None = None,
    quantity: float | None = None,
    unit_price: float | None = None,
    amount: float | None = None,
) -> MagicMock:
    """Build a mock DocumentField representing a single Items[] entry."""
    obj = {}
    if description is not None:
        obj["Description"] = _field(value_string=description)
    if quantity is not None:
        obj["Quantity"] = _field(value_number=quantity)
    if unit_price is not None:
        obj["UnitPrice"] = _field(value_currency=_currency(unit_price))
    if amount is not None:
        obj["Amount"] = _field(value_currency=_currency(amount))
    return _field(value_object=obj)


# ------------------------------------
# _str
# ------------------------------------

@pytest.mark.unit
class TestStrExtractor:
    def test_returns_string_value(self):
        assert _str({"VendorName": _field(value_string="Acme Corp")}, "VendorName") == "Acme Corp"

    def test_returns_none_when_field_absent(self):
        assert _str({}, "VendorName") is None

    def test_returns_none_when_value_is_none(self):
        assert _str({"VendorName": _field(value_string=None)}, "VendorName") is None

    def test_empty_string_returned_as_is(self):
        assert _str({"InvoiceId": _field(value_string="")}, "InvoiceId") == ""


# ------------------------------------
# _address
# ------------------------------------

@pytest.mark.unit
class TestAddressExtractor:
    def test_returns_content(self):
        assert _address(
            {"VendorAddress": _field(content="12 George St, Sydney NSW 2000")},
            "VendorAddress",
        ) == "12 George St, Sydney NSW 2000"

    def test_returns_none_when_field_absent(self):
        assert _address({}, "VendorAddress") is None


# ------------------------------------
# _date
# ------------------------------------

@pytest.mark.unit
class TestDateExtractor:
    def test_returns_date_value(self):
        d = date(2026, 6, 1)
        assert _date({"InvoiceDate": _field(value_date=d)}, "InvoiceDate") == d

    def test_returns_none_when_field_absent(self):
        assert _date({}, "InvoiceDate") is None

    def test_returns_none_when_value_is_none(self):
        assert _date({"DueDate": _field(value_date=None)}, "DueDate") is None


# ------------------------------------
# _amount
# ------------------------------------

@pytest.mark.unit
class TestAmountExtractor:
    def test_returns_amount_from_currency_value(self):
        assert _amount({"InvoiceTotal": _field(value_currency=_currency(495.00))}, "InvoiceTotal") == 495.00

    def test_returns_none_when_field_absent(self):
        assert _amount({}, "InvoiceTotal") is None

    def test_returns_none_when_value_currency_is_none(self):
        assert _amount({"SubTotal": _field(value_currency=None)}, "SubTotal") is None

    def test_zero_amount_returned_as_zero(self):
        assert _amount({"TotalTax": _field(value_currency=_currency(0.0))}, "TotalTax") == 0.0

    def test_decimal_amount_preserved(self):
        assert _amount({"SubTotal": _field(value_currency=_currency(1234.56))}, "SubTotal") == 1234.56


# ------------------------------------
# _currency_symbol
# ------------------------------------

@pytest.mark.unit
class TestCurrencySymbolExtractor:
    def test_returns_symbol(self):
        assert _currency_symbol(
            {"InvoiceTotal": _field(value_currency=_currency(495.00, "$"))},
            "InvoiceTotal",
        ) == "$"

    def test_returns_none_when_field_absent(self):
        assert _currency_symbol({}, "InvoiceTotal") is None

    def test_returns_none_when_no_currency(self):
        assert _currency_symbol({"InvoiceTotal": _field(value_currency=None)}, "InvoiceTotal") is None

    def test_aud_symbol(self):
        assert _currency_symbol(
            {"InvoiceTotal": _field(value_currency=_currency(100.0, "AUD"))},
            "InvoiceTotal",
        ) == "AUD"


# ------------------------------------
# _line_items
# ------------------------------------

@pytest.mark.unit
class TestLineItemsExtractor:
    def test_returns_empty_list_when_no_items_field(self):
        assert _line_items({}) == []

    def test_returns_empty_list_when_value_array_is_none(self):
        assert _line_items({"Items": _field(value_array=None)}) == []

    def test_returns_empty_list_when_array_is_empty(self):
        assert _line_items({"Items": _field(value_array=[])}) == []

    def test_single_item_description(self):
        items = _line_items({
            "Items": _field(value_array=[_item_field(description="Pipe repair")])
        })
        assert len(items) == 1
        assert items[0].description == "Pipe repair"

    def test_single_item_quantity(self):
        items = _line_items({
            "Items": _field(value_array=[_item_field(quantity=2.0)])
        })
        assert items[0].quantity == 2.0

    def test_single_item_unit_price(self):
        items = _line_items({
            "Items": _field(value_array=[_item_field(unit_price=225.00)])
        })
        assert items[0].unit_price == 225.00

    def test_single_item_amount(self):
        items = _line_items({
            "Items": _field(value_array=[_item_field(amount=450.00)])
        })
        assert items[0].amount == 450.00

    def test_multiple_items(self):
        items = _line_items({
            "Items": _field(value_array=[
                _item_field(description="Labour", amount=300.00),
                _item_field(description="Materials", amount=150.00),
            ])
        })
        assert len(items) == 2
        assert items[0].description == "Labour"
        assert items[1].description == "Materials"

    def test_missing_description_is_none(self):
        items = _line_items({
            "Items": _field(value_array=[_item_field(amount=100.00)])
        })
        assert items[0].description is None

    def test_missing_amount_is_none(self):
        items = _line_items({
            "Items": _field(value_array=[_item_field(description="Call-out fee")])
        })
        assert items[0].amount is None

    def test_item_with_no_fields_returns_all_none(self):
        empty_item = _field(value_object={})
        items = _line_items({"Items": _field(value_array=[empty_item])})
        assert items[0].description is None
        assert items[0].quantity is None
        assert items[0].unit_price is None
        assert items[0].amount is None

    def test_item_with_none_value_object_returns_all_none(self):
        null_item = _field(value_object=None)
        items = _line_items({"Items": _field(value_array=[null_item])})
        assert items[0].description is None
        assert items[0].amount is None
