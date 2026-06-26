"""
Unit tests for DocumentTypeClassifier.
Azure DI and LLM are fully mocked — no credentials required.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.infrastructure.document_classifier import (
    DocumentTypeClassification,
    DocumentTypeClassifier,
)

_PATCH_LOGGER = patch("app.infrastructure.document_classifier.logger")


# ------------------------------------
# _classify_from_text (pure function)
# ------------------------------------

@pytest.mark.unit
class TestClassifyFromText:
    def test_receipt_keyword_returns_receipt(self):
        assert DocumentTypeClassifier._classify_from_text("eftpos payment accepted") == "receipt"

    def test_sale_tax_invoice_phrase_returns_receipt(self):
        assert DocumentTypeClassifier._classify_from_text("sale tax invoice\nbunnings warehouse") == "receipt"

    def test_cashier_returns_receipt(self):
        assert DocumentTypeClassifier._classify_from_text("cashier: jane\nchange due: $5.00") == "receipt"

    def test_invoice_keyword_returns_invoice(self):
        assert DocumentTypeClassifier._classify_from_text("invoice number: inv-0042") == "invoice"

    def test_bill_to_returns_invoice(self):
        assert DocumentTypeClassifier._classify_from_text("bill to: sunshine realty pty ltd") == "invoice"

    def test_due_date_returns_invoice(self):
        assert DocumentTypeClassifier._classify_from_text("due date: 30 june 2026\npayment terms: net 30") == "invoice"

    def test_no_keywords_returns_none(self):
        assert DocumentTypeClassifier._classify_from_text("thank you for your business") is None

    def test_both_sets_present_returns_none(self):
        assert DocumentTypeClassifier._classify_from_text("receipt\nbill to: acme corp") is None

    def test_receipt_keyword_in_uppercase_not_matched(self):
        # Input must be pre-lowercased by caller; uppercase not matched
        assert DocumentTypeClassifier._classify_from_text("RECEIPT") is None

    def test_change_colon_matched(self):
        assert DocumentTypeClassifier._classify_from_text("change: $2.50") == "receipt"


# ------------------------------------
# _match_keyword
# ------------------------------------

@pytest.mark.unit
class TestMatchKeyword:
    def test_whole_word_match(self):
        assert DocumentTypeClassifier._match_keyword("paid by eftpos", "eftpos") is True

    def test_substring_not_matched(self):
        # "cashier" inside "cashierdesk" should not match as whole-word keyword
        assert DocumentTypeClassifier._match_keyword("cashierdesk", "cashier") is False

    def test_multi_word_phrase_matched(self):
        assert DocumentTypeClassifier._match_keyword("sale tax invoice", "sale tax invoice") is True

    def test_keyword_ending_in_colon_matched(self):
        assert DocumentTypeClassifier._match_keyword("change: $2.50", "change:") is True

    def test_no_match_returns_false(self):
        assert DocumentTypeClassifier._match_keyword("hello world", "eftpos") is False


# ------------------------------------
# classify() — fast-path
# ------------------------------------

@pytest.mark.unit
class TestClassifyFastPath:
    async def test_receipt_fast_path(self):
        classifier = DocumentTypeClassifier.__new__(DocumentTypeClassifier)
        with (
            patch.object(classifier, "_read_text", return_value="eftpos payment accepted"),
            _PATCH_LOGGER,
        ):
            result = await classifier.classify(b"data", "scan.jpg")
        assert result == "receipt"

    async def test_invoice_fast_path(self):
        classifier = DocumentTypeClassifier.__new__(DocumentTypeClassifier)
        with (
            patch.object(classifier, "_read_text", return_value="Invoice Number: INV-001\nBill To: ABC Corp"),
            _PATCH_LOGGER,
        ):
            result = await classifier.classify(b"data", "invoice.pdf")
        assert result == "invoice"

    async def test_fast_path_does_not_call_llm(self):
        classifier = DocumentTypeClassifier.__new__(DocumentTypeClassifier)
        with (
            patch.object(classifier, "_read_text", return_value="eftpos cashier"),
            patch("app.infrastructure.document_classifier.get_llm") as mock_llm,
            _PATCH_LOGGER,
        ):
            await classifier.classify(b"data", "receipt.pdf")
        mock_llm.assert_not_called()


# ------------------------------------
# classify() — LLM fallback
# ------------------------------------

@pytest.mark.unit
class TestClassifyLlmFallback:
    async def test_llm_fallback_called_when_inconclusive(self):
        classifier = DocumentTypeClassifier.__new__(DocumentTypeClassifier)
        mock_llm_chain = AsyncMock()
        mock_llm_chain.ainvoke.return_value = DocumentTypeClassification(doc_type="receipt")

        with (
            patch.object(classifier, "_read_text", return_value="thank you for visiting"),
            patch("app.infrastructure.document_classifier.get_llm") as mock_get_llm,
            _PATCH_LOGGER,
        ):
            mock_get_llm.return_value.with_structured_output.return_value = mock_llm_chain
            result = await classifier.classify(b"data", "doc.pdf")

        assert result == "receipt"
        mock_llm_chain.ainvoke.assert_called_once()

    async def test_llm_fallback_returns_invoice(self):
        classifier = DocumentTypeClassifier.__new__(DocumentTypeClassifier)
        mock_llm_chain = AsyncMock()
        mock_llm_chain.ainvoke.return_value = DocumentTypeClassification(doc_type="invoice")

        with (
            patch.object(classifier, "_read_text", return_value="general document text"),
            patch("app.infrastructure.document_classifier.get_llm") as mock_get_llm,
            _PATCH_LOGGER,
        ):
            mock_get_llm.return_value.with_structured_output.return_value = mock_llm_chain
            result = await classifier.classify(b"data", "doc.pdf")

        assert result == "invoice"

    async def test_llm_error_defaults_to_invoice(self):
        classifier = DocumentTypeClassifier.__new__(DocumentTypeClassifier)
        mock_llm_chain = AsyncMock()
        mock_llm_chain.ainvoke.side_effect = RuntimeError("LLM timeout")

        with (
            patch.object(classifier, "_read_text", return_value="ambiguous text here"),
            patch("app.infrastructure.document_classifier.get_llm") as mock_get_llm,
            _PATCH_LOGGER,
        ):
            mock_get_llm.return_value.with_structured_output.return_value = mock_llm_chain
            result = await classifier.classify(b"data", "doc.pdf")

        assert result == "invoice"
