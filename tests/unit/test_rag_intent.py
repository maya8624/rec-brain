"""
Unit tests for classify_rag_intent and its keyword fast path.

Fast path: keyword matching — no LLM call.
LLM path:  patched LLM — tests fallback and structured output handling.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.agents.nodes.rag_intent import classify_rag_intent
from app.schemas.rag import RagClassification, RagIntent


# ── keyword fast path ─────────────────────────────────────────────────────────

class TestKeywordFastPath:
    async def test_water_bill_keyword(self):
        result = await classify_rag_intent("I have a question about my water bill")
        assert result == RagIntent.WATER_BILL

    async def test_water_usage_keyword(self):
        result = await classify_rag_intent("Can you explain the water usage charge?")
        assert result == RagIntent.WATER_BILL

    async def test_maintenance_leak_keyword(self):
        result = await classify_rag_intent("The tap is leaking in the bathroom")
        assert result == RagIntent.MAINTENANCE

    async def test_maintenance_broken_keyword(self):
        result = await classify_rag_intent("The dishwasher is broken")
        assert result == RagIntent.MAINTENANCE

    async def test_maintenance_plumber_keyword(self):
        result = await classify_rag_intent("Do I need to arrange a plumber?")
        assert result == RagIntent.MAINTENANCE

    async def test_bond_keyword(self):
        result = await classify_rag_intent("When will I get my bond back?")
        assert result == RagIntent.BOND

    async def test_bond_refund_keyword(self):
        result = await classify_rag_intent("I'd like to request a bond refund")
        assert result == RagIntent.BOND

    async def test_security_deposit_keyword(self):
        result = await classify_rag_intent("What happens to my security deposit?")
        assert result == RagIntent.BOND

    async def test_rent_payment_keyword(self):
        result = await classify_rag_intent("I want to pay rent this week")
        assert result == RagIntent.RENT_PAYMENT

    async def test_overdue_rent_keyword(self):
        result = await classify_rag_intent("I received a notice about overdue rent")
        assert result == RagIntent.RENT_PAYMENT

    async def test_lease_renewal_keyword(self):
        result = await classify_rag_intent("I'd like to renew my lease")
        assert result == RagIntent.LEASE_RENEWAL

    async def test_end_of_lease_keyword(self):
        result = await classify_rag_intent("What happens at the end of lease?")
        assert result == RagIntent.LEASE_RENEWAL

    async def test_inspection_keyword(self):
        result = await classify_rag_intent("I received a routine inspection notice")
        assert result == RagIntent.INSPECTION

    async def test_entry_notice_keyword(self):
        result = await classify_rag_intent("I got an entry notice for next week")
        assert result == RagIntent.INSPECTION

    async def test_lease_clause_keyword(self):
        result = await classify_rag_intent("I have a question about a lease clause")
        assert result == RagIntent.LEASE_CLAUSE

    async def test_breach_of_lease_keyword(self):
        result = await classify_rag_intent("Is this a breach of lease?")
        assert result == RagIntent.LEASE_CLAUSE

    async def test_document_request_keyword(self):
        result = await classify_rag_intent("Can I get a copy of lease please?")
        assert result == RagIntent.DOCUMENT_REQUEST

    async def test_rental_statement_keyword(self):
        result = await classify_rag_intent("I need a rental statement for my accountant")
        assert result == RagIntent.DOCUMENT_REQUEST

    async def test_empty_message_returns_general(self):
        result = await classify_rag_intent("")
        assert result == RagIntent.GENERAL

    async def test_fast_path_is_case_insensitive(self):
        result = await classify_rag_intent("WATER BILL query")
        assert result == RagIntent.WATER_BILL


# ── LLM path ──────────────────────────────────────────────────────────────────

def _make_llm_mock(intent: RagIntent) -> MagicMock:
    structured = AsyncMock(return_value=RagClassification(intent=intent))
    llm = MagicMock()
    llm.with_structured_output.return_value = MagicMock(ainvoke=structured)
    return llm


@patch("app.agents.nodes.rag_intent.get_llm")
class TestLLMPath:
    async def test_ambiguous_message_calls_llm(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock(RagIntent.GENERAL)
        await classify_rag_intent("I have a question about my tenancy")
        mock_get_llm.assert_called_once()

    async def test_llm_returns_correct_intent(self, mock_get_llm):
        mock_get_llm.return_value = _make_llm_mock(RagIntent.MAINTENANCE)
        result = await classify_rag_intent("Something is wrong in the unit")
        assert result == RagIntent.MAINTENANCE

    async def test_llm_failure_falls_back_to_general(self, mock_get_llm):
        llm = MagicMock()
        llm.with_structured_output.return_value = MagicMock(
            ainvoke=AsyncMock(side_effect=RuntimeError("LLM down"))
        )
        mock_get_llm.return_value = llm
        result = await classify_rag_intent("Something is wrong in the unit")
        assert result == RagIntent.GENERAL

    async def test_keyword_match_skips_llm(self, mock_get_llm):
        await classify_rag_intent("my water bill seems too high")
        mock_get_llm.assert_not_called()
