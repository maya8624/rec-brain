"""
Unit tests for EnquiryService.

classify_enquiry_intent and get_llm are both patched — no real LLM calls.
RagRetriever is mocked — no real vector store calls.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.enquiry import EnquiryIntent, EnquiryRequest, EnquiryResponse
from app.services.enquiry_service import EnquiryService


def _make_request(body: str = "The tap is leaking", tenant_id: str = "user-1") -> EnquiryRequest:
    return EnquiryRequest(id="req-1", body=body, tenant_id=tenant_id, property_id=None, intent=None)


def _make_llm_mock(content: str = "Dear Tenant, ...") -> MagicMock:
    result = MagicMock()
    result.content = content
    bound = MagicMock()
    bound.ainvoke = AsyncMock(return_value=result)
    llm = MagicMock()
    llm.bind.return_value = bound
    return llm


def _make_rag(nodes=None, raise_error=None) -> MagicMock:
    mock = MagicMock()
    if raise_error:
        mock.aretrieve = AsyncMock(side_effect=raise_error)
    else:
        mock.aretrieve = AsyncMock(return_value=nodes or [])
    return mock


def _make_node(doc_type: str = "lease", content: str = "Sample content") -> MagicMock:
    node = MagicMock()
    node.node.metadata = {"doc_type": doc_type}
    node.node.get_content.return_value = content
    return node


@patch("app.services.enquiry_service.get_llm")
@patch("app.services.enquiry_service.classify_enquiry_intent")
class TestDraftResponse:
    async def test_returns_llm_content(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock("Dear Tenant, we will action this shortly.")

        result = await EnquiryService(rag=_make_rag()).draft_response(_make_request())

        assert isinstance(result, EnquiryResponse)
        assert result.draft == "Dear Tenant, we will action this shortly."

    async def test_classifies_before_drafting(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.BOND
        mock_get_llm.return_value = _make_llm_mock()

        await EnquiryService(rag=_make_rag()).draft_response(_make_request(body="bond refund query"))

        mock_classify.assert_called_once_with("bond refund query")

    async def test_intent_label_included_in_human_message(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.RENT_PAYMENT
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock

        await EnquiryService(rag=_make_rag()).draft_response(_make_request())

        call_args = llm_mock.bind.return_value.ainvoke.call_args
        prompt = call_args.args[0]
        human_content = prompt[-1].content  # HumanMessage is always last
        assert "rent_payment" in human_content

    async def test_max_tokens_bound_to_400(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.GENERAL
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock

        await EnquiryService(rag=_make_rag()).draft_response(_make_request())

        llm_mock.bind.assert_called_once_with(max_tokens=400)

    async def test_llm_failure_returns_empty_string(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.MAINTENANCE
        llm = MagicMock()
        llm.bind.return_value = MagicMock(
            ainvoke=AsyncMock(side_effect=RuntimeError("LLM down"))
        )
        mock_get_llm.return_value = llm

        result = await EnquiryService(rag=_make_rag()).draft_response(_make_request())

        assert isinstance(result, EnquiryResponse)
        assert result.draft == ""

    # ── RAG retrieval ──────────────────────────────────────────────────────────

    async def test_aretrieve_called_with_correct_doc_types(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        rag = _make_rag()

        await EnquiryService(rag=rag).draft_response(_make_request())

        rag.aretrieve.assert_called_once()
        call_kwargs = rag.aretrieve.call_args.kwargs
        assert call_kwargs["doc_types"] == frozenset(["maintenance_log", "lease"])

    async def test_retrieved_docs_included_in_prompt(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.BOND
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock
        node = _make_node(doc_type="bond_lodgement", content="Bond amount: $2000")
        rag = _make_rag(nodes=[node])

        await EnquiryService(rag=rag).draft_response(_make_request())

        call_args = llm_mock.bind.return_value.ainvoke.call_args
        prompt = call_args.args[0]
        # System + docs context + human = 3 messages
        assert len(prompt) == 3
        assert "bond_lodgement" in prompt[1].content
        assert "Bond amount: $2000" in prompt[1].content

    async def test_no_docs_context_message_when_rag_returns_empty(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.WATER_BILL
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock
        rag = _make_rag(nodes=[])

        await EnquiryService(rag=rag).draft_response(_make_request())

        call_args = llm_mock.bind.return_value.ainvoke.call_args
        prompt = call_args.args[0]
        # System + human only — no docs message
        assert len(prompt) == 2

    async def test_rag_failure_logs_error_and_continues(self, mock_classify, mock_get_llm, caplog):
        mock_classify.return_value = EnquiryIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        rag = _make_rag(raise_error=RuntimeError("vector store down"))

        import logging
        with caplog.at_level(logging.ERROR, logger="app.services.enquiry_service"):
            result = await EnquiryService(rag=rag).draft_response(_make_request())

        assert "RAG retrieval failed" in caplog.text
        assert result.draft == "Dear Tenant, ..."  # LLM still runs without docs

    async def test_missing_intent_in_doc_types_logs_error(self, mock_classify, mock_get_llm, caplog):
        mock_classify.return_value = EnquiryIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        rag = _make_rag()

        import logging
        # Patch INTENT_DOC_TYPES to simulate a missing entry
        with patch("app.services.enquiry_service.INTENT_DOC_TYPES", {}):
            with caplog.at_level(logging.ERROR, logger="app.services.enquiry_service"):
                await EnquiryService(rag=rag).draft_response(_make_request())

        assert "No doc types mapped for intent" in caplog.text

    async def test_missing_intent_skips_rag(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        rag = _make_rag()

        with patch("app.services.enquiry_service.INTENT_DOC_TYPES", {}):
            await EnquiryService(rag=rag).draft_response(_make_request())

        rag.aretrieve.assert_not_called()

    # ── sources ───────────────────────────────────────────────────────────────

    async def test_sources_contain_file_names_from_docs(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.BOND
        mock_get_llm.return_value = _make_llm_mock()
        node1 = _make_node(doc_type="lease", content="...")
        node1.node.metadata["file_name"] = "lease_2024.pdf"
        node2 = _make_node(doc_type="bond_lodgement", content="...")
        node2.node.metadata["file_name"] = "bond_form.pdf"
        rag = _make_rag(nodes=[node1, node2])

        result = await EnquiryService(rag=rag).draft_response(_make_request())

        assert result.sources == ["lease_2024.pdf", "bond_form.pdf"]

    async def test_sources_deduplicates_same_file(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        node1 = _make_node(doc_type="lease", content="clause 1")
        node1.node.metadata["file_name"] = "lease_2024.pdf"
        node2 = _make_node(doc_type="lease", content="clause 2")
        node2.node.metadata["file_name"] = "lease_2024.pdf"  # same file
        rag = _make_rag(nodes=[node1, node2])

        result = await EnquiryService(rag=rag).draft_response(_make_request())

        assert result.sources == ["lease_2024.pdf"]  # deduplicated

    async def test_sources_empty_when_no_docs(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.GENERAL
        mock_get_llm.return_value = _make_llm_mock()

        result = await EnquiryService(rag=_make_rag(nodes=[])).draft_response(_make_request())

        assert result.sources == []

    async def test_sources_empty_on_llm_failure(self, mock_classify, mock_get_llm):
        mock_classify.return_value = EnquiryIntent.MAINTENANCE
        llm = MagicMock()
        llm.bind.return_value = MagicMock(ainvoke=AsyncMock(side_effect=RuntimeError("down")))
        mock_get_llm.return_value = llm
        rag = _make_rag(nodes=[])

        result = await EnquiryService(rag=rag).draft_response(_make_request())

        assert result.sources == []
