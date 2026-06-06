"""
Unit tests for EnquiryService.

classify_rag_intent and get_llm are both patched — no real LLM calls.
RagRetriever is mocked — no real vector store calls.
"""
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.core.config import settings
from app.schemas.enquiry import EnquiryRequest, EnquiryResponse
from app.schemas.rag import INTENT_COMPLIANCE_RULES, RagIntent
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
@patch("app.services.enquiry_service.classify_rag_intent")
class TestDraftResponse:
    async def test_returns_llm_content(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock("Dear Tenant, we will action this shortly.")

        result = await EnquiryService(rag=_make_rag()).draft_response(_make_request())

        assert isinstance(result, EnquiryResponse)
        assert result.draft == "Dear Tenant, we will action this shortly."

    async def test_classifies_before_drafting(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.BOND
        mock_get_llm.return_value = _make_llm_mock()

        await EnquiryService(rag=_make_rag()).draft_response(_make_request(body="bond refund query"))

        mock_classify.assert_called_once_with("bond refund query")

    async def test_intent_label_included_in_human_message(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.RENT_PAYMENT
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock

        await EnquiryService(rag=_make_rag()).draft_response(_make_request())

        call_args = llm_mock.bind.return_value.ainvoke.call_args
        prompt = call_args.args[0]
        human_content = prompt[-1].content  # HumanMessage is always last
        assert "rent_payment" in human_content

    async def test_max_tokens_bound_to_400(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.GENERAL
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock

        await EnquiryService(rag=_make_rag()).draft_response(_make_request())

        llm_mock.bind.assert_called_once_with(max_tokens=400)

    async def test_llm_failure_returns_empty_string(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.MAINTENANCE
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
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        rag = _make_rag()

        await EnquiryService(rag=rag).draft_response(_make_request())

        rag.aretrieve.assert_called_once()
        call_kwargs = rag.aretrieve.call_args.kwargs
        assert call_kwargs["doc_types"] == frozenset(["maintenance_log", "lease", "legislation"])

    async def test_retrieved_docs_included_in_prompt(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.BOND
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
        mock_classify.return_value = RagIntent.WATER_BILL
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock
        rag = _make_rag(nodes=[])

        await EnquiryService(rag=rag).draft_response(_make_request())

        call_args = llm_mock.bind.return_value.ainvoke.call_args
        prompt = call_args.args[0]
        # System + no-docs fallback message + human
        assert len(prompt) == 3
        assert "No relevant tenancy documents" in prompt[1].content

    async def test_rag_failure_logs_error_and_continues(self, mock_classify, mock_get_llm, caplog):
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        rag = _make_rag(raise_error=RuntimeError("vector store down"))

        import logging
        with caplog.at_level(logging.ERROR, logger="app.services.enquiry_service"):
            result = await EnquiryService(rag=rag).draft_response(_make_request())

        assert "enquiry_rag_failed" in caplog.text
        assert result.draft == "Dear Tenant, ..."  # LLM still runs without docs

    async def test_missing_intent_in_doc_types_logs_error(self, mock_classify, mock_get_llm, caplog):
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        rag = _make_rag()

        import logging
        # Patch INTENT_DOC_TYPES to simulate a missing entry
        with patch("app.services.enquiry_service.INTENT_DOC_TYPES", {}):
            with caplog.at_level(logging.ERROR, logger="app.services.enquiry_service"):
                await EnquiryService(rag=rag).draft_response(_make_request())

        assert "enquiry_no_doc_types" in caplog.text

    async def test_missing_intent_skips_rag(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        rag = _make_rag()

        with patch("app.services.enquiry_service.INTENT_DOC_TYPES", {}):
            await EnquiryService(rag=rag).draft_response(_make_request())

        rag.aretrieve.assert_not_called()

    # ── sources ───────────────────────────────────────────────────────────────

    async def test_sources_contain_file_names_from_docs(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.BOND
        mock_get_llm.return_value = _make_llm_mock()
        node1 = _make_node(doc_type="lease", content="...")
        node1.node.metadata["file_name"] = "lease_2024.pdf"
        node2 = _make_node(doc_type="bond_lodgement", content="...")
        node2.node.metadata["file_name"] = "bond_form.pdf"
        rag = _make_rag(nodes=[node1, node2])

        result = await EnquiryService(rag=rag).draft_response(_make_request())

        assert [s.file_name for s in result.sources] == ["lease_2024.pdf", "bond_form.pdf"]

    async def test_sources_preserves_all_chunks_for_same_file(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()
        node1 = _make_node(doc_type="lease", content="clause 1")
        node1.node.metadata["file_name"] = "lease_2024.pdf"
        node2 = _make_node(doc_type="lease", content="clause 2")
        node2.node.metadata["file_name"] = "lease_2024.pdf"
        rag = _make_rag(nodes=[node1, node2])

        result = await EnquiryService(rag=rag).draft_response(_make_request())

        assert [s.file_name for s in result.sources] == ["lease_2024.pdf", "lease_2024.pdf"]

    async def test_sources_empty_when_no_docs(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.GENERAL
        mock_get_llm.return_value = _make_llm_mock()

        result = await EnquiryService(rag=_make_rag(nodes=[])).draft_response(_make_request())

        assert result.sources == []

    async def test_sources_empty_on_llm_failure(self, mock_classify, mock_get_llm):
        mock_classify.return_value = RagIntent.MAINTENANCE
        llm = MagicMock()
        llm.bind.return_value = MagicMock(ainvoke=AsyncMock(side_effect=RuntimeError("down")))
        mock_get_llm.return_value = llm
        rag = _make_rag(nodes=[])

        result = await EnquiryService(rag=rag).draft_response(_make_request())

        assert result.sources == []


# ── helpers for stream tests ───────────────────────────────────────────────────

async def _collect(gen) -> tuple[list[dict], bool]:
    """
    Drain an async generator from stream_draft_response.
    Returns (list of parsed JSON event dicts, True if data: [DONE] was seen).
    """
    events: list[dict] = []
    done = False
    async for chunk in gen:
        stripped = chunk.strip()
        if stripped == "data: [DONE]":
            done = True
        elif stripped.startswith("data: "):
            events.append(json.loads(stripped.removeprefix("data: ")))
    return events, done


def _make_llm_failure():
    llm = MagicMock()
    llm.bind.return_value = MagicMock(ainvoke=AsyncMock(side_effect=RuntimeError("LLM down")))
    return llm


# ── stream_draft_response ──────────────────────────────────────────────────────

@patch("app.services.enquiry_service.get_llm")
@patch("app.services.enquiry_service.classify_rag_intent")
class TestStreamDraftResponse:

    # ── happy path ─────────────────────────────────────────────────────────────

    async def test_four_step_events_emitted_in_order(self, mock_classify, mock_get_llm):
        """All four pipeline steps appear in the correct order."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()

        events, _ = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        steps = [e["step"] for e in events if e["type"] == "step"]
        assert steps == ["intent_classified", "rag_retrieval", "llm_draft", "compliance_check"]

    async def test_intent_classified_meta_contains_intent_value(self, mock_classify, mock_get_llm):
        """intent_classified meta starts with the classified intent slug."""
        mock_classify.return_value = RagIntent.BOND
        mock_get_llm.return_value = _make_llm_mock()

        events, _ = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        step = next(e for e in events if e.get("step") == "intent_classified")
        assert step["meta"].startswith("bond ·")

    async def test_rag_retrieval_meta_contains_doc_count(self, mock_classify, mock_get_llm):
        """rag_retrieval meta reflects the number of documents retrieved."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()

        events, _ = await _collect(
            EnquiryService(rag=_make_rag(nodes=[_make_node(), _make_node()]))
            .stream_draft_response(_make_request())
        )

        step = next(e for e in events if e.get("step") == "rag_retrieval")
        assert step["meta"].startswith("2 docs ·")

    async def test_llm_draft_meta_contains_model_name(self, mock_classify, mock_get_llm):
        """llm_draft meta starts with the configured model name."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()

        events, _ = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        step = next(e for e in events if e.get("step") == "llm_draft")
        assert step["meta"].startswith(settings.OPENAI_MODEL_NAME)

    async def test_compliance_check_meta_matches_intent_rule(self, mock_classify, mock_get_llm):
        """compliance_check meta matches the INTENT_COMPLIANCE_RULES lookup for the intent."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()

        events, _ = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        step = next(e for e in events if e.get("step") == "compliance_check")
        assert step["meta"] == INTENT_COMPLIANCE_RULES[RagIntent.MAINTENANCE]

    async def test_result_event_contains_draft_content(self, mock_classify, mock_get_llm):
        """result event carries the LLM-generated draft."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock("Dear Tenant, we will action this shortly.")

        events, _ = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        result = next(e for e in events if e["type"] == "result")
        assert result["draft"] == "Dear Tenant, we will action this shortly."

    async def test_result_event_contains_sources_from_doc_metadata(self, mock_classify, mock_get_llm):
        """result event sources list is populated from retrieved node file_name metadata."""
        mock_classify.return_value = RagIntent.BOND
        mock_get_llm.return_value = _make_llm_mock()
        node = _make_node(doc_type="bond_lodgement")
        node.node.metadata["file_name"] = "bond_form.pdf"

        events, _ = await _collect(
            EnquiryService(rag=_make_rag(nodes=[node])).stream_draft_response(_make_request())
        )

        result = next(e for e in events if e["type"] == "result")
        assert [s["file_name"] for s in result["sources"]] == ["bond_form.pdf"]

    async def test_done_sentinel_is_emitted(self, mock_classify, mock_get_llm):
        """Generator always ends with data: [DONE]."""
        mock_classify.return_value = RagIntent.GENERAL
        mock_get_llm.return_value = _make_llm_mock()

        _, done = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        assert done is True

    # ── intent classification failure ──────────────────────────────────────────

    async def test_intent_failure_emits_error_event(self, mock_classify, mock_get_llm):
        """If intent classification raises, an error event is yielded."""
        mock_classify.side_effect = RuntimeError("classifier down")

        events, _ = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1
        assert "intent" in error_events[0]["message"].lower()

    async def test_intent_failure_still_emits_done(self, mock_classify, mock_get_llm):
        """Generator terminates with [DONE] even after intent classification failure."""
        mock_classify.side_effect = RuntimeError("classifier down")

        _, done = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        assert done is True

    async def test_intent_failure_does_not_call_llm(self, mock_classify, mock_get_llm):
        """No LLM call is made when intent classification fails."""
        mock_classify.side_effect = RuntimeError("classifier down")

        await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        mock_get_llm.assert_not_called()

    # ── RAG retrieval failure ──────────────────────────────────────────────────

    async def test_rag_failure_emits_step_with_zero_docs(self, mock_classify, mock_get_llm):
        """rag_retrieval step is still emitted with '0 docs' when the retriever raises."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()

        events, _ = await _collect(
            EnquiryService(rag=_make_rag(raise_error=RuntimeError("vector store down")))
            .stream_draft_response(_make_request())
        )

        step = next(e for e in events if e.get("step") == "rag_retrieval")
        assert step["meta"].startswith("0 docs ·")

    async def test_rag_failure_continues_to_llm_and_emits_result(self, mock_classify, mock_get_llm):
        """LLM still runs and result is emitted even when RAG retrieval fails."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock("Draft despite RAG failure.")

        events, _ = await _collect(
            EnquiryService(rag=_make_rag(raise_error=RuntimeError("vector store down")))
            .stream_draft_response(_make_request())
        )

        result = next((e for e in events if e["type"] == "result"), None)
        assert result is not None
        assert result["draft"] == "Draft despite RAG failure."

    async def test_rag_failure_logs_error(self, mock_classify, mock_get_llm, caplog):
        """A RAG retrieval error is recorded at ERROR level."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_mock()

        with caplog.at_level(logging.ERROR, logger="app.services.enquiry_service"):
            await _collect(
                EnquiryService(rag=_make_rag(raise_error=RuntimeError("vector store down")))
                .stream_draft_response(_make_request())
            )

        assert "enquiry_stream_rag_failed" in caplog.text

    # ── LLM draft failure ──────────────────────────────────────────────────────

    async def test_llm_failure_emits_error_event(self, mock_classify, mock_get_llm):
        """If the LLM call raises, an error event is yielded."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_failure()

        events, _ = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        error_events = [e for e in events if e["type"] == "error"]
        assert len(error_events) == 1

    async def test_llm_failure_still_emits_done(self, mock_classify, mock_get_llm):
        """Generator terminates with [DONE] even after an LLM failure."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_failure()

        _, done = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        assert done is True

    async def test_llm_failure_does_not_emit_result(self, mock_classify, mock_get_llm):
        """No result event is emitted when the LLM call fails."""
        mock_classify.return_value = RagIntent.MAINTENANCE
        mock_get_llm.return_value = _make_llm_failure()

        events, _ = await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        assert not any(e["type"] == "result" for e in events)

    # ── prompt construction ────────────────────────────────────────────────────

    async def test_docs_context_added_to_prompt_when_rag_returns_docs(self, mock_classify, mock_get_llm):
        """A SystemMessage with retrieved doc text is injected before the HumanMessage."""
        mock_classify.return_value = RagIntent.BOND
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock
        node = _make_node(doc_type="bond_lodgement", content="Bond amount: $2000")

        await _collect(
            EnquiryService(rag=_make_rag(nodes=[node])).stream_draft_response(_make_request())
        )

        prompt = llm_mock.bind.return_value.ainvoke.call_args.args[0]
        assert len(prompt) == 3  # system + docs context + human
        assert "bond_lodgement" in prompt[1].content
        assert "Bond amount: $2000" in prompt[1].content

    async def test_no_docs_context_when_rag_returns_empty(self, mock_classify, mock_get_llm):
        """No-docs fallback SystemMessage is injected when RAG returns no documents."""
        mock_classify.return_value = RagIntent.GENERAL
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock

        await _collect(
            EnquiryService(rag=_make_rag(nodes=[])).stream_draft_response(_make_request())
        )

        prompt = llm_mock.bind.return_value.ainvoke.call_args.args[0]
        # System + no-docs fallback message + human
        assert len(prompt) == 3
        assert "No relevant tenancy documents" in prompt[1].content

    async def test_intent_label_included_in_human_message(self, mock_classify, mock_get_llm):
        """The HumanMessage always carries a [intent: <value>] prefix."""
        mock_classify.return_value = RagIntent.RENT_PAYMENT
        llm_mock = _make_llm_mock()
        mock_get_llm.return_value = llm_mock

        await _collect(
            EnquiryService(rag=_make_rag()).stream_draft_response(_make_request())
        )

        prompt = llm_mock.bind.return_value.ainvoke.call_args.args[0]
        assert "rent_payment" in prompt[-1].content
