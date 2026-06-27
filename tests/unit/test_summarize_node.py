"""
Unit tests for summarize_node.
"""
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agents.nodes.summarize import summarize_node
from app.core.constants import StateKeys


def _make_state(
    messages: list,
    intent: str = "document_query",
    summary: str | None = None,
    summary_message_count: int = 0,
) -> dict:
    return {
        "messages": messages,
        "user_intent": intent,
        "conversation_summary": summary,
        "summary_message_count": summary_message_count,
    }


def _human_ai_pair(i: int) -> list:
    return [
        HumanMessage(content=f"question {i}"),
        AIMessage(content=f"answer {i}"),
    ]


_PATCH = patch("app.agents.nodes.summarize.get_llm")


@pytest.mark.unit
class TestSummarizeNodeNoOp:
    async def test_skips_when_messages_within_window(self):
        """document_query window=4 — 4 Human+AI messages → nothing to summarise."""
        msgs = _human_ai_pair(1) + _human_ai_pair(2)  # 4 messages
        with _PATCH as mock_get_llm:
            result = await summarize_node(_make_state(msgs, intent="document_query"))
        assert result == {}
        mock_get_llm.assert_not_called()

    async def test_skips_for_booking_intent(self):
        """booking is not in SUMMARY_INTENTS."""
        msgs = _human_ai_pair(1) * 10
        with _PATCH as mock_get_llm:
            result = await summarize_node(_make_state(msgs, intent="booking"))
        assert result == {}
        mock_get_llm.assert_not_called()

    async def test_skips_for_cancellation_intent(self):
        msgs = _human_ai_pair(1) * 10
        with _PATCH as mock_get_llm:
            result = await summarize_node(_make_state(msgs, intent="cancellation"))
        assert result == {}
        mock_get_llm.assert_not_called()

    async def test_skips_when_no_new_overflow(self):
        """summary_message_count already covers the entire overflow."""
        # document_query window=4; 6 messages → overflow=2; already summarised=2
        msgs = _human_ai_pair(1) + _human_ai_pair(2) + _human_ai_pair(3)  # 6 msgs
        with _PATCH as mock_get_llm:
            result = await summarize_node(
                _make_state(msgs, intent="document_query", summary_message_count=2)
            )
        assert result == {}
        mock_get_llm.assert_not_called()


@pytest.mark.unit
class TestSummarizeNodeUpdate:
    def _mock_llm(self, summary_text: str = "Tenant asked about lease bond conditions."):
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=AIMessage(content=summary_text))
        return llm

    async def test_writes_summary_when_overflow_exists(self):
        """document_query window=4; 6 messages → overflow=2 → summary written."""
        msgs = _human_ai_pair(1) + _human_ai_pair(2) + _human_ai_pair(3)
        with _PATCH as mock_get_llm:
            mock_get_llm.return_value = self._mock_llm()
            result = await summarize_node(_make_state(msgs, intent="document_query"))
        assert result["conversation_summary"] == "Tenant asked about lease bond conditions."
        assert result["summary_message_count"] == 2

    async def test_summary_message_count_set_to_overflow_length(self):
        # search window=6; 10 messages → overflow=4
        msgs = sum((_human_ai_pair(i) for i in range(5)), [])  # 10 messages
        with _PATCH as mock_get_llm:
            mock_get_llm.return_value = self._mock_llm("Some context.")
            result = await summarize_node(_make_state(msgs, intent="search"))
        assert result["summary_message_count"] == 4

    async def test_incremental_delta_passed_to_llm(self):
        """Only the newly overflowed turns (not already-summarised ones) are in the prompt."""
        # document_query window=4; 8 messages → overflow=4; already_summarised=2 → delta=2
        msgs = sum((_human_ai_pair(i) for i in range(4)), [])  # 8 messages
        with _PATCH as mock_get_llm:
            mock_get_llm.return_value = self._mock_llm()
            await summarize_node(
                _make_state(
                    msgs,
                    intent="document_query",
                    summary="Prior summary.",
                    summary_message_count=2,
                )
            )
            call_args = mock_get_llm.return_value.ainvoke.call_args.args[0]

        # Prompt should contain: system prompt + existing summary SystemMessage + 2 delta messages
        system_msgs = [m for m in call_args if isinstance(m, SystemMessage)]
        human_ai_msgs = [m for m in call_args if isinstance(m, (HumanMessage, AIMessage))]
        assert len(human_ai_msgs) == 2  # only the delta, not the full overflow
        assert any("Prior summary." in m.content for m in system_msgs)

    async def test_existing_summary_included_in_prompt(self):
        msgs = sum((_human_ai_pair(i) for i in range(4)), [])  # 8 msgs
        with _PATCH as mock_get_llm:
            mock_get_llm.return_value = self._mock_llm()
            await summarize_node(
                _make_state(msgs, intent="document_query", summary="Existing context.")
            )
            call_args = mock_get_llm.return_value.ainvoke.call_args.args[0]

        assert any(
            isinstance(m, SystemMessage) and "Existing context." in m.content
            for m in call_args
        )

    async def test_no_existing_summary_not_injected(self):
        msgs = sum((_human_ai_pair(i) for i in range(4)), [])
        with _PATCH as mock_get_llm:
            mock_get_llm.return_value = self._mock_llm()
            await summarize_node(_make_state(msgs, intent="document_query", summary=None))
            call_args = mock_get_llm.return_value.ainvoke.call_args.args[0]

        assert not any(
            isinstance(m, SystemMessage) and "Existing" in m.content
            for m in call_args
        )


@pytest.mark.unit
class TestSummarizeNodeEviction:
    def _make_state_with_property(
        self,
        messages: list,
        summary: str,
        summary_property_id: str | None,
        current_property_id: str | None,
        summary_message_count: int = 0,
    ) -> dict:
        return {
            "messages": messages,
            "user_intent": "document_query",
            "conversation_summary": summary,
            "summary_message_count": summary_message_count,
            "summary_property_id": summary_property_id,
            "property_context": {"property_id": current_property_id} if current_property_id else {},
        }

    async def test_evicts_summary_on_property_change(self):
        """When property_id changes, existing summary is discarded and rebuilt from scratch."""
        msgs = sum((_human_ai_pair(i) for i in range(4)), [])  # 8 msgs, overflow=4
        with _PATCH as mock_get_llm:
            llm = MagicMock()
            llm.ainvoke = AsyncMock(return_value=AIMessage(content="Fresh summary."))
            mock_get_llm.return_value = llm

            result = await summarize_node(
                self._make_state_with_property(
                    msgs,
                    summary="Old summary about property A lease.",
                    summary_property_id="prop-A",
                    current_property_id="prop-B",
                )
            )

        # Old summary must not appear in the LLM prompt
        call_args = llm.ainvoke.call_args.args[0]
        assert not any(
            isinstance(m, SystemMessage) and "Old summary about property A" in m.content
            for m in call_args
        )
        assert result["conversation_summary"] == "Fresh summary."
        assert result[StateKeys.SUMMARY_PROPERTY_ID] == "prop-B"

    async def test_no_eviction_when_property_unchanged(self):
        """Same property — existing summary is retained and extended incrementally."""
        # 8 msgs total, already_summarised=2 → delta=2
        msgs = sum((_human_ai_pair(i) for i in range(4)), [])
        with _PATCH as mock_get_llm:
            llm = MagicMock()
            llm.ainvoke = AsyncMock(return_value=AIMessage(content="Extended summary."))
            mock_get_llm.return_value = llm

            await summarize_node(
                self._make_state_with_property(
                    msgs,
                    summary="Existing summary about property A.",
                    summary_property_id="prop-A",
                    current_property_id="prop-A",
                    summary_message_count=2,
                )
            )

        call_args = llm.ainvoke.call_args.args[0]
        assert any(
            isinstance(m, SystemMessage) and "Existing summary about property A." in m.content
            for m in call_args
        )

    async def test_no_eviction_when_summary_property_is_none(self):
        """Summary built without a property focus remains valid regardless of current property."""
        msgs = sum((_human_ai_pair(i) for i in range(4)), [])
        with _PATCH as mock_get_llm:
            llm = MagicMock()
            llm.ainvoke = AsyncMock(return_value=AIMessage(content="General summary retained."))
            mock_get_llm.return_value = llm

            await summarize_node(
                self._make_state_with_property(
                    msgs,
                    summary="General context summary.",
                    summary_property_id=None,
                    current_property_id="prop-B",
                )
            )

        call_args = llm.ainvoke.call_args.args[0]
        assert any(
            isinstance(m, SystemMessage) and "General context summary." in m.content
            for m in call_args
        )

    async def test_summary_property_id_written_on_update(self):
        """The property_id in context at summary time is persisted to state."""
        msgs = sum((_human_ai_pair(i) for i in range(4)), [])
        with _PATCH as mock_get_llm:
            llm = MagicMock()
            llm.ainvoke = AsyncMock(return_value=AIMessage(content="Summary."))
            mock_get_llm.return_value = llm

            result = await summarize_node(
                self._make_state_with_property(
                    msgs,
                    summary=None,
                    summary_property_id=None,
                    current_property_id="prop-123",
                )
            )

        assert result[StateKeys.SUMMARY_PROPERTY_ID] == "prop-123"


@pytest.mark.unit
class TestSummarizeNodeFailure:
    async def test_llm_failure_returns_empty_dict(self):
        """State must never be corrupted when the LLM call fails."""
        msgs = sum((_human_ai_pair(i) for i in range(4)), [])
        with _PATCH as mock_get_llm:
            llm = MagicMock()
            llm.ainvoke = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
            mock_get_llm.return_value = llm
            result = await summarize_node(_make_state(msgs, intent="document_query"))
        assert result == {}
