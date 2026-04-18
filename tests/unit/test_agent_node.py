"""
Unit tests for agent_node and its _needs_tools helper.
"""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from app.agents.nodes.agent import agent_node, _needs_tools, _trim_history


class TestNeedsTools:
    def test_booking_with_last_human_message(self):
        state = {
            "messages": [HumanMessage(content="book an inspection")],
            "user_intent": "booking",
        }
        assert _needs_tools(state) is True

    def test_cancellation_with_last_human_message(self):
        state = {
            "messages": [HumanMessage(content="cancel my inspection")],
            "user_intent": "cancellation",
        }
        assert _needs_tools(state) is True

    def test_general_intent_never_needs_tools(self):
        state = {
            "messages": [HumanMessage(content="what are your hours?")],
            "user_intent": "general",
        }
        assert _needs_tools(state) is False

    def test_search_intent_never_needs_tools(self):
        state = {
            "messages": [HumanMessage(content="show me houses")],
            "user_intent": "search",
        }
        assert _needs_tools(state) is False

    def test_booking_but_last_message_is_ai(self):
        """Second pass after tool call — last message is not HumanMessage → no tools."""
        state = {
            "messages": [
                HumanMessage(content="book"),
                AIMessage(content="Checking availability…"),
            ],
            "user_intent": "booking",
        }
        assert _needs_tools(state) is False

    def test_booking_but_last_message_is_tool_message(self):
        state = {
            "messages": [
                HumanMessage(content="book"),
                AIMessage(content="", tool_calls=[
                          {"name": "check_availability", "args": {}, "id": "tc_1", "type": "tool_call"}]),
                ToolMessage(content="{}", name="check_availability",
                            tool_call_id="tc_1"),
            ],
            "user_intent": "booking",
        }
        assert _needs_tools(state) is False

    def test_empty_messages_returns_false(self):
        state = {"messages": [], "user_intent": "booking"}
        assert _needs_tools(state) is False


class TestTrimHistory:
    """Search results no longer live in messages — _trim_history is now a pass-through."""

    def test_empty_list(self):
        assert _trim_history([]) == []

    def test_returns_messages_unchanged(self):
        msgs = [HumanMessage(content="a"), AIMessage(content="b")]
        assert _trim_history(msgs) == msgs

    def test_system_messages_not_stripped(self):
        """SystemMessages are no longer written to state["messages"], so no stripping needed."""
        msgs = [
            HumanMessage(content="show me flats"),
            SystemMessage(content="You are a real estate agent."),
            AIMessage(content="Here are some flats."),
        ]
        assert _trim_history(msgs) == msgs


@pytest.fixture
def mock_llm():
    """Mock LLM — ainvoke returns a plain AI reply; bind_tools returns itself."""
    llm = MagicMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="Here are the results.")
    )
    llm.bind_tools = MagicMock(return_value=llm)
    return llm


@pytest.fixture
def base_state():
    return {
        "messages": [HumanMessage(content="hello")],
        "user_intent": "general",
        "error_count": 0
    }


@patch("app.agents.nodes.agent.get_llm")
class TestAgentNode:
    async def test_returns_messages_list(self, mock_get_llm, mock_llm, base_state):
        mock_get_llm.return_value = mock_llm
        result = await agent_node(base_state)

        assert "messages" in result
        assert len(result["messages"]) == 1

    async def test_response_is_ai_message(self, mock_get_llm, mock_llm, base_state):
        mock_get_llm.return_value = mock_llm
        result = await agent_node(base_state)

        assert isinstance(result["messages"][0], AIMessage)

    async def test_general_intent_does_not_bind_tools(self, mock_get_llm, mock_llm, base_state):
        mock_get_llm.return_value = mock_llm
        await agent_node(base_state)

        mock_llm.bind_tools.assert_not_called()

    async def test_booking_intent_binds_tools(self, mock_get_llm, mock_llm):
        mock_get_llm.return_value = mock_llm
        state = {
            "messages": [HumanMessage(content="book an inspection")],
            "user_intent": "booking",
            "error_count": 0,
        }

        await agent_node(state)

        mock_llm.bind_tools.assert_called_once()

    async def test_cancellation_intent_binds_tools(self, mock_get_llm, mock_llm):
        mock_get_llm.return_value = mock_llm
        state = {
            "messages": [HumanMessage(content="cancel my booking")],
            "user_intent": "cancellation",
            "error_count": 0,
        }

        await agent_node(state)
        mock_llm.bind_tools.assert_called_once()

    async def test_llm_invoked_once(self, mock_get_llm, mock_llm, base_state):
        mock_get_llm.return_value = mock_llm
        await agent_node(base_state)

        mock_llm.ainvoke.assert_called_once()

    async def test_general_intent_caps_history_at_4(self, mock_get_llm, mock_llm):
        """general intent uses depth 4, not the old flat 10."""
        mock_get_llm.return_value = mock_llm
        long_history = [HumanMessage(content=f"msg {i}") for i in range(10)]
        state = {
            "messages": long_history,
            "user_intent": "general",
            "error_count": 0,
        }
        await agent_node(state)
        call_messages = mock_llm.ainvoke.call_args.args[0]
        # 1 SystemMessage + 4 history
        assert len(call_messages) == 5

    async def test_booking_intent_caps_history_at_10(self, mock_get_llm, mock_llm):
        """booking intent retains up to 10 messages for multi-turn contact collection."""
        mock_get_llm.return_value = mock_llm
        long_history = [HumanMessage(content=f"msg {i}") for i in range(15)]
        state = {
            "messages": long_history,
            "user_intent": "booking",
            "error_count": 0,
        }
        await agent_node(state)
        call_messages = mock_llm.ainvoke.call_args.args[0]
        # 1 SystemMessage + 10 history
        assert len(call_messages) == 11

    async def test_retrieved_docs_injected_into_prompt(self, mock_get_llm, mock_llm):
        """retrieved_docs is appended as a SystemMessage at the end of the prompt."""
        mock_get_llm.return_value = mock_llm
        state = {
            "messages": [HumanMessage(content="show me houses")],
            "user_intent": "search",
            "error_count": 0,
            "retrieved_docs": "[PROPERTY SEARCH RESULTS — 2 listing(s) found.]\n1. 12 Park St...",
        }
        await agent_node(state)
        call_messages = mock_llm.ainvoke.call_args.args[0]
        last_msg = call_messages[-1]
        assert isinstance(last_msg, SystemMessage)
        assert "[PROPERTY SEARCH RESULTS" in last_msg.content

    async def test_retrieved_docs_cleared_in_return(self, mock_get_llm, mock_llm):
        """agent_node always returns retrieved_docs=None to clear it for next turn."""
        mock_get_llm.return_value = mock_llm
        state = {
            "messages": [HumanMessage(content="show me houses")],
            "user_intent": "search",
            "error_count": 0,
            "retrieved_docs": "[PROPERTY SEARCH RESULTS — 2 listing(s) found.]\n...",
        }
        result = await agent_node(state)
        assert result["retrieved_docs"] is None

    async def test_no_retrieved_docs_prompt_has_no_extra_system_message(self, mock_get_llm, mock_llm, base_state):
        """When retrieved_docs is absent, the prompt is just system + history."""
        mock_get_llm.return_value = mock_llm
        await agent_node(base_state)
        call_messages = mock_llm.ainvoke.call_args.args[0]
        # Only the REAL_ESTATE_AGENT_SYSTEM prompt + 1 HumanMessage
        assert len(call_messages) == 2
