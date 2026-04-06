"""
Unit tests for agent_node and its _needs_tools helper.
"""
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents.nodes.agent import agent_node, _needs_tools


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

    async def test_only_last_10_messages_sent_to_llm(self, mock_get_llm, mock_llm):
        """agent_node trims to _MAX_HISTORY (10) messages."""
        mock_get_llm.return_value = mock_llm
        long_history = [HumanMessage(content=f"msg {i}") for i in range(15)]
        state = {
            "messages": long_history,
            "user_intent": "general",
            "error_count": 0
        }
        await agent_node(state)
        call_messages = mock_llm.ainvoke.call_args.args[0]
        # First is SystemMessage, then up to 10 history messages
        assert len(call_messages) <= 11  # 1 system + 10 history max
