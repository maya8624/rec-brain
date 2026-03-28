"""
End-to-end graph flow tests — runs the full LangGraph agent.

Usage:
    python scripts/test_graph.py                  # run all
    python scripts/test_graph.py search           # listing search flow
    python scripts/test_graph.py booking          # booking flow
    python scripts/test_graph.py cancellation     # cancellation flow
    python scripts/test_graph.py general          # general question
    python scripts/test_graph.py compound         # compound intent
"""
from app.infrastructure.llm import get_llm
from app.infrastructure.database import get_db
from app.services.sql_search import SqlViewService
from app.agents.graph import build_graph
from langchain_core.messages import HumanMessage
import asyncio
import json
import os
import sys
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pretty(result):
    messages = result.get("messages", [])
    last = messages[-1] if messages else None
    print(f"  intent    : {result.get('user_intent', 'unknown')}")
    print(f"  messages  : {len(messages)}")
    if last:
        content = getattr(last, "content", str(last))
        print(f"  response  : {content[:300]}")
    if result.get("early_response"):
        print(f"  early_resp: {result['early_response']}")


def print_header(name: str):
    print(f"\n── {name} {'─' * (50 - len(name))}")


def make_mock_request(booking_service=None, sql_view_service=None):
    """Build a mock FastAPI request with app.state services."""
    mock_request = AsyncMock()
    mock_request.app.state.sql_view_service = sql_view_service
    mock_request.app.state.booking_service = booking_service
    return mock_request


def make_booking_service():
    mock = AsyncMock()
    mock.get_availability.return_value = [
        {"datetime": "2026-04-12 10:00",
            "agent_name": "Jane Smith", "available": True},
        {"datetime": "2026-04-12 14:00",
            "agent_name": "Jane Smith", "available": True},
    ]
    mock.book.return_value = {
        "confirmation_id": "CONF-12345",
        "property_address": "123 Main St, Sydney NSW 2000",
        "confirmed_datetime": "2026-04-12 10:00",
        "agent_name": "Jane Smith",
        "agent_phone": "0412 345 678",
    }
    mock.cancel.return_value = {"success": True}
    return mock


def get_config(request):
    return {
        "configurable": {
            "thread_id": "test-thread-001",
            "request": request,
        }
    }


# ── Tests ──────────────────────────────────────────────────────────────────────

async def test_search_flow(graph):
    print_header("search — listing search flow")

    sql_view_service = SqlViewService(llm=get_llm(), db=get_db())
    request = make_mock_request(sql_view_service=sql_view_service)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content="Show me 3 bedroom houses in Sydney under $800k")]},
        config=get_config(request),
    )
    pretty(result)
    assert result["user_intent"] == "search", f"Expected 'search', got {result['user_intent']}"
    print("  ✓ search flow complete")


async def test_general_flow(graph):
    print_header("general — conversational question")

    request = make_mock_request()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(content="What are your office hours?")]},
        config=get_config(request),
    )
    pretty(result)
    assert result["user_intent"] == "general", f"Expected 'general', got {result['user_intent']}"
    print("  ✓ general flow complete")


async def test_booking_flow(graph):
    print_header("booking — check availability flow")

    booking_service = make_booking_service()
    request = make_mock_request(booking_service=booking_service)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content="I'd like to book an inspection for property prop_123")]},
        config=get_config(request),
    )
    pretty(result)
    assert result["user_intent"] == "booking", f"Expected 'booking', got {result['user_intent']}"
    print("  ✓ booking flow complete")


async def test_cancellation_flow(graph):
    print_header("cancellation — cancel inspection flow")

    booking_service = make_booking_service()
    request = make_mock_request(booking_service=booking_service)

    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content="I want to cancel my inspection booking")]},
        config=get_config(request),
    )
    pretty(result)
    assert result[
        "user_intent"] == "cancellation", f"Expected 'cancellation', got {result['user_intent']}"
    print("  ✓ cancellation flow complete")


async def test_compound_intent(graph):
    print_header("compound — search + booking")

    request = make_mock_request()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content="Find me houses in Sydney and book an inspection")]},
        config=get_config(request),
    )
    pretty(result)
    assert result[
        "user_intent"] == "general", f"Expected 'general' for compound, got {result['user_intent']}"
    assert result.get(
        "early_response"), "Expected early_response for compound intent"
    print("  ✓ compound intent handled correctly")


async def test_document_query_flow(graph):
    print_header("document_query — lease question")

    request = make_mock_request()

    result = await graph.ainvoke(
        {"messages": [HumanMessage(
            content="What are the break lease conditions?")]},
        config=get_config(request),
    )
    pretty(result)
    assert result[
        "user_intent"] == "document_query", f"Expected 'document_query', got {result['user_intent']}"
    print("  ✓ document_query flow complete")


# ── Router ─────────────────────────────────────────────────────────────────────

TESTS = {
    "search":         test_search_flow,
    "general":        test_general_flow,
    "booking":        test_booking_flow,
    "cancellation":   test_cancellation_flow,
    "compound":       test_compound_intent,
    "document_query": test_document_query_flow,
}


async def run_all(graph):
    passed = 0
    failed = 0

    for name, fn in TESTS.items():
        try:
            await fn(graph)
            passed += 1
        except Exception as e:
            print(f"  ✗ {name} FAILED: {e}")
            failed += 1

    print(f"\n── Results: {passed}/{len(TESTS)} passed", end="")
    if failed:
        print(f" | {failed} FAILED ✗")
    else:
        print(" ✓")


if __name__ == "__main__":
    print("Building graph...")
    graph = build_graph()
    print("Graph ready.\n")

    if len(sys.argv) < 2:
        asyncio.run(run_all(graph))
    else:
        name = sys.argv[1]
        if name not in TESTS:
            print(f"Unknown test: {name}")
            print(f"Available: {', '.join(TESTS)}")
            sys.exit(1)
        asyncio.run(TESTS[name](graph))
