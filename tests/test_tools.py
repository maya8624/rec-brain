"""
Usage:
    python test_tools.py search_listings
    python test_tools.py search_documents
    python test_tools.py check_availability
    python test_tools.py book_inspection
    python test_tools.py cancel_inspection
"""
import asyncio
import json
import os
import sys

from app.tools.search_listings import search_listings
from app.tools.check_availability import check_availability
from app.tools.book_inspection import book_inspection
from app.tools.cancel_inspection import cancel_inspection
from app.services.sql_service import SqlAgentService
from app.infrastructure.database import get_db_wrapper
from app.infrastructure.llm import get_llm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def pretty(result):
    print(json.dumps(result, indent=2))


# ── Tests ──────────────────────────────────────────────────────────────────────

async def test_search_listings():
    sql_service = SqlAgentService(llm=get_llm(), db=get_db_wrapper())

    # db = get_db_wrapper()
    # print(db.get_usable_table_names())

    pretty(await search_listings.ainvoke({
        "question": "show 3 bedroom houses in Shepparton, Queensland under $800k",
        "sql_service": sql_service,
    }))


# async def test_search_documents():
#     from app.tools.search_documents import search_documents
#     pretty(await search_documents.ainvoke({
#         "question": "what are the break lease conditions"
#     }))


async def test_check_availability():
    pretty(await check_availability.ainvoke({
        "property_id": "prop_123",
        "preferred_date": "2026-04-12",
    }))


async def test_book_inspection():
    pretty(await book_inspection.ainvoke({
        "property_id": "prop_123",
        "datetime_slot": "2026-04-12 10:00",
        "contact_name": "John Smith",
        "contact_email": "john@email.com",
        "contact_phone": "0412 345 678",
    }))


async def test_cancel_inspection():
    pretty(await cancel_inspection.ainvoke({
        "confirmation_id": "CONF-12345",
        "reason": "Change of plans",
    }))


# ── Router ─────────────────────────────────────────────────────────────────────

TOOLS = {
    "search_listings":    (test_search_listings,    False),
    # "search_documents":   (test_search_documents,   True),
    "check_availability": (test_check_availability, True),
    "book_inspection":    (test_book_inspection,    True),
    "cancel_inspection":  (test_cancel_inspection,  True),
}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_tools.py <tool_name>")
        print(f"Available: {', '.join(TOOLS)}")
        sys.exit(1)

    tool_name = sys.argv[1]

    if tool_name not in TOOLS:
        print(f"Unknown tool: {tool_name}")
        print(f"Available: {', '.join(TOOLS)}")
        sys.exit(1)

    fn, is_async = TOOLS[tool_name]

    print(f"\n── {tool_name} ────────────────────────────────────────")

    asyncio.run(fn())
