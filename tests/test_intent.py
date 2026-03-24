"""
Tests for intent_node classification.

Usage:
    python scripts/test_intent.py
"""
from app.agents.nodes.intent import _classify_intent
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Test cases ─────────────────────────────────────────────────────────────────

TESTS = [
    # (message, expected_intent)

    # Search
    ("Show me 3 bedroom houses in Sydney under $800k",       "search"),
    ("Find apartments in Melbourne",                          "search"),
    ("I'm looking for a unit in Parramatta",                 "search"),
    ("List properties under $500k",                          "search"),

    # Booking
    ("I'd like to book an inspection",                        "booking"),
    ("Can I arrange a viewing for this property?",            "booking"),
    ("Is this property available for inspection?",            "booking"),
    ("When can I view this apartment?",                       "booking"),

    # Cancellation
    ("I want to cancel my inspection",                        "cancellation"),
    ("Please cancel my booking",                              "cancellation"),
    ("I no longer want to inspect this property",             "cancellation"),

    # Document query
    ("What are the break lease conditions?",                  "document_query"),
    ("Can you explain the strata report?",                    "document_query"),
    ("What does the lease say about pets?",                   "document_query"),
    ("Tell me about the bond requirements",                   "document_query"),

    # General
    ("What are your office hours?",                           "general"),
    ("Hello, how are you?",                                   "general"),
    ("How does the rental process work?",                     "general"),

    # Compound — should fall through to general
    ("Find me a house in Sydney and book an inspection",      "general"),
    ("Search for apartments and cancel my booking",           "general"),
    ("Show me properties and book a viewing",                 "general"),
]


def run_tests():
    passed = 0
    failed = 0

    print("\n── Intent Classification Tests ─────────────────────────────────\n")

    for message, expected in TESTS:
        result = _classify_intent(message)
        ok = result == expected
        status = "✓" if ok else "✗"

        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  {status} [{expected:>15}] {message[:60]}")
        if not ok:
            print(f"      └─ got: {result}")

    print(f"\n── Results: {passed}/{len(TESTS)} passed", end="")
    if failed:
        print(f" | {failed} FAILED ✗")
    else:
        print(" ✓")

    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
