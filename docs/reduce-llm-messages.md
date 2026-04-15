# Reducing LLM Message Payload

## Problem

`agent_node` sends the last 10 messages to the LLM on every turn:

```python
messages = [SystemMessage(content=REAL_ESTATE_AGENT_SYSTEM), *state["messages"][-_MAX_HISTORY:]]
```

Because `state["messages"]` is append-only and search/RAG nodes push **large JSON payloads** as `SystemMessage` into it, the rolling window quickly fills with stale search results the LLM already formatted. The result is wasted tokens every turn.

---

## Suggestions

### 1. Strip stale search-result messages from history (highest impact)

Search result injections are one-time context. Once `agent_node` has formatted them into an `AIMessage`, the raw JSON blob is useless to future turns.

**`app/agents/nodes/agent.py`**
```python
from langchain_core.messages import SystemMessage

_RESULT_PREFIXES = (
    "[PROPERTY SEARCH RESULTS",
    "[DOCUMENT SEARCH RESULTS",
    "[HYBRID SEARCH RESULTS",
)

def _trim_history(messages: list) -> list:
    """Drop stale search-result SystemMessages; keep human/AI/tool messages."""
    return [
        m for m in messages
        if not (isinstance(m, SystemMessage) and m.content.startswith(_RESULT_PREFIXES))
    ]
```

Then in `agent_node`:
```python
history = _trim_history(list(state["messages"]))[-_MAX_HISTORY:]
messages = [SystemMessage(content=REAL_ESTATE_AGENT_SYSTEM), *history]
```

The current turn's search result is always the last message in `state["messages"]`, so it is **not** stripped — only results from older turns are discarded.

---

### 2. Intent-aware history depth (quick win)

`_MAX_HISTORY = 10` is applied uniformly. Most intents don't need that many turns.

| Intent | Suggested depth | Reason |
|---|---|---|
| `booking` / `cancellation` | 10 | Multi-turn flow collects contact details |
| `search` / `hybrid_search` | 6 | Remembers accumulated search criteria |
| `document_query` / `general` | 4 | Stateless per question |

```python
_HISTORY_BY_INTENT = {
    "booking": 10,
    "cancellation": 10,
    "search": 6,
    "hybrid_search": 6,
    "document_query": 4,
    "general": 4,
}

history_limit = _HISTORY_BY_INTENT.get(state.get("user_intent", "general"), 6)
messages = [SystemMessage(content=REAL_ESTATE_AGENT_SYSTEM), *history[-history_limit:]]
```

---

### 3. Move search context out of messages into state (structural fix)

The architecture already defines a `retrieved_docs` state field but it is not in `state.py`. The intended design is to pass search context **outside** the message list so it does not pollute history.

- Search nodes write results to `state["retrieved_docs"]` (not `messages`)
- `agent_node` reads `retrieved_docs` and prepends it as a fresh `SystemMessage` only for the current call
- Message history stays clean — only `HumanMessage` / `AIMessage` / `ToolMessage` pairs

This is consistent with how `search_results` already works (stored in state, not in messages). It is the right long-term fix but requires updating `state.py` and all three search nodes.

---

### 4. Compress processed tool call pairs (booking flows only)

After `context_update_node` runs, the `AIMessage(tool_call)` + `ToolMessage(result)` pair has already been fully processed. For long booking sessions these pairs could be replaced with a compact summary `SystemMessage`:

```
"Availability checked: slots [2025-06-14 10:00, 2025-06-14 14:00]. Contact collected: name=John, email=john@example.com."
```

Only relevant if booking flows span many tool rounds.

---

## Recommended order of implementation

1. **#1 — strip stale search messages** — biggest token savings, no architectural change, implement first
2. **#2 — intent-aware depth** — one-liner change, immediate reduction for `general` and `document_query` turns
3. **#3 — `retrieved_docs` state field** — cleanest long-term design, implement when refactoring search nodes
4. **#4 — tool pair compression** — niche, defer unless booking flows become a token bottleneck
