# TODO List

Tracked from inline `TODO` comments across the codebase.

---

## Security

- [ ] **Switch `ALLOWED_TABLES` to views**
  `app/infrastructure/database.py:20`
  Change from raw tables to read-only views for better security and to simplify
  the schema exposed to the LLM.

---

## Agent Graph

- [ ] **Add `human_escalation_node`**
  `app/agents/graph.py:47`
  `requires_human=True` currently exits via every router's `_requires_human()`
  guard with no AIMessage — the reply is patched in `_build_response`
  (`app/api/routes/chat.py`) as a formatting fallback. Promote to a dedicated
  graph node when escalation needs side effects (webhook calls, CRM
  notifications, staff alerts). All `_requires_human` guards in `router.py`
  should route to it instead of `END`.

- [ ] **Multi-step tool calling for compound intents**
  `app/agents/graph.py:56`

- [ ] **Use `search_context` entities in `listing_search_node`**
  `app/agents/graph.py:57`
  Build SQL from structured `state["search_context"]` data instead of
  re-parsing the raw message via `sql_service`.

---

## API

- [ ] **Refactor `property_id` extraction in `_build_response`**
  `app/api/routes/chat.py:226`
  The current inline walrus-operator expression is hard to read — extract into
  a helper.

---

## Infrastructure

- [ ] **Verify `RequestLoggingMiddleware`**
  `main.py:134`
  Double-check that the middleware correctly propagates `X-Request-ID` for
  tracing across .NET → Python.

---

## Resolved

- [x] **Pre-format slot times to Sydney time in Python**
  `app/agents/nodes/context.py:62`
  Resolved: `check_availability` tool now converts `start_at` / `end_at` from
  UTC to AEST/AEDT using `zoneinfo` before returning results to the LLM.
