# Implementation Plan: Fingerprint — `/classify` and `/summarize` (rec-brain)

## Context

The Fingerprint log-triage system spans three codebases: a .NET backend (App Insights ingest, Postgres, GitHub issue filing, ownership routing), a React dashboard, and this Python service (rec-brain). Only rec-brain's code lives in this repo.

**rec-brain's scope is two stateless internal endpoints**: `POST /classify` (LLM-fallback categorization for a log fingerprint the .NET rule classifier couldn't handle) and `POST /summarize` (writes the title/body for a GitHub issue from a fingerprint + its recent occurrences). Everything else — Postgres schema, App Insights polling, ownership map, GitHub API calls — lives in .NET and is out of scope here.

Confirmed decisions:
- **Auth**: reuse the existing `verify_internal_key` dependency (checks `X-API-Key` against `settings.BACKEND_API_KEY`) — no new secret. Matches the .NET side's plan exactly: `FingerprintAiService` calls both endpoints with `X-API-Key: <AiServiceSettings.ApiKey>`, same host/key as every other `AiService` call.
- **No LangGraph graph** — every single-shot LLM call in this codebase (`DocumentTypeClassifier`, `SearchService.get_suburb_summary`, intent's LLM fallback) uses plain `get_llm().with_structured_output(...).ainvoke(...)`. `StateGraph` is reserved for the one multi-turn conversational agent. Using it here would add checkpointing complexity these stateless calls don't need.
- **One service class**, not a "module-as-singleton" (the spec's stated Python convention doesn't match this codebase — see Notes below).
- **LLM failures → 5xx, not a fallback 200** (departs from `DocumentTypeClassifier`'s "fall back to `invoice`" precedent — see Phase 4 rationale). This applies to failure to produce `title`/`body` — a `null` `suggested_fix` is a normal, successful response, not a failure (see next bullet).
- **`suggested_fix` gating is entirely .NET-side** — the updated .NET plan adds an optional `suggested_fix` string to `/summarize`'s response. rec-brain always returns whatever the LLM produces (or `null` when it has too little context, e.g. `NEW_REGRESSION`); `GitHubIssueService` decides whether to actually surface it on the issue based on `fp.AutoFixEligible` (an allow/deny-list computation rec-brain has no visibility into and shouldn't try to replicate).

---

## Phase 1 — Schemas (`app/schemas/fingerprint.py`, new)

Define the request/response contract both endpoints share:

- `FingerprintCategory = Literal["DEPENDENCY_FAILURE", "NEW_REGRESSION", "RECURRING_KNOWN", "CONFIG_AUTH", "DATA_QUALITY", "PERFORMANCE"]` — colocated here (not `app/core/constants.py`, which is almost entirely graph/agent-internal), same precedent as `DocType = Literal["invoice", "receipt"]` in `app/infrastructure/document_classifier.py`.
- `ClassifyRequest` — `exception_type`, `message_template`, `sample_trace`, `operation` (all `str`).
- `ClassifyResponse` — `category: FingerprintCategory`, `confidence: float` (bounded 0–1 via `Field(ge=0.0, le=1.0)`), `rationale: str`.
- `FingerprintOccurrence` — `occurred_at: datetime`, `occurrence_count: int`, `rendered_message: str`.
- `FingerprintRow` — `id`, `level`, `exception_type`, `message_template`, `operation`, `service_name`, `category: FingerprintCategory`, `first_seen`, `last_seen`, `total_count`, `sample_trace: str | None = None`.
- `SummarizeRequest` — `fingerprint: FingerprintRow`, `occurrences: list[FingerprintOccurrence]` (.NET now caps this at the 5 most recent occurrences before sending — no server-side count truncation needed here; per-item `rendered_message` truncation via `FingerprintConfig` still applies).
- `SummarizeResponse` — `title: str`, `body: str` (markdown), `suggested_fix: str | None` — a human-reviewed starting point for the assignee, not an autonomous edit; `None` when the LLM has too little context (e.g. `NEW_REGRESSION`). New in this revision — .NET's `AiFingerprintSummarizeResponse` now carries this field; whether it's actually surfaced on the issue is entirely .NET's call (`fp.AutoFixEligible`), rec-brain just returns the value.

Notes:
- `level` / `service_name` are plain `str`, not `Literal` — those vocabularies are owned by .NET/ingest; this service shouldn't reject a payload over an unrecognized value there. (The .NET plan's example sends lowercase `"level": "error"` — irrelevant here since it's untyped.)
- `FingerprintRow.category` **is** the strict `FingerprintCategory` Literal — `/summarize` only runs after a category already exists, so a mismatch means taxonomy drift between the two codebases and should hard-fail (422). The .NET plan's 6-value taxonomy matches this Literal exactly, confirmed in its "Python HTTP contract" section.
- `id` typed as `str` — **resolved** (was open question #3): the .NET plan commits to `fingerprints.id` as a `TEXT` business key (`fp_<hash prefix>`, e.g. `fp_a1b2c3d4`), never a numeric/Guid surrogate. `str` is correct as planned.
- `sample_trace` made **optional**, not required — the .NET plan's `/summarize` request example omits it from the `fingerprint` object entirely (unlike `/classify`'s request, which does include it), and `AiFingerprintSummarizeRequest.cs`'s exact shape isn't spelled out. Defaulting to `None` avoids a spurious 422 if .NET's DTO genuinely never sends it on this endpoint — confirm with whoever owns `AiFingerprintSummarizeRequest.cs` (flagged in Open questions below).

No dependencies on other new code — this can be written and reviewed standalone.

---

## Phase 2 — Constants (`app/core/constants.py`, modify)

Add two small, additive pieces:

- `PromptLabels.FINGERPRINT_CLASSIFY_INPUT = "[FINGERPRINT LOG DATA]"` and `PromptLabels.FINGERPRINT_SUMMARY_INPUT = "[FINGERPRINT SUMMARY DATA]"` — injection-boundary markers, same convention as `PromptLabels.DOCUMENT_TYPE_CLASSIFIER`.
- New `FingerprintConfig` class — `TRACE_TEXT_LIMIT = 2000`, `RENDERED_MESSAGE_LIMIT = 500` — caps on raw log text before it's interpolated into a prompt. Mirrors `DocumentClassifierConfig.LLM_TEXT_LIMIT`. Matters because `sample_trace`/`rendered_message` are externally-supplied, untrusted content — a crafted exception message is a plausible prompt-injection vector, same threat model as RAG-retrieved docs.

Small, isolated diff — easy to review on its own.

---

## Phase 3 — Prompts (new files)

- `app/prompts/fingerprint_classify.py` — `FINGERPRINT_CLASSIFY_PROMPT`: role framing ("fallback classifier — a rule-based pass already failed"), the 6 categories with a one-line definition each, instruction to return a genuine (not-always-1.0) confidence + 1–2 sentence rationale. Explicitly instructs the model not to guess `RECURRING_KNOWN` from a single sample with no frequency signal. Kept as a defensive guardrail even though open question #2 is now resolved: `FingerprintRuleClassifier` does its own baseline-spike detection (`GetHourlyBaselineAsync` + 3x multiplier) and short-circuits `NEW_REGRESSION` on `isNewFingerprint == true` **before** ever calling `/classify` — so in practice this endpoint only sees the other four categories, but the schema still allows all six and the instruction costs nothing to keep.
- `app/prompts/issue_summary.py` (**not** `summarize.py`, which already exists for the conversational agent's conversation-summary node) — `FINGERPRINT_ISSUE_SUMMARY_PROMPT`: role framing ("writing a GitHub issue for an engineer triaging a production error"), required body structure (summary, first/last seen, occurrence count, affected operation, suggested first debugging step), short-title instruction, explicit "don't invent details not in the payload" rule (matches `app/prompts/rag.py`'s no-hallucination style). Also instructs the model to populate `suggested_fix` as a short (1–3 sentence), clearly-hedged starting point ("a starting point, not an authoritative fix") when the payload gives it enough to work with, and to return `null` rather than invent something when it doesn't — the model only ever sees fingerprint metadata + up to 5 rendered messages, never source code, so it's expected to be weakest for `NEW_REGRESSION`, where there's no established pattern yet.

Both follow the existing `SystemMessage(PROMPT + PromptLabels.X)` / `HumanMessage(untrusted content)` split from `document_classifier.py` — static instructions in the system message, 100% of caller-supplied payload in the human message.

---

## Phase 4 — Service (`app/services/fingerprint_service.py`, new)

`FingerprintServiceError(Exception)` — plain exception, same shape as `InvoiceExtractionError` (not part of the `AIServiceError` hierarchy; caught locally at the route, matching `documents.py`'s pattern).

`FingerprintTriageService(llm: BaseChatModel)`:
- `classify(req: ClassifyRequest) -> ClassifyResponse` — `[SystemMessage, HumanMessage]` → `get_llm().with_structured_output(ClassifyResponse)` (targets the response schema directly, same shape as `SearchService.get_suburb_summary`) → wraps failures in `FingerprintServiceError`.
- `summarize_for_issue(req: SummarizeRequest) -> SummarizeResponse` — same shape, `with_structured_output(SummarizeResponse)`. `suggested_fix` coming back as `None` is a normal successful result, not something the service treats as a failure.
- Private `_build_classify_input` / `_build_summary_input` — pure string-formatting helpers (truncating via `FingerprintConfig`), testable without mocking the LLM.
- structlog events: `fingerprint_classify_start/complete/failed`, `fingerprint_summarize_start/complete/failed`. **Never log `sample_trace`, `message_template`, or `rendered_message` content** — only lengths/ids/category — matching `document_classifier.py`'s `text_length=len(text)` convention, since raw exception messages can carry secrets or PII.

**Why failures raise instead of returning a safe fallback (departure from `DocumentTypeClassifier`'s "default to invoice" precedent):**
- `/classify` only ever sees the tail case .NET's rules already couldn't handle — a silently-wrong category isn't a rare edge case here, it's the entire population this endpoint sees.
- `/summarize`'s output gets baked into a permanent, publicly-visible GitHub issue — there's no safe synthetic fallback worth generating.
- A 5xx lets .NET's own retry/idempotency logic decide what to do, which is its call to make, not something rec-brain should paper over.
- A genuinely low-confidence-but-valid LLM answer is **not** a failure — `confidence` is exactly the field for expressing that, and it should pass through untouched. Same logic extends to `suggested_fix: None` — a deliberate "nothing useful to add" is a valid answer, not an error.

---

## Phase 5 — Wiring (`app/api/dependencies.py`, `app/api/routes/fingerprint.py`, `main.py`)

- `app/api/dependencies.py` — add `get_fingerprint_service(request: Request) -> FingerprintTriageService: return request.app.state.fingerprint_triage_service`.
- `app/api/routes/fingerprint.py` (new) — `APIRouter(prefix="/api/fingerprint", tags=["fingerprint"], dependencies=[Depends(verify_internal_key)])` (auth at router level, since both routes are internal-only). Two `POST` handlers, each: call the service, catch `FingerprintServiceError` → `HTTPException(502, ...)`, catch bare `Exception` → `HTTPException(500, ...)` + `logger.exception(...)`.
- `main.py` — import the route module + `FingerprintTriageService`; in `lifespan()`, alongside the other `get_llm()`-backed services: `_app.state.fingerprint_triage_service = FingerprintTriageService(llm=get_llm())`; `app.include_router(fingerprint.router)`. No new settings needed.

**Route path:** `/api/fingerprint/classify` + `/api/fingerprint/summarize`, for consistency with every other router in this repo (`/api/search`, `/api/documents`) — confirmed non-blocking, since .NET's `AiServiceSettings.Classify`/`Summarize` are config-driven path properties, not hardcoded (see open question #1, resolved).

---

## Phase 6 — Tests

- `tests/unit/test_fingerprint_service.py` — `classify()`/`summarize_for_issue()` happy paths, LLM-failure → `FingerprintServiceError` (no fallback value), truncation behavior on `_build_classify_input`/`_build_summary_input` tested directly (no LLM mock needed). Follows `test_invoice_service.py`/`test_search_service.py`'s `AsyncMock` + `with_structured_output` mocking style.
- `tests/unit/test_fingerprint_route.py` — this repo currently has **no HTTP-level test anywhere** exercising `verify_internal_key`. Since these two endpoints are the most security-sensitive surface in the repo, build a minimal standalone `FastAPI()` app (just this router + `dependency_overrides`) rather than importing the real `main.app` — importing `main.app` and using `TestClient`/`LifespanManager` would trigger the real lifespan (Postgres, Azure DI, .NET backend client), which isn't appropriate for a unit test. Covers: missing header → 422 (FastAPI's own required-header validation), wrong key → 403, service raises → 502, happy path → 200.
- `tests/integration/test_api.py` (optional) — one real end-to-end call per endpoint behind the existing `skip_if_no_env` gate.

---

## Open questions for the .NET/spec owner (flag back, don't guess)

1. ~~**Literal route path**~~ — **resolved.** The updated .NET plan doesn't hardcode `/classify`/`/summarize`; `AiServiceSettings` gets two new bindable path properties (`Classify`, `Summarize`), config-driven like everything else in `FingerprintAiService`. Proceeding with `/api/fingerprint/classify` + `/api/fingerprint/summarize` per this repo's existing router convention — .NET's `appsettings.json` (their Phase 7) just needs to be set to match.
2. ~~**`RECURRING_KNOWN` has no supporting signal in `/classify`'s request contract**~~ — **resolved, differently than recommended.** Rather than extending the request contract, `FingerprintRuleClassifier` now does its own baseline-spike detection (`GetHourlyBaselineAsync` + 3x multiplier) and a `NEW_REGRESSION` short-circuit, both **before** `/classify` is ever called — so `/classify` only reaches the LLM once no .NET rule (including the frequency-based ones) has already matched. No Python-side change needed; the prompt's existing "don't guess `RECURRING_KNOWN` from one sample" instruction stays as a defensive guardrail.
3. ~~**`FingerprintRow.id` type**~~ — **resolved.** Confirmed `str` — the .NET plan commits to `fingerprints.id` as a `TEXT` business key (`fp_<hash prefix>`), not a Guid/numeric surrogate.
4. **Cross-repo enum coupling has no compile-time guard** — still true, but the taxonomy values themselves are now confirmed identical: the .NET plan's "Python HTTP contract" section and this repo's `FingerprintCategory` Literal both list the same six values. Risk is narrower (values match today) but the lack of a compile-time guard against future drift is unchanged.
5. **`sample_trace` presence on `/summarize`** — the .NET plan's example `/summarize` request omits `sample_trace` from the `fingerprint` object (present on `/classify`'s request), and `AiFingerprintSummarizeRequest.cs`'s exact field list isn't spelled out in the plan. Made `FingerprintRow.sample_trace` optional (`str | None = None`) defensively rather than guessing either way — confirm with whoever owns that DTO whether it's ever actually sent here.

## Other notes on the spec (non-blocking)

- The spec's "Python: ... module-as-singleton" line doesn't match this codebase — services here are constructor-injected classes instantiated once in `main.py`'s `lifespan()`, attached to `app.state`.
- "small LangGraph or plain calls" — going with plain calls only; no existing precedent in this codebase wraps a single-shot classification/summarization call in a graph.
- Section 7's noise threshold and Section 6's auto-fix eligibility allowlist are entirely .NET-side logic and don't touch this repo.

## Verification

1. `pytest -m unit tests/unit/test_fingerprint_service.py tests/unit/test_fingerprint_route.py -v`
2. `pytest -m unit` — full suite stays green.
3. Manual smoke test via `uvicorn main:app --reload` + `curl` against both endpoints with the `X-API-Key` header, confirming 200 on valid calls and 403 on a bad key.
4. Check `/docs` — both routes appear under the `fingerprint` tag with `ApiKeyAuth` applied.
