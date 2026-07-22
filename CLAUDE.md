# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**rec-brain** is a FastAPI-based AI orchestration service for a real estate platform. It uses LangGraph for multi-turn agentic workflows, Groq (llama-3.3-70b-versatile) as the LLM, PostgreSQL with pgvector for RAG, and ChromaDB as a secondary vector store. It connects to an external .NET backend API for property data and booking operations.

## Commands

```bash
# Setup
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.mock .env               # then fill in GROQ_API_KEY, POSTGRES_URL, BACKEND_BASE_URL

# Run (development)
uvicorn main:app --reload

# Run (production)
uvicorn main:app --host 0.0.0.0 --port 8000

# Run all tests
pytest

# Run a single test file
pytest tests/test_intent.py -v
pytest tests/test_vector_search.py -v
```

API docs available at `/docs` (Swagger) and `/redoc` — disabled in production.

## Architecture

### Layers

| Layer | Location | Responsibility |
|---|---|---|
| API | `app/api/routes/` | FastAPI endpoints — `/api/chat` (sync + SSE), `/api/documents/ingest`, `/api/documents/invoice-extract`, `/api/search/*`, `/api/enquiry/draft` (sync + SSE), `/api/ai/health` |
| Agent/Graph | `app/agents/` | LangGraph state machine, nodes, routing logic — powers `/api/chat` only |
| Services | `app/services/` | Business logic: SQL search, RAG retrieval, booking, deposit lookup, enquiry drafting, .NET backend client, document ingestion, invoice extraction |
| Infrastructure | `app/infrastructure/` | DB pool, LLM init, embedding model, vector stores, Azure DI parsers, document type classifier |
| Tools | `app/tools/` | LangGraph tool callables: `check_availability`, `book_inspection`, `cancel_inspection`, `get_booking`, `get_deposit` |
| Core | `app/core/` | Config (pydantic-settings), logging (structlog), exceptions, constants |

### Agent Graph Flow

```
START
  ↓
intent_node  (keyword fast-path → LLM classifier for ambiguous cases)
  ├── "search"               → listing_search_node   ──┐
  ├── "search_then_book"     → listing_search_node   ──┤
  ├── "search_then_deposit"  → listing_search_node   ──┤
  ├── "document_query"       → vector_search_node    ──┤
  ├── "hybrid_search"        → hybrid_search_node    ──┼→ agent_node → [summarize_node] → END
  ├── "suburb_summary"       → suburb_summary_node   ──┘
  ├── "booking" / "cancellation" / "booking_lookup"       │
  │   / "deposit_payment"                                 │
  │     → agent_node (tool-calling mode)                  │
  │           ↓                                           │
  │       tools_node                                      │
  │           ↓                                           │
  │       context_update_node                             │
  │           ↓                                           │
  │       safety_node                                     │
  │           ↓                                           │
  │       agent_node (format response) ────────────────────┘
  ├── "general"          → agent_node → [summarize_node] → END
  └── compound intent → early_response → END (user must clarify)
```

- **intent_node** uses keyword matching as a fast path; falls through to an LLM classifier (`with_structured_output`) for ambiguous or compound messages.
- **intent_node** sets `user_intent` and optionally `early_response` on state to short-circuit the graph.
- Search nodes populate `search_results`; vector nodes populate `retrieved_docs` — `agent_node` formats both into the reply.
- **suburb_summary_node** calls `SearchService.get_suburb_summary()` directly (no `agent_node` formatting pass) and appends the result as an `AIMessage` so follow-ups keep conversation context.
- **summarize_node** runs after `agent_node` only for intents in `IntentConfig.SUMMARY_INTENTS` (search, hybrid_search, document_query, suburb_summary, general — booking/cancellation/deposit flows skip it). It incrementally rolls turns that fell outside the active history window into `conversation_summary`, so long conversations don't blow the token budget; the summary is evicted when the user switches to a different property.
- `error_count` is incremented on tool failures; reaching `MAX_ERRORS_BEFORE_ESCALATION` sets `requires_human = True`.
- `listing_search_node` / `vector_search_node` set `node_error = "db_unavailable"` instead of raising on DB failure; `agent_node` checks it first and short-circuits with a fixed apology (`Messages.SEARCH_ERROR`), then clears it.

### State (`app/agents/state.py`)

`RealEstateAgentState` is the LangGraph state type persisted via `PostgresSaver` across turns:

- `messages` — conversation history (append-only via `operator.add`)
- `user_intent`, `early_response`
- `property_context`, `booking_context`, `booking_status`
- `search_context`, `search_results`, `retrieved_docs`
- `deposit_result` — populated by `get_deposit` tool; forwarded to frontend via SSE `result` event
- `suburb_summary_result` — populated by `suburb_summary_node`
- `conversation_summary`, `summary_message_count`, `summary_property_id` — rolling summary maintained by `summarize_node` for turns outside the active history window
- `intent_completed` — True after a tool flow finishes; tells the LLM classifier to treat the next message as a fresh request
- `last_intent` — intent from the just-completed flow; used by continuation checks and the LLM classifier hint
- `error_count`, `requires_human`
- `node_error` — set by search/vector nodes on DB failure (e.g. `"db_unavailable"`); read once by `agent_node` to short-circuit with a friendly fallback message, then cleared

### Key Design Decisions

- **Hybrid intent classification**: `intent_node` uses a keyword fast-path (regex/keyword matching, no LLM) for high-confidence intents (booking, cancellation, deposit, slot continuation). Ambiguous or compound messages fall through to an LLM classifier (`with_structured_output`) that also extracts search entities (location, price, bedrooms, etc.).
- **LLM never writes state**: LLM output is processed by nodes that update state explicitly.
- **Tool injection via `InjectedToolArg`**: Tools receive services (DB pool, HTTP client) as injected args, not globals.
- **Typed service dependencies**: `app/api/dependencies.py` exposes typed `Depends()` factories (e.g. `get_agent`, `get_invoice_service`) that bridge the untyped `app.state` namespace so Pylance has full intelligence in routes.
- **Async-first**: All I/O (DB, HTTP, embeddings) is async (`asyncpg`, `httpx`).
- **Dual vector stores**: pgvector (via LlamaIndex) for persistent RAG; ChromaDB for in-memory/fast lookups.
- **Stateless services outside the graph**: `/api/search/*` and `/api/enquiry/*` are single-shot request/response endpoints (`SearchService`, `EnquiryService`) using direct `get_llm()` / `with_structured_output()` calls — not LangGraph nodes. `StateGraph` is reserved for `/api/chat`'s multi-turn conversational flow; these don't need checkpointing.

### API Contract

**POST `/api/chat`**
- Input: `ChatRequest` — `message`, `thread_id`, `user_id`, `is_new_conversation`
- Output: `ChatResponse` — `reply`, `thread_id`, `listings`, `property_id`, `deposit`

**POST `/api/chat/stream`** — same input, SSE token stream

**POST `/api/documents/ingest`**
- Input: multipart — `file` (PDF/TXT/DOCX, ≤ 20 MB), `property_id`, `doc_type`
- Output: `IngestResponse` — `success`, `filename`, `property_id`, `doc_type`, `chunk_count`, `message`
- Pipeline: Azure DI (`prebuilt-layout`, markdown output) → keyword classifier → LLM fallback → `MarkdownNodeParser` → `SentenceSplitter(512, overlap=50)` → embed → pgvector upsert
- `DocstoreStrategy.UPSERTS` — prevents in-session dedup from skipping re-ingested nodes after deletion
- Azure DI S0 tier required — F0 silently truncates documents to 2 pages

**POST `/api/documents/invoice-extract`**
- Input: multipart — `file` (PDF/JPEG/PNG/TIFF/BMP, ≤ 20 MB), `property_id`
- Output: `InvoiceExtractionResponse` — `tool_name`, `success`, `filename`, `property_id`, `data: InvoiceData`
- `InvoiceData`: `doc_type` (`"invoice"` | `"receipt"`), `vendor_name`, `vendor_address`, `customer_name`, `invoice_id`, `invoice_date`, `due_date`, `subtotal`, `tax`, `total`, `currency` (ISO 4217 code), `line_items`, `confidence`
- Document type is auto-classified before parsing: keyword fast-path (Azure DI `prebuilt-read`) → LLM fallback (`with_structured_output`) → defaults to `"invoice"`
- Routes to `AzureInvoiceParser` (`prebuilt-invoice`) or `AzureReceiptParser` (`prebuilt-receipt`) based on classification
- Logs `invoice_low_confidence` / `receipt_low_confidence` warning when `confidence < 0.8`

**POST `/api/search/preferences`** — structured-preference listing search (not the chat agent); builds a search query from `TenantPreference`, runs it through `SqlViewService`, returns `PreferenceSearchResponse` (message, listings, display_count, total_count, has_more)

**POST `/api/search/suburb-summary`** — RAG-backed suburb summary for one or more suburbs; returns `SuburbSummaryResponse` (empty if no suburbs / no matching docs)

**POST `/api/search/tenancy-docs`** — extracts structured tenancy details (agreement type, commencement, rent, bond, etc.) from a specific tenancy agreement via RAG + `with_structured_output`; 404 if the doc isn't found, 422 if extraction fails

**POST `/api/enquiry/draft`** — classifies a tenant/landlord enquiry (`classify_rag_intent`), retrieves relevant tenancy docs by intent, drafts an LLM reply; returns `EnquiryResponse` (draft, sources). Empty `draft` on LLM failure rather than an error response.

**POST `/api/enquiry/draft/stream`** — same pipeline as above, SSE step events (`intent_classified` → `rag_retrieval` → `llm_draft` → `compliance_check` → `result`)

**GET `/api/ai/health`** — liveness check, no auth

All endpoints except `/api/ai/health` require an `X-API-Key` header (internal service key) via the `verify_internal_key` dependency — not just the document endpoints.

## Configuration

All settings live in `app/core/config.py` (pydantic-settings), sourced from `.env`:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key |
| `OPENAI_MODEL_NAME` | Default: `gpt-4o-mini` |
| `LLM_PROVIDER` | `openai` (default) \| `groq` |
| `LLM_TEMPERATURE` | Default: `0.0` |
| `POSTGRES_URL` | pgvector-enabled PostgreSQL connection string |
| `SIMILARITY_THRESHOLD` | RAG retrieval cutoff — default `0.35` (lower for HTML-heavy docs, raise to `0.5`+ after HTML stripping) |
| `SIMILARITY_TOP_K` | Candidates fetched before cutoff — default `3` |
| `EMBEDDING_MODEL` | Default: `text-embedding-3-small` (OpenAI) |
| `EMBEDDING_DIM` | Default: `1536` (matches `text-embedding-3-small`) |
| `BACKEND_BASE_URL` | .NET backend API base URL |
| `BACKEND_API_KEY` | .NET backend API key |
| `AZURE_DOC_INTEL_ENDPOINT` | Azure Document Intelligence endpoint |
| `AZURE_DOC_INTEL_KEY` | Azure Document Intelligence API key |
| `ENVIRONMENT` | `development` \| `staging` \| `production` |
| `ALLOWED_ORIGINS` | CORS origins |

## Testing

Uses `pytest-asyncio` (`asyncio_mode = auto`). Two suites, separated by marker:

```bash
pytest -m unit          # fast, no external services — runs in ~10s
pytest -m integration   # requires live Groq API + PostgreSQL + .NET backend
pytest                  # both (integration tests auto-skip if env vars missing)
```

**Marker hygiene gap**: only a minority of files under `tests/unit/` actually carry `pytestmark = pytest.mark.unit` (currently: `test_agent_node.py`, `test_listing_search.py`, `test_pgvector_store.py`, `test_rag_retriever.py`, `test_search_service.py`, `test_vector_search.py`) — most others, including `test_invoice_service.py`, `test_document_classifier.py`, `test_invoice_parser.py`, and `test_summarize_node.py`, have no marker at all, so `pytest -m unit` currently runs a subset, not the full fast suite. Always add `pytestmark = pytest.mark.unit` at module level to new unit test files.

**No CI test gate**: `.github/workflows/deploy.yml` builds and deploys to Azure Container Apps on every push to `main` without running `pytest` anywhere — the suite is advisory only right now, not a merge/deploy gate.

### `tests/unit/` — fully mocked, no I/O

| File | Covers |
|---|---|
| `test_intent.py` | `_classify_intent` + `intent_node` (all intents, hybrid, compound, state mutations) |
| `test_router.py` | All 6 conditional edge functions — every routing branch |
| `test_tools.py` | `check_availability`, `book_inspection`, `cancel_inspection` tools |
| `test_vector_search.py` | `vector_search_node` success, guard, and error paths |
| `test_hybrid_search.py` | `hybrid_search_node` — concurrent SQL+RAG, partial failures |
| `test_listing_search.py` | `listing_search_node` — SQL search node |
| `test_agent_node.py` | `agent_node` + `_needs_tools` — LLM patched with `unittest.mock` |
| `test_safety_node.py` | `safety_node` — error counting and escalation threshold |
| `test_context_update.py` | `context_update_node` — all three tool handlers + JSON resilience |
| `test_booking_service.py` | `BookingService` with mocked `BackendClient` |
| `test_sql_service.py` | `SqlViewService._validate_sql`, `_generate_sql`, `search_listings` |
| `test_chat_route.py` | `_build_response`, `_extract_tools_used`, `_extract_sources`, `_to_sse_event` |
| `test_document_classifier.py` | `DocumentTypeClassifier` — keyword fast-path, LLM fallback, ambiguous cases |
| `test_invoice_parser.py` | `AzureInvoiceParser` + `AzureReceiptParser` field extractors including `_receipt_line_items` |
| `test_invoice_service.py` | `InvoiceExtractionService` — routing to invoice/receipt parser, classifier errors |
| `test_search_service.py` | `SearchService` — preference search, suburb summary, tenancy docs |
| `test_enquiry_service.py` | `EnquiryService` — `draft_response` + `stream_draft_response` SSE steps |
| `test_rag_intent.py` | `classify_rag_intent` — keyword fast-path + LLM fallback |
| `test_summarize_node.py` | `summarize_node` — incremental rollup, eviction on property switch |
| `test_pgvector_store.py` | `PgVectorStoreService.create_vector_store` |

### `tests/integration/` — live infrastructure required

| File | Covers |
|---|---|
| `test_api.py` | `/api/chat` and `/api/chat/stream` endpoints via `httpx.AsyncClient` |
| `test_graph.py` | Full LangGraph flows for all intent types |
| `test_sql_live.py` | `SqlViewService` against real Groq + PostgreSQL |

### Shared fixtures (`tests/conftest.py`)

All fixtures follow the **factory-fixture pattern** — inject the fixture, then call it with custom args:

```python
async def test_something(make_rag_retriever, make_config):
    rag = make_rag_retriever(nodes=[])          # empty result
    rag = make_rag_retriever(raise_error=RuntimeError("down"))  # error path
    config = make_config(rag_retriever=rag)
```

Available: `make_node`, `make_rag_retriever`, `make_sql_service`, `make_booking_service`, `make_config`, `make_state`. The `parsed(result)` helper (not a fixture) deserialises `SystemMessage` JSON from node results.

Don't redefine local factory helpers that shadow these with different signatures (`test_search_service.py` does this — local `make_node`/`make_rag`/`make_sql`/`make_llm` diverge from the shared ones above). If a new shape is needed, extend `conftest.py` instead.

### Test dependencies

Beyond `pytest` + `pytest-asyncio`, the suite uses:
- `pytest-httpx` — already present, used for future `BackendClient` HTTP-level tests
- `pytest-mock` — `mocker` fixture for patching (install via `pip install -r requirements.txt`)
- `freezegun` — installed for datetime freezing, but currently unused anywhere in `tests/unit/` — don't cite it as existing precedent without checking
- `faker` — realistic test data generation
