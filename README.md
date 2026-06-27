# rec-brain

AI orchestration service for a real estate platform. Handles multi-turn conversations for property search, inspection booking, cancellation, and document queries via a LangGraph agent graph.

## Tech Stack

| Layer | Technology |
|---|---|
| API | FastAPI |
| Agent | LangGraph (PostgresSaver checkpointer) |
| LLM | OpenAI — `gpt-4o-mini` (default) |
| Vector search | pgvector (LlamaIndex) + ChromaDB |
| Database | PostgreSQL (asyncpg) |
| Backend | .NET REST API (httpx) |
| Embeddings | OpenAI `text-embedding-3-small` (1536-dim) |
| Document parsing | Azure Document Intelligence (prebuilt-invoice + prebuilt-receipt + prebuilt-layout + prebuilt-read) |
| Logging | structlog — colored console in development, JSON to rotating daily log files in all environments |

## Architecture

```
POST /api/chat  or  POST /api/chat/stream
        ↓
    intent_node   (keyword fast-path → LLM classifier)
        ↓
  ┌─────────────────────────────────────────────────────┐
  │  search / search_then_book / search_then_deposit     │→ listing_search_node ─┐
  │  document_query                                      │→ vector_search_node  ─┤
  │  hybrid_search                                       │→ hybrid_search_node  ─┼→ agent_node → END
  │  booking / cancellation / booking_lookup             │                       │
  │  deposit_payment                                     │→ agent_node           │
  │                                                      │    ↓ tools_node       │
  │                                                      │    ↓ context_update   │
  │                                                      │    ↓ safety_node      │
  │                                                      │    ↓ agent_node ──────┘
  │  general                                             │→ agent_node → END
  └─────────────────────────────────────────────────────┘
```

State is persisted across turns in PostgreSQL via LangGraph's `PostgresSaver`, keyed by `thread_id`.

## Error Handling

Exceptions are handled centrally in `app/core/error_handlers.py`, mirroring .NET's `ExceptionHandlingMiddleware` pattern.

`to_http_response()` maps exception types to structured responses via `match/case`:

| Exception | HTTP status | `name` field |
|---|---|---|
| `ToolValidationError` | 422 | `VALIDATION_ERROR` |
| `BookingServiceError` | 503 | `BOOKING_ERROR` |
| `DepositServiceError` | 503 | `DEPOSIT_ERROR` |
| `BackendClientError` | 502 | `BACKEND_ERROR` |
| `AIServiceError` | 500 | `AI_SERVICE_ERROR` |
| anything else | 500 | `UNEXPECTED_ERROR` |

All error responses follow the shape `{ name, code, message, thread_id }`.

Exceptions caught inside the LangGraph graph (nodes, tools) are handled locally for graceful degradation and do not reach this handler. Streaming SSE endpoints handle errors internally to comply with the SSE protocol.

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt   # includes python-multipart for file uploads
cp .env.mock .env                 # fill in the values below
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `POSTGRES_URL` | Yes | pgvector-enabled PostgreSQL connection string |
| `BACKEND_BASE_URL` | Yes | .NET backend API base URL |
| `BACKEND_API_KEY` | Yes | .NET backend API key |
| `AZURE_DOC_INTEL_ENDPOINT` | Yes | Azure Document Intelligence endpoint URL (S0 tier required) |
| `AZURE_DOC_INTEL_KEY` | Yes | Azure Document Intelligence API key |
| `OPENAI_MODEL_NAME` | No | Default: `gpt-4o-mini` |
| `LLM_PROVIDER` | No | `openai` (default) \| `groq` |
| `LLM_TEMPERATURE` | No | Default: `0.0` |
| `LLM_MAX_TOKENS` | No | Default: `2048` |
| `SIMILARITY_THRESHOLD` | No | RAG retrieval cutoff — default `0.35` |
| `SIMILARITY_TOP_K` | No | RAG candidates fetched before cutoff — default `3` |
| `EMBEDDING_MODEL` | No | Default: `text-embedding-3-small` |
| `EMBEDDING_DIM` | No | Default: `1536` (matches `text-embedding-3-small`) |
| `ENVIRONMENT` | No | `development` \| `staging` \| `production` |
| `ALLOWED_ORIGINS` | No | Comma-separated CORS origins |

## Running

```bash
# Development
uvicorn main:app --reload

# Production
uvicorn main:app --host 0.0.0.0 --port 8000
```

API docs: `/docs` (Swagger) and `/redoc` — disabled in production.

## API

### `POST /api/chat`

Synchronous. Returns the full agent reply once the graph completes.

**Request**
```json
{
  "message": "Show me 3-bedroom houses in Castle Hill",
  "thread_id": "abc-123",
  "user_id": "user-456",
  "is_new_conversation": true
}
```

**Response**
```json
{
  "reply": "...",
  "thread_id": "abc-123",
  "listings": [],
  "property_id": null,
  "deposit": null
}
```

- `listings` — structured property cards for the frontend to render
- `property_id` — set only when exactly one property is in context (enables check-availability / book flow)
- `deposit` — present when `get_deposit` succeeds; contains `session_url` for the Stripe popup

### `POST /api/chat/stream`

Same request body. Returns tokens via Server-Sent Events.

| Event type | Payload | Description |
|---|---|---|
| `token` | `content` | LLM token — append to buffer |
| `tool_start` | `tool` | Agent called a tool |
| `tool_end` | `tool` | Tool returned |
| `result` | `thread_id`, `listings`, `property_id`, `deposit` | Final metadata |
| `error` | `message` | Unhandled error |
| `[DONE]` | — | Stream complete |

### `POST /api/documents/ingest`

Parses, classifies, chunks, embeds, and upserts a document into pgvector. Called by an Azure Function when a file is uploaded to blob storage. Requires the internal API key (`X-Internal-Key` header).

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | PDF, TXT, or DOCX — max 20 MB |
| `property_id` | string | No | Property this document belongs to; omit for global docs |
| `doc_type` | string | No | Auto-classified if omitted |

**Response**
```json
{
  "success": true,
  "filename": "lease-agreement.pdf",
  "property_id": "prop-123",
  "doc_type": "lease",
  "chunk_count": 14,
  "message": "Ingested lease-agreement.pdf — 14 chunks stored"
}
```

### `POST /api/documents/invoice-extract`

Extracts structured fields from an invoice or receipt via Azure Document Intelligence. The document type is classified automatically before parsing. Requires the internal API key (`X-API-Key` header).

**Document type classification** — runs before parsing:
1. Keyword fast-path: uses Azure DI `prebuilt-read` to extract text, then matches against receipt/invoice keyword sets
2. LLM fallback (`with_structured_output`): used when keywords are ambiguous or absent
3. Defaults to `"invoice"` if the LLM call fails

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | PDF, JPEG, PNG, TIFF, or BMP — max 20 MB |
| `property_id` | string | No | Property this document belongs to |

**Response**
```json
{
  "tool_name": "save_invoice",
  "success": true,
  "filename": "CB-19847_invoice.pdf",
  "property_id": "prop-123",
  "data": {
    "doc_type": "invoice",
    "vendor_name": "CoolBreeze Air Conditioning",
    "vendor_address": "12 Industrial Ave, Sydney NSW 2000",
    "customer_name": "Acme Property Group",
    "invoice_id": "INV-0042",
    "invoice_date": "2026-06-01",
    "due_date": "2026-06-30",
    "subtotal": 1450.00,
    "tax": 145.00,
    "total": 1595.00,
    "currency": "AUD",
    "line_items": [
      { "description": "Split system installation", "quantity": 1, "unit_price": 1450.00, "amount": 1450.00 }
    ],
    "confidence": 0.94
  }
}
```

- `doc_type` is `"invoice"` or `"receipt"`
- `currency` is an ISO 4217 code (e.g. `"AUD"`, `"USD"`)
- `confidence` is `0.0–1.0`; a low-confidence warning is logged when below `0.8`

## Tools

| Tool | Trigger intent | Description |
|---|---|---|
| `check_availability` | `booking` | Fetches available inspection slots from .NET |
| `book_inspection` | `booking` | Confirms an inspection booking with .NET |
| `cancel_inspection` | `cancellation` | Cancels an existing booking with .NET |
| `get_booking` | `booking_lookup` | Retrieves booking details from .NET |
| `get_deposit` | `deposit_payment` | Looks up holding deposit; returns Stripe session URL |

## Deployment

Hosted on Azure Container Apps. CI/CD via GitHub Actions.

## Testing

```bash
pytest -m unit          # fast, fully mocked — no external services (~10s)
pytest -m integration   # requires live Groq + PostgreSQL + .NET backend
pytest                  # both (integration tests auto-skip if env vars missing)
```
