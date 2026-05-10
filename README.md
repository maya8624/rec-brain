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
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |

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

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows
pip install -r requirements.txt
cp .env.mock .env               # fill in the values below
```

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `OPENAI_API_KEY` | Yes | OpenAI API key |
| `POSTGRES_URL` | Yes | pgvector-enabled PostgreSQL connection string |
| `BACKEND_BASE_URL` | Yes | .NET backend API base URL |
| `BACKEND_API_KEY` | Yes | .NET backend API key |
| `OPENAI_MODEL_NAME` | No | Default: `gpt-4o-mini` |
| `LLM_PROVIDER` | No | `openai` (default) \| `groq` |
| `LLM_TEMPERATURE` | No | Default: `0.0` |
| `LLM_MAX_TOKENS` | No | Default: `2048` |
| `CHROMA_PATH` | No | ChromaDB persistence directory |
| `SIMILARITY_THRESHOLD` | No | RAG retrieval cutoff score |
| `EMBEDDING_MODEL` | No | Default: `sentence-transformers/all-MiniLM-L6-v2` |
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
