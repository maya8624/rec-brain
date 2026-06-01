import json
import logging
import time

from llama_index.core.schema import NodeWithScore
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._base import extract_sources
from app.agents.nodes.rag_intent import classify_rag_intent
from app.core.config import settings
from app.infrastructure.llm import get_llm
from app.prompts.enquiry import ENQUIRY_DRAFT_PROMPT, ENQUIRY_NO_DOCS_PROMPT
from app.schemas.enquiry import EnquiryRequest, EnquiryResponse
from app.schemas.rag import INTENT_COMPLIANCE_RULES, INTENT_DOC_TYPES
from app.services.rag_service import RagRetriever

logger = logging.getLogger(__name__)


def _sse(type_: str, **kwargs) -> str:
    return f"data: {json.dumps({'type': type_, **kwargs})}\n\n"


def _fmt_elapsed(seconds: float) -> str:
    """Format elapsed time: under 1 s → '94ms', 1 s and above → '1.4s'."""
    if seconds < 1.0:
        return f"{round(seconds * 1000)}ms"
    return f"{seconds:.1f}s"


def _format_docs(nodes: list[NodeWithScore]) -> str:
    parts = []
    for n in nodes:
        doc_type = n.node.metadata.get("doc_type", "document")
        content = n.node.get_content()
        parts.append(f"[{doc_type}]\n{content}")
    return "\n\n".join(parts)


def _build_prompt(docs: list[NodeWithScore], intent, body: str) -> list:
    prompt = [SystemMessage(content=ENQUIRY_DRAFT_PROMPT)]
    if docs:
        prompt.append(SystemMessage(content=f"Relevant tenancy documents:\n\n{_format_docs(docs)}"))
    else:
        prompt.append(SystemMessage(content=ENQUIRY_NO_DOCS_PROMPT))

    prompt.append(HumanMessage(content=f"[intent: {intent.value}]\n\n{body}"))
    return prompt


class EnquiryService:
    def __init__(self, rag: RagRetriever) -> None:
        self._rag = rag

    async def draft_response(self, enquiry: EnquiryRequest) -> EnquiryResponse:
        """Classifies the enquiry intent and returns an LLM-drafted email reply."""
        intent = await classify_rag_intent(enquiry.body)

        doc_types = INTENT_DOC_TYPES.get(intent)
        if doc_types is None:
            logger.error("EnquiryService.draft_response | No doc types mapped for intent: %s", intent)
            doc_types = frozenset()

        docs: list[NodeWithScore] = []
        if doc_types:
            try:
                docs = await self._rag.aretrieve(query=enquiry.body, doc_types=doc_types, property_id=enquiry.property_id)
            except Exception as exc:
                logger.error("EnquiryService.draft_response | RAG retrieval failed for intent %s: %s", intent, exc)

        prompt = _build_prompt(docs, intent, enquiry.body)

        try:
            llm = get_llm().bind(max_tokens=400)
            result = await llm.ainvoke(prompt)
            return EnquiryResponse(draft=result.content, sources=extract_sources(docs))
        except Exception as exc:
            logger.error("EnquiryService.draft_response | LLM failed: %s", exc)
            return EnquiryResponse(draft="", sources=extract_sources(docs))

    async def stream_draft_response(self, enquiry: EnquiryRequest):
        """
        Async generator — yields SSE strings for each pipeline step, then the final draft.

        Event sequence:
            {"type": "step",   "step": "intent_classified", "label": "Intent classified", "meta": "maintenance · 18ms"}
            {"type": "step",   "step": "rag_retrieval",     "label": "RAG retrieval",     "meta": "3 docs · 94ms"}
            {"type": "step",   "step": "llm_draft",         "label": "LLM draft",         "meta": "gpt-4o-mini · 1.4s"}
            {"type": "step",   "step": "compliance_check",  "label": "Compliance check",  "meta": "NSW RTA — landlord repair duties"}
            {"type": "result", "draft": "...",               "sources": [...]}
            data: [DONE]
        """
        # ── Step 1: intent classification ────────────────────────────────────────
        t0 = time.monotonic()
        try:
            intent = await classify_rag_intent(enquiry.body)
        except Exception as exc:
            logger.error(
                "EnquiryService.stream_draft_response | intent classification failed: %s", exc
            )
            yield _sse("error", message="Failed to classify enquiry intent.")
            yield "data: [DONE]\n\n"
            return

        yield _sse(
            "step",
            step="intent_classified",
            label="Intent classified",
            meta=f"{intent.value} · {_fmt_elapsed(time.monotonic() - t0)}",
        )

        # ── Step 2: RAG retrieval ────────────────────────────────────────────────
        doc_types = INTENT_DOC_TYPES.get(intent, frozenset())
        docs: list[NodeWithScore] = []
        if doc_types:
            t0 = time.monotonic()
            try:
                docs = await self._rag.aretrieve(query=enquiry.body, doc_types=doc_types)
            except Exception as exc:
                logger.error(
                    "EnquiryService.stream_draft_response | RAG retrieval failed for intent %s: %s",
                    intent, exc,
                )
            yield _sse(
                "step",
                step="rag_retrieval",
                label="RAG retrieval",
                meta=f"{len(docs)} docs · {_fmt_elapsed(time.monotonic() - t0)}",
            )

        # ── Step 3: LLM draft ────────────────────────────────────────────────────
        prompt = _build_prompt(docs, intent, enquiry.body)

        t0 = time.monotonic()
        try:
            llm = get_llm().bind(max_tokens=400)
            result = await llm.ainvoke(prompt)
            draft = result.content
        except Exception as exc:
            logger.error("EnquiryService.stream_draft_response | LLM failed: %s", exc)
            yield _sse("error", message="Failed to generate draft reply.")
            yield "data: [DONE]\n\n"
            return

        yield _sse(
            "step",
            step="llm_draft",
            label="LLM draft",
            meta=f"{settings.OPENAI_MODEL_NAME} · {_fmt_elapsed(time.monotonic() - t0)}",
        )

        # ── Step 4: compliance check (lookup, no LLM) ────────────────────────────
        rules = INTENT_COMPLIANCE_RULES.get(intent, "NSW tenancy regulations")
        yield _sse("step", step="compliance_check", label="Compliance check", meta=rules)

        # ── Final result ─────────────────────────────────────────────────────────
        yield _sse("result", draft=draft, sources=[c.model_dump() for c in extract_sources(docs)])
        yield "data: [DONE]\n\n"
