import logging

from llama_index.core.schema import NodeWithScore
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes.enquiry_intent import classify_enquiry_intent
from app.infrastructure.llm import get_llm
from app.prompts.enquiry import ENQUIRY_DRAFT_PROMPT
from app.schemas.enquiry import INTENT_DOC_TYPES, EnquiryRequest, EnquiryResponse
from app.services.rag_service import RagRetriever

logger = logging.getLogger(__name__)


def _format_docs(nodes: list[NodeWithScore]) -> str:
    parts = []
    for n in nodes:
        doc_type = n.node.metadata.get("doc_type", "document")
        content = n.node.get_content()
        parts.append(f"[{doc_type}]\n{content}")
    return "\n\n".join(parts)


def _extract_sources(nodes: list[NodeWithScore]) -> list[str]:
    """Returns unique file names from node metadata, preserving retrieval order."""
    seen: set[str] = set()
    sources: list[str] = []
    for n in nodes:
        name = n.node.metadata.get("file_name")
        if name and name not in seen:
            seen.add(name)
            sources.append(name)
    return sources


class EnquiryService:
    def __init__(self, rag: RagRetriever) -> None:
        self._rag = rag

    async def draft_response(self, enquiry: EnquiryRequest) -> EnquiryResponse:
        """Classifies the enquiry intent and returns an LLM-drafted email reply."""
        intent = await classify_enquiry_intent(enquiry.body)

        doc_types = INTENT_DOC_TYPES.get(intent)
        if doc_types is None:
            logger.error(
                "EnquiryService.draft_response | No doc types mapped for intent: %s",
                intent,
            )
            doc_types = frozenset()

        docs: list[NodeWithScore] = []
        if doc_types:
            try:
                docs = await self._rag.aretrieve(query=enquiry.body, doc_types=doc_types)
            except Exception as exc:
                logger.error(
                    "EnquiryService.draft_response | RAG retrieval failed for intent %s: %s",
                    intent,
                    exc,
                )

        prompt = [SystemMessage(content=ENQUIRY_DRAFT_PROMPT)]
        if docs:
            prompt.append(
                SystemMessage(content=f"Relevant tenancy documents:\n\n{_format_docs(docs)}")
            )
        prompt.append(HumanMessage(content=f"[intent: {intent.value}]\n\n{enquiry.body}"))

        try:
            llm = get_llm().bind(max_tokens=400)
            result = await llm.ainvoke(prompt)
            return EnquiryResponse(draft=result.content, sources=_extract_sources(docs))
        except Exception as exc:
            logger.error("EnquiryService.draft_response | LLM failed: %s", exc)
            return EnquiryResponse(draft="", sources=_extract_sources(docs))
