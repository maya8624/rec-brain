"""
rag_intent — classifies the intent of a tenant/owner enquiry.

Strategy: hybrid keyword + LLM

    Fast path (keyword, no LLM):
        - Water bill keywords      → "water_bill"
        - Maintenance keywords     → "maintenance"
        - Bond keywords            → "bond"
        - Rent payment keywords    → "rent_payment"
        - Lease renewal keywords   → "lease_renewal"
        - Inspection keywords      → "inspection"
        - Lease clause keywords    → "lease_clause"
        - Document request kw      → "document_request"

    LLM path (everything else):
        - Ambiguous or multi-topic enquiries
        - Returns RagClassification via with_structured_output
        - Falls back to "general" on any LLM error
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from app.agents.nodes._fast_path import matches_keywords
from app.prompts.rag import RAG_CLASSIFICATION_PROMPT
from app.infrastructure.llm import get_llm
from app.schemas.rag import RAG_KEYWORD_MAP, RagClassification, RagIntent

logger = logging.getLogger(__name__)

def _keyword_intent(message: str) -> RagIntent | None:
    for intent, keywords in RAG_KEYWORD_MAP.items():
        if matches_keywords(message, keywords):
            return intent
    return None


async def classify_rag_intent(enquiry: str) -> RagIntent:
    """Classifies a tenant/owner message into an RagIntent."""
    if not enquiry:
        return RagIntent.GENERAL

    fast = _keyword_intent(enquiry.lower())
    if fast:
        return fast

    prompt = [
        SystemMessage(content=RAG_CLASSIFICATION_PROMPT),
        HumanMessage(content=enquiry),
    ]

    try:
        llm = get_llm().with_structured_output(RagClassification)
        classification: RagClassification = await llm.ainvoke(prompt)
    except Exception as exc:
        logger.error("classify_rag_intent | LLM classification failed: %s", exc)
        return RagIntent.GENERAL

    return classification.intent
