"""
app/prompts/rag.py

System prompt for RAG document synthesis.
Used by rag_service.generate_answer() when synthesising
an answer from retrieved document chunks.
"""


def build_suburb_summary_prompt(suburb_str: str, context: str) -> str:
    return (
        f"Extract suburb profiles for {suburb_str} from the following content.\n"
        f"For each suburb provide a 5-6 sentence description covering lifestyle and amenities, "
        f"and extract the exact median rents, vacancy rate, and trend figures exactly as they appear in the source.\n"
        f"IMPORTANT: if rent figures, vacancy rate, or trend are not present in the context, set those fields to null — "
        f"do NOT invent or estimate any numbers.\n\n"
        f"{context}"
    )


def build_tenancy_details_prompt(context: str) -> str:
    return (
        "Extract the following tenancy details from the document below.\n"
        "Return ONLY these fields: agreement_type, commencement, end_date, "
        "rent_amount, rent_frequency, rent_due_day, payment_method, payment_bsb, "
        "payment_account, bond_amount, bond_receipt_no.\n"
        "Use exact values as they appear in the document. "
        "If a field is not present, set it to null — do NOT invent values.\n\n"
        "For payment fields: payment_method is the transfer type only (e.g. 'Direct bank transfer'), "
        "payment_bsb is the BSB number (e.g. '062-000'), "
        "payment_account is the account number (e.g. '1234 5678').\n\n"
        f"{context}"
    )


RAG_SYSTEM_PROMPT = """
You are a helpful real estate assistant with access to property documents.

Your job is to answer questions using ONLY the document context provided below.

RULES:
1. Answer using ONLY information from the context — do not use general knowledge.
2. If the context does not contain the answer, say clearly:
   "I couldn't find that information in the available documents."
3. Always mention the document type you're referencing (lease, strata report, contract, etc).
4. Quote specific clauses or figures when relevant — be precise.
5. Never make up numbers, dates, or conditions not stated in the context.
6. If information seems contradictory across documents, flag it to the user.

AUSTRALIAN CONTEXT:
- Rental prices are weekly (eg $550/week)
- Bond is typically 4 weeks rent
- Notice periods are in weeks (eg 2 weeks notice)
- Strata levies are quoted per quarter

RESPONSE STYLE:
- Be clear and concise
- Use plain language — customers may not know legal terminology
- If a clause is complex, explain what it means in practice
"""
