"""
app/schemas/rag.py

Intent classification types and lookup maps for the RAG enquiry pipeline.
"""

from enum import Enum

from pydantic import BaseModel


class SourceChunk(BaseModel):
    file_name: str
    page: int | None
    score: float
    text: str


class RagIntent(str, Enum):
    WATER_BILL       = "water_bill"
    INSPECTION       = "inspection"
    LEASE_RENEWAL    = "lease_renewal"
    MAINTENANCE      = "maintenance"
    BOND             = "bond"
    RENT_PAYMENT     = "rent_payment"
    LEASE_CLAUSE     = "lease_clause"
    DOCUMENT_REQUEST = "document_request"
    GENERAL          = "general"


class RagClassification(BaseModel):
    intent: RagIntent

RAG_KEYWORD_MAP: dict[RagIntent, frozenset[str]] = {
    RagIntent.WATER_BILL: frozenset([
        "water bill",
        "water usage",
        "water rate",
        "water charge",
        "water invoice",
        "water meter",
    ]),
    RagIntent.MAINTENANCE: frozenset([
        "repair",
        "fix",
        "broken",
        "leaking",
        "leak",
        "fault",
        "damage",
        "damaged",
        "not working",
        "maintenance request",
        "maintenance issue",
        "tradesperson",
        "plumber",
        "electrician",
    ]),
    RagIntent.BOND: frozenset([
        "bond refund",
        "bond return",
        "bond claim",
        "security deposit",
        "bond lodgement",
        "about the bond",
        "my bond",
    ]),
    RagIntent.RENT_PAYMENT: frozenset([
        "pay rent",
        "rent payment",
        "rent due",
        "overdue rent",
        "rent arrears",
        "rent receipt",
        "rental payment",
    ]),
    RagIntent.LEASE_RENEWAL: frozenset([
        "renew lease",
        "lease renewal",
        "extend lease",
        "lease extension",
        "new lease",
        "end of lease",
    ]),
    RagIntent.INSPECTION: frozenset([
        "routine inspection",
        "outgoing inspection",
        "inspection notice",
        "entry notice",
        "move-out inspection",
        "final inspection",
        "condition report",
    ]),
    RagIntent.LEASE_CLAUSE: frozenset([
        "lease clause",
        "lease condition",
        "lease says",
        "breach of lease",
        "what does my lease",
        "lease agreement says",
    ]),
    RagIntent.DOCUMENT_REQUEST: frozenset([
        "copy of lease",
        "lease copy",
        "send me the lease",
        "inspection report copy",
        "rental statement",
        "payment receipt",
        "copy of the",
        "send me a copy",
    ]),
}

INTENT_COMPLIANCE_RULES: dict[RagIntent, str] = {
    RagIntent.WATER_BILL:       "NSW Residential Tenancies Act s.39",
    RagIntent.INSPECTION:       "NSW RTA — entry & inspection rules",
    RagIntent.LEASE_RENEWAL:    "NSW RTA — lease renewal terms",
    RagIntent.MAINTENANCE:      "NSW RTA — landlord repair duties",
    RagIntent.BOND:             "NSW Fair Trading bond guidelines",
    RagIntent.RENT_PAYMENT:     "NSW tenancy rules — rent obligations",
    RagIntent.LEASE_CLAUSE:     "NSW tenancy rules — lease terms",
    RagIntent.DOCUMENT_REQUEST: "NSW RTA — document disclosure",
    RagIntent.GENERAL:          "NSW tenancy regulations",
}

INTENT_DOC_TYPES: dict[RagIntent, frozenset[str]] = {
    RagIntent.WATER_BILL:       frozenset(["water_bill", "lease", "legislation"]),
    RagIntent.INSPECTION:       frozenset(["inspection_notice", "lease", "legislation"]),
    RagIntent.LEASE_RENEWAL:    frozenset(["lease", "renewal_offer", "legislation"]),
    RagIntent.MAINTENANCE:      frozenset(["maintenance_log", "lease", "legislation"]),
    RagIntent.BOND:             frozenset(["bond_lodgement", "lease", "legislation"]),
    RagIntent.RENT_PAYMENT:     frozenset(["rent_ledger", "lease", "legislation"]),
    RagIntent.LEASE_CLAUSE:     frozenset(["lease", "legislation"]),
    RagIntent.DOCUMENT_REQUEST: frozenset(["lease", "notice", "inspection_notice", "bond_lodgement", "rent_ledger"]),
    RagIntent.GENERAL:          frozenset(["lease", "policy", "faq", "guide", "legislation"]),
}
