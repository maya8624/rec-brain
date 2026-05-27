"""
app/schemas/enquiry.py

Types and keyword map for tenant/owner enquiry classification.
"""

from enum import Enum

from pydantic import BaseModel

class EnquiryRequest(BaseModel):
    id: str
    body: str
    tenant_id: str | None
    property_id: str | None
    intent: str | None


class EnquiryResponse(BaseModel):
    draft: str
    sources: list[str]  # unique file names from retrieved documents

class EnquiryIntent(str, Enum):
    WATER_BILL       = "water_bill"
    INSPECTION       = "inspection"
    LEASE_RENEWAL    = "lease_renewal"
    MAINTENANCE      = "maintenance"
    BOND             = "bond"
    RENT_PAYMENT     = "rent_payment"
    LEASE_CLAUSE     = "lease_clause"
    DOCUMENT_REQUEST = "document_request"
    GENERAL          = "general"


class EnquiryClassification(BaseModel):
    intent: EnquiryIntent

ENQUIRY_KEYWORD_MAP: dict[EnquiryIntent, frozenset[str]] = {
    EnquiryIntent.WATER_BILL: frozenset([
        "water bill", 
        "water usage", 
        "water rate", 
        "water charge",
        "water invoice", 
        "water meter",
    ]),
    EnquiryIntent.MAINTENANCE: frozenset([
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
    EnquiryIntent.BOND: frozenset([
        "bond", 
        "bond refund", 
        "bond return", 
        "bond claim",
        "security deposit", 
        "bond lodgement",
    ]),
    EnquiryIntent.RENT_PAYMENT: frozenset([
        "pay rent", 
        "rent payment", 
        "rent due", 
        "overdue rent",
        "rent arrears", 
        "rent receipt", 
        "rental payment",
    ]),
    EnquiryIntent.LEASE_RENEWAL: frozenset([
        "renew lease", 
        "lease renewal", 
        "extend lease", 
        "lease extension",
        "new lease", 
        "end of lease",
    ]),
    EnquiryIntent.INSPECTION: frozenset([
        "routine inspection", 
        "inspection notice", 
        "entry notice",
        "move-out inspection", 
        "final inspection", 
        "condition report",
    ]),
    EnquiryIntent.LEASE_CLAUSE: frozenset([
        "lease clause", 
        "lease condition", 
        "lease says", 
        "breach of lease",
        "what does my lease", 
        "lease agreement says",
    ]),
    EnquiryIntent.DOCUMENT_REQUEST: frozenset([
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

INTENT_DOC_TYPES: dict[EnquiryIntent, frozenset[str]] = {
    EnquiryIntent.WATER_BILL:       frozenset(["water_bill", "lease"]),
    EnquiryIntent.INSPECTION:       frozenset(["inspection_notice", "lease"]),
    EnquiryIntent.LEASE_RENEWAL:    frozenset(["lease", "renewal_offer"]),
    EnquiryIntent.MAINTENANCE:      frozenset(["maintenance_log", "lease"]),
    EnquiryIntent.BOND:             frozenset(["bond_lodgement", "lease"]),
    EnquiryIntent.RENT_PAYMENT:     frozenset(["rent_ledger", "lease"]),
    EnquiryIntent.LEASE_CLAUSE:     frozenset(["lease"]),
    EnquiryIntent.DOCUMENT_REQUEST: frozenset(["lease", "notice", "inspection_notice", "bond_lodgement", "rent_ledger"]),
    EnquiryIntent.GENERAL:          frozenset(["lease", "policy", "faq", "guide"]),
}
