from typing import Final

DOC_TYPE_FAQ: Final[str] = "faq"
DOC_TYPE_POLICY: Final[str] = "policy"
DOC_TYPE_GUIDE: Final[str] = "guide"
DOC_TYPE_LEASE: Final[str] = "lease"
DOC_TYPE_REPORT: Final[str] = "report"
DOC_TYPE_NOTICE: Final[str] = "notice"
DOC_TYPE_INVOICE: Final[str] = "invoice"
DOC_TYPE_WATER_BILL: Final[str] = "water_bill"
DOC_TYPE_INSPECTION_NOTICE: Final[str] = "inspection_notice"
DOC_TYPE_RENEWAL_OFFER: Final[str] = "renewal_offer"
DOC_TYPE_MAINTENANCE_LOG: Final[str] = "maintenance_log"
DOC_TYPE_BOND_LODGEMENT: Final[str] = "bond_lodgement"
DOC_TYPE_RENT_LEDGER: Final[str] = "rent_ledger"
DOC_TYPE_LEGISLATION: Final[str] = "legislation"
DOC_TYPE_GENERIC: Final[str] = "generic"

ALLOWED_DOC_TYPES: Final[frozenset[str]] = frozenset({
    DOC_TYPE_FAQ,
    DOC_TYPE_POLICY,
    DOC_TYPE_GUIDE,
    DOC_TYPE_LEASE,
    DOC_TYPE_REPORT,
    DOC_TYPE_NOTICE,
    DOC_TYPE_INVOICE,
    DOC_TYPE_WATER_BILL,
    DOC_TYPE_INSPECTION_NOTICE,
    DOC_TYPE_RENEWAL_OFFER,
    DOC_TYPE_MAINTENANCE_LOG,
    DOC_TYPE_BOND_LODGEMENT,
    DOC_TYPE_RENT_LEDGER,
    DOC_TYPE_LEGISLATION,
    DOC_TYPE_GENERIC,
})

SUPPORTED_EXTENSIONS: Final[frozenset[str]] = frozenset({
    ".pdf", ".docx", ".xlsx", ".pptx",
    ".jpeg", ".jpg", ".png", ".bmp", ".tiff",
})

# Rule-based classification keywords
FAQ_KEYWORDS: Final[tuple[str, ...]] = ("q:", "a:")

POLICY_KEYWORDS: Final[tuple[str, ...]] = (
    "policy", "terms", "agency profile", "about us",
)

GUIDE_KEYWORDS: Final[tuple[str, ...]] = (
    "step 1", "guide", "suburb", "median rent", "vacancy rate",
    "transport", "schools",
)

LEASE_KEYWORDS: Final[tuple[str, ...]] = (
    "residential tenancy agreement", "lease agreement",
    "signed by", "tenant", "landlord", "bond",
)

REPORT_KEYWORDS: Final[tuple[str, ...]] = (
    "market report", "valuation", "inspection", "summary report",
)

NOTICE_KEYWORDS: Final[tuple[str, ...]] = (
    "notice to", "hereby notified", "vacate", "breach",
)

INVOICE_KEYWORDS: Final[tuple[str, ...]] = (
    "invoice", "amount due", "payment receipt", "total:",
    "fee schedule", "fees payable", "schedule of fees",
)

WATER_BILL_KEYWORDS: Final[tuple[str, ...]] = (
    "water usage", "water bill", "water charges", "kilolitres", "sydney water",
)

INSPECTION_NOTICE_KEYWORDS: Final[tuple[str, ...]] = (
    "inspection notice", "routine inspection", "notice of entry", "property inspection",
)

RENEWAL_OFFER_KEYWORDS: Final[tuple[str, ...]] = (
    "renewal offer", "lease renewal", "offer to renew", "renew your lease",
)

MAINTENANCE_LOG_KEYWORDS: Final[tuple[str, ...]] = (
    "maintenance log", "maintenance request", "work order", "repair log",
)

BOND_LODGEMENT_KEYWORDS: Final[tuple[str, ...]] = (
    "bond lodgement", "bond receipt", "rental bond board", "bond reference",
)

RENT_LEDGER_KEYWORDS: Final[tuple[str, ...]] = (
    "rent ledger", "rental ledger", "rent statement", "payment history",
)

LEGISLATION_KEYWORDS: Final[tuple[str, ...]] = (
    "parliament of new south wales",
    "statutory instrument",
    "residential tenancies amendment act",
    "residential tenancies amendment regulation",
    "residential tenancies regulation 2019",
    "residential tenancies act 2010",
)

# Ordered rules: (doc_type, keywords, match_all)
# match_all=True  → all keywords must be present
# match_all=False → any keyword is sufficient
# Specific types ordered before general to avoid early mis-classification.
CLASSIFICATION_RULES: Final[tuple[tuple[str, tuple[str, ...], bool], ...]] = (
    (DOC_TYPE_FAQ,               FAQ_KEYWORDS,               True),
    (DOC_TYPE_INSPECTION_NOTICE, INSPECTION_NOTICE_KEYWORDS, False),
    (DOC_TYPE_RENEWAL_OFFER,     RENEWAL_OFFER_KEYWORDS,     False),
    (DOC_TYPE_BOND_LODGEMENT,    BOND_LODGEMENT_KEYWORDS,    False),
    (DOC_TYPE_RENT_LEDGER,       RENT_LEDGER_KEYWORDS,       False),
    (DOC_TYPE_WATER_BILL,        WATER_BILL_KEYWORDS,        False),
    (DOC_TYPE_MAINTENANCE_LOG,   MAINTENANCE_LOG_KEYWORDS,   False),
    (DOC_TYPE_NOTICE,            NOTICE_KEYWORDS,            False),
    (DOC_TYPE_LEGISLATION,       LEGISLATION_KEYWORDS,       False),
    (DOC_TYPE_LEASE,             LEASE_KEYWORDS,             False),
    (DOC_TYPE_INVOICE,           INVOICE_KEYWORDS,           False),
    (DOC_TYPE_REPORT,            REPORT_KEYWORDS,            False),
    (DOC_TYPE_POLICY,            POLICY_KEYWORDS,            False),
    (DOC_TYPE_GUIDE,             GUIDE_KEYWORDS,             False),
)
