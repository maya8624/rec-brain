# Invoice Extraction Endpoint

## Purpose

Provide a dedicated extraction service for invoice PDFs and images. The .NET backend uploads a file, receives structured fields in the response, and owns persistence. rec-brain acts as a pure extraction layer — no data is stored here.

This is intentionally separate from `/api/documents/ingest`, which is RAG-oriented (parse → chunk → embed → pgvector). Invoices need structured fields (amounts, dates, line items), not semantic search chunks.

---

## Endpoint

```
POST /api/documents/invoice-extract
Header: X-API-Key: <BACKEND_API_KEY>
Content-Type: multipart/form-data
```

**Form fields**

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | file | Yes | PDF or image — max 20 MB |
| `property_id` | string | No | Property this invoice belongs to |

**Accepted file types:** PDF, JPEG, PNG, TIFF, BMP

**Response**

```json
{
  "tool_name": "save_invoice",
  "success": true,
  "filename": "maintenance-invoice.pdf",
  "property_id": "prop-123",
  "data": {
    "vendor_name": "Acme Plumbing",
    "vendor_address": "12 George St, Sydney NSW 2000",
    "customer_name": "Sunshine Realty",
    "invoice_id": "INV-0042",
    "invoice_date": "2026-06-01",
    "due_date": "2026-06-15",
    "subtotal": 450.00,
    "tax": 45.00,
    "total": 495.00,
    "currency": "$",
    "confidence": 0.97,
    "line_items": [
      {
        "description": "Emergency pipe repair",
        "quantity": 1.0,
        "unit_price": 350.00,
        "amount": 350.00
      },
      {
        "description": "Materials",
        "quantity": 1.0,
        "unit_price": 100.00,
        "amount": 100.00
      }
    ]
  }
}
```

`tool_name` is always `"save_invoice"`. The .NET backend uses this field to dispatch to the correct handler. All `data` fields are nullable — a partial extraction returns HTTP 200 with `null` for undetected fields. `line_items` is `[]` when no line items are found. Dates are `YYYY-MM-DD` strings.

**Error responses**

| Status | When |
|---|---|
| 400 | Unsupported file type or file exceeds 20 MB |
| 403 | Missing or invalid `X-API-Key` |
| 422 | Azure DI returned no documents or extraction failed |
| 500 | Unexpected server error |

---

## Azure Document Intelligence Model

Uses `prebuilt-invoice` — purpose-built for B2B billing documents.

`prebuilt-layout` (used by the existing ingest pipeline) extracts markdown text for RAG. `prebuilt-invoice` instead returns a typed field graph: vendor, customer, amounts, line items, dates. Choosing the right model per document type avoids unnecessary chunking and embedding overhead for financial documents.

`prebuilt-receipt` was not included in this endpoint because the real estate domain deals with B2B invoices (agency fees, maintenance, contractors, utilities), not consumer POS receipts. Receipt extraction will be implemented as a separate endpoint (`POST /api/receipts/extract`) for the accounts demo — it extracts different fields (`MerchantName`, `TransactionDate`, `Items`, `Subtotal`, `Tax`, `Tip`, `Total`) and targets consumer receipts such as expense claims and reimbursements.

**Extracted fields**

| Azure DI field | Mapped to |
|---|---|
| `VendorName` | `vendor_name` |
| `VendorAddress` | `vendor_address` |
| `CustomerName` | `customer_name` |
| `InvoiceId` | `invoice_id` |
| `InvoiceDate` | `invoice_date` |
| `DueDate` | `due_date` |
| `SubTotal` | `subtotal` |
| `TotalTax` | `tax` |
| `InvoiceTotal` | `total` |
| `InvoiceTotal.symbol` | `currency` |
| `Items[]` | `line_items[]` |
| `doc.confidence` | `confidence` |

All fields are optional in the response — Azure DI may not detect every field depending on document quality.

---

## Architecture

```
POST /api/documents/invoice-extract (app/api/routes/documents.py)
        ↓  file validation, size check
InvoiceExtractionService.extract()  (app/services/invoice_service.py)
        ↓  via InvoiceParserProtocol
AzureInvoiceParser._extract()       (app/infrastructure/invoice_parser.py)
        ↓  begin_analyze_document("prebuilt-invoice")
Azure Document Intelligence API
        ↓  DocumentField graph → InvoiceData
InvoiceExtractionResponse           (app/schemas/invoice.py)
```

**Key design decision — Protocol boundary:**
`InvoiceExtractionService` depends on `InvoiceParserProtocol`, not directly on `AzureInvoiceParser`. This means unit tests can inject a mock parser without needing Azure credentials. All SDK types (`DocumentField`, `CurrencyValue`, etc.) are contained inside `invoice_parser.py` and never leak into the service or route layers.

---

## Files

| File | Role |
|---|---|
| `app/schemas/invoice.py` | `LineItem`, `InvoiceData`, `InvoiceExtractionResponse` |
| `app/infrastructure/invoice_parser.py` | `InvoiceParserProtocol` + `AzureInvoiceParser` |
| `app/services/invoice_service.py` | `InvoiceExtractionService`, `InvoiceExtractionError` |
| `app/api/routes/documents.py` | FastAPI routes — `/ingest` and `/extract` |
| `main.py` | Wires `AzureInvoiceParser` → `InvoiceExtractionService` → `app.state` |

---

## Config

No new environment variables. Reuses:

| Variable | Purpose |
|---|---|
| `AZURE_DOC_INTEL_ENDPOINT` | Azure DI resource endpoint |
| `AZURE_DOC_INTEL_KEY` | Azure DI API key |
| `BACKEND_API_KEY` | Internal auth — verified via `X-API-Key` header |

---

## Testing

**Manual (Swagger)**
1. Start server: `uvicorn main:app --reload`
2. Open `/docs` → `POST /api/documents/invoice-extract`
3. Set `X-API-Key` header, upload a PDF invoice, submit
4. Confirm `data.vendor_name`, `data.total`, `data.line_items` are populated

**Unit test approach**
Inject a mock `InvoiceParserProtocol` into `InvoiceExtractionService` — no Azure credentials needed:

```python
class MockInvoiceParser:
    async def parse(self, content: bytes, filename: str) -> InvoiceData:
        return InvoiceData(vendor_name="Acme", total=495.00, confidence=0.99)

service = InvoiceExtractionService(parser=MockInvoiceParser())
result = await service.extract(b"...", "invoice.pdf", "prop-123")
assert result.total == 495.00
```

**Edge cases to cover**
- Unsupported file type → 400
- File over 20 MB → 400
- Missing `X-API-Key` → 403
- Azure DI finds no documents (blank page, corrupt file) → 422
- Partial extraction (some fields None) → 200 with nulls in response
