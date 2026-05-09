# Intent Classification — Test Cases

Each section lists example messages, the expected intent, and whether it hits the fast path (keyword match, no LLM) or the LLM path.

---

## `search`

User wants to find, browse, or list properties.

| Message | Path |
|---|---|
| "Show me 3-bedroom houses in Parramatta" | LLM |
| "I'm looking for apartments under $600k in Sydney" | LLM |
| "Find rentals in Chatswood" | LLM |
| "Any 2-bed units in the Inner West?" | LLM |
| "List townhouses for sale in Bondi" | LLM |
| "What's available in Surry Hills under $800 a week?" | LLM |
| "Show me something with a garage in Penrith" | LLM |

**Follow-up / refinement (must stay `search`, not `general`)**

| Message | Context | Path |
|---|---|---|
| "What about apartments?" | Previous search for houses | LLM |
| "Any cheaper ones?" | Previous search results shown | LLM |
| "Also show me townhouses" | After a search | LLM |
| "What about Hurstville as well?" | After Chatswood search | LLM |

**Early response expected** (no location, no prior context)

| Message | Expected early_response |
|---|---|
| "Show me 2-bedroom apartments" | Ask for suburb/area |
| "Find me something under $500k" | Ask for suburb/area |

---

## `document_query`

User asks about leases, contracts, strata, agency info, staff/agent details, office hours, fees, or contact information. Routes to vector search — answer must come from `[RETRIEVED DOCUMENTS]`.

| Message | Path |
|---|---|
| "What are the agent details?" | LLM |
| "Who is the managing agent?" | LLM |
| "What is the agent's licence number?" | LLM |
| "Can I get the agent's contact details?" | LLM |
| "What are your office hours?" | LLM |
| "What is your phone number?" | LLM |
| "How do I contact Harbour Realty Group?" | LLM |
| "What are your agency fees?" | LLM |
| "What does a strata report include?" | LLM |
| "Can you explain what a section 32 is?" | LLM |
| "What happens if I break my lease early?" | LLM |
| "What is a building and pest inspection?" | LLM |
| "What are the terms in my lease agreement?" | LLM |
| "Who are the key staff at your agency?" | LLM |
| "What is your website?" | LLM |

**No data in retrieved docs — must NOT fabricate**

If `[RETRIEVED DOCUMENTS]` does not contain the answer, the LLM must respond:
> "I don't have that detail on hand — please contact Harbour Realty Group directly."

It must NEVER invent agent names, licence numbers, phone numbers, or email addresses.

---

## `hybrid_search`

User wants both property listings AND document/agency information in the same message. No `early_response` should be set.

| Message | Path |
|---|---|
| "Show me 2-bed apartments in Chatswood and explain what a strata report is" | LLM |
| "Find rentals in Parramatta and tell me about break lease rules" | LLM |
| "List houses in Bondi and what are your agency fees?" | LLM |
| "Search for units in the city and what does a section 32 mean?" | LLM |
| "Find 3-bedroom houses under $1m in Sydney — also, what are your trading hours?" | LLM |

---

## `booking`

User wants to book or schedule a property inspection.

**Fast path** (booking keyword present, no search keyword)

| Message | Path |
|---|---|
| "I'd like to book an inspection" | Fast |
| "Can I schedule a viewing?" | Fast |
| "I want to arrange an open for inspection" | Fast |
| "Book a viewing for me" | Fast |

**LLM path** (ambiguous or combined with property reference)

| Message | Path |
|---|---|
| "I'd like to see the first property" | LLM (booking continuation check) |
| "Can I come and look at that apartment?" | LLM |

**Booking continuation** (state-based — slots are pending, user selects one)

| Message | State condition |
|---|---|
| "I'll take the Saturday 10am slot" | `available_slots` populated, not confirmed/cancelled |
| "The second one works for me" | Same |
| "Let's go with option 3" | Same |
| "Saturday morning please" | Same |

---

## `cancellation`

User wants to cancel an existing booking.

**Fast path** (cancellation keyword present, no search keyword)

| Message | Path |
|---|---|
| "I need to cancel my inspection" | Fast |
| "Please cancel my booking" | Fast |
| "I want to cancel" | Fast |
| "Cancellation please" | Fast |
| "I'm no longer available" | Fast |
| "I no longer want to attend" | Fast |
| "Withdraw my booking" | Fast |

**Must NOT trigger on ambiguous phrasing**

| Message | Expected intent | Reason |
|---|---|---|
| "Can I cancel and find a new property?" | `general` (compound) | search keyword suppresses fast path |

---

## `booking_lookup`

User wants to view or check an existing booking.

**Fast path** (lookup keyword match)

| Message | Path |
|---|---|
| "Can I see my booking?" | Fast |
| "What time is my inspection?" | Fast |
| "Show my booking details" | Fast |
| "I booked an inspection, can I check it?" | Fast |
| "Find my booking" | Fast |
| "My confirmation number is ABC123, can you look it up?" | Fast |
| "What's my booking status?" | Fast |
| "View my inspection" | Fast |

---

## `search_then_book`

User wants to search for a property AND book an inspection in the same message.

| Message | Path |
|---|---|
| "Find 2-bed apartments in Newtown and book an inspection" | LLM |
| "Show me houses in Randwick and schedule a viewing" | LLM |
| "Search for rentals in Glebe and I'd like to book one" | LLM |

**Expected behaviour:** search results are shown first; no booking tools are called until the user selects a property.

---

## `deposit_payment`

User wants to pay or check a holding deposit and the property is already identified from conversation context.

| Message | Context | Path |
|---|---|---|
| "Can I pay the deposit?" | Property in context | Fast |
| "Pay my holding deposit" | Listing in context | Fast |
| "I want to pay the holding deposit" | After search, property confirmed | Fast |
| "Can I check the deposit status?" | Listing in context | LLM |

---

## `search_then_deposit`

User provides a property address/location AND wants to pay a deposit, or follows up on earlier search results but hasn't clearly identified the property.

| Message | Path |
|---|---|
| "Pay deposit for the apartment at 1 George St Chatswood" | Fast (deposit + search keywords) |
| "I want to pay the deposit for the apartment in Parramatta" | Fast |
| "I need to pay the holding deposit but I'm not sure of the address" | LLM |
| "Can I pay the deposit for one of those properties?" | LLM (follow-up, no clear property) |

---

## `general`

Greeting, unclear, out-of-scope, or unclassifiable. No RAG or search runs.

| Message | Notes |
|---|---|
| "Hello" | Greeting |
| "Hi there" | Greeting |
| "Thanks!" | Acknowledgement |
| "What is the weather today?" | Out of scope |
| "Can you book me a flight?" | Out of scope |
| "Tell me a joke" | Out of scope |
| "Can you cancel my booking and find me a new property?" | Compound → general + early_response |

**Greeting + question in the same message — classify by the question**

| Message | Expected intent |
|---|---|
| "Hello! What are your trading hours?" | `document_query` |
| "Hi, can I book an inspection?" | `booking` |
| "Hey, show me apartments in Surry Hills" | `search` |

---

## Edge cases

| Message | Expected intent | Reason |
|---|---|---|
| "Show me houses in Bondi and book one" | `search_then_book` | search + booking |
| "Find apartments in Newtown and what is a strata report?" | `hybrid_search` | search + document |
| "Cancel and find me something else" | `general` | compound with no clean split |
| "No 2 please" (slots pending) | `booking` (continuation) | ordinal during slot selection |
| "The second property" (after search, no slots) | `booking` | ordinal booking reference |
| "units in the CBD" | `search` | "unit" → treat as "apartment" |
