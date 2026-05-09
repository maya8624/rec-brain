# Agent Design Notes

## Problem Summary

The booking flow usually works, but it becomes unreliable when the LLM receives too much mixed conversation history. In longer threads, the model can:

- anchor on earlier failed booking attempts
- reuse stale search context
- return a plausible plain-text answer instead of calling the correct tool

Example failure:

- user intent is correctly classified as `booking`
- user says `Book an inspection for the first one`
- instead of calling `check_availability`, the LLM replies with a made-up "no available slots" style answer

The core issue is not only prompt wording. It is also a context and responsibility problem:

- too much irrelevant history reaches the model
- deterministic reference resolution still depends on prompt-following
- one large prompt mixes search, booking, cancellation, and document behaviors

## Recommended Direction

Use a simpler, more deterministic design:

1. Keep one shared base prompt
2. Add a small prompt overlay per user intent
3. Resolve deterministic references in code/state before the LLM step
4. Shrink the context window for booking and cancellation flows
5. Add evals for multi-turn workflow regressions

## Prompt Structure

### Shared Base Prompt

Keep only global rules here:

- brand voice
- scope boundaries
- Australian formatting rules
- never invent property, booking, or agent data
- escalation rules

Do not keep full booking, cancellation, and search workflows together in one monolithic prompt.

### Intent-Specific Prompt Overlays

Build prompt instructions based on the active intent.

`search`

- search/filter behavior
- result formatting
- no booking/cancellation rules

`booking`

- if `booking_context.property_id` exists, use it
- call `check_availability`
- present slots
- wait for slot choice
- confirm
- call `book_inspection`

`cancellation`

- use `booking_context.confirmation_id` if present
- otherwise use lookup flow
- confirm cancellation
- call `cancel_inspection`

`booking_lookup`

- call `get_booking`
- present results clearly
- if multiple bookings are returned, ask the user to choose

`document_query`

- answer only from retrieved documents/context
- do not mix in booking logic

## Context Strategy

Structured state should be the main memory, not long transcript history.

For `booking`, the model should usually receive:

- latest user message
- latest relevant assistant turn if needed
- current `[PROPERTY SEARCH RESULTS]`
- current `[BOOKING CONTEXT]`

Avoid sending:

- unrelated office-hours or document-chat turns
- earlier failed booking attempts for different properties
- stale search branches that are no longer relevant

For `cancellation`, use the same idea:

- latest user message
- current booking context
- only the most relevant prior turn or two

## Deterministic Resolution Before the LLM

If the application can resolve something from state with certainty, code should do it before the LLM is asked to act.

Examples:

- `the first one` -> `search_results[0]`
- `the second one` -> `search_results[1]`
- `option 3` -> `search_results[2]`

For booking, persist the resolved selection into `booking_context`:

- `property_id`
- `property_address`

For cancellation/lookup, if exactly one booking is known, persist:

- `confirmation_id`
- `property_address`

If multiple candidates exist, do not guess. Ask the user to choose.

## Suggested Graph Responsibilities

### `intent_node`

- classify the latest user message

### `reference_resolution_node`

- for booking/deposit flows
- resolve ordinal and follow-up references from `search_results`
- store the resolved property in state

### `lookup_resolution_node`

- for cancellation or booking lookup
- decide whether existing booking context is sufficient
- if not, use booking lookup flow

### `agent_node`

- assemble the base prompt + intent overlay
- receive only focused, relevant context
- make tool calls or format tool outputs

### `context_update_node`

- persist successful tool outputs back into structured state

## Why This Is Better

- less prompt competition between unrelated workflows
- less chance of hallucinated plain-text replies during tool workflows
- easier debugging because state carries the workflow explicitly
- easier testing because each intent has a smaller surface area
- safer handling of multi-turn follow-ups like `first one` or `go ahead`

## Evals To Add

1. Search results shown, then `Book an inspection for the first one`
   Expected: `check_availability` tool call

2. Several earlier failed booking attempts, then `Book an inspection for the first one`
   Expected: current selected property is resolved correctly; no hallucinated no-slot reply without tool call

3. Cancellation requested, user does not know confirmation ID, lookup returns one booking, then user says `go ahead`
   Expected: `cancel_inspection` tool call

4. Booking lookup returns multiple bookings, then user says `cancel it`
   Expected: ask the user which booking to cancel; do not guess

## Bottom Line

The recommended long-term fix is not "keep patching the main agent prompt."

The better direction is:

- smaller prompts by intent
- tighter context by workflow
- explicit structured state
- deterministic resolution for unambiguous references
- eval coverage for multi-turn failures
