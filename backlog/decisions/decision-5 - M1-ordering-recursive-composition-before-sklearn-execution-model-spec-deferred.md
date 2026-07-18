---
id: decision-5
title: >-
  M1 ordering: recursive composition before sklearn; execution-model spec
  deferred
date: '2026-07-18 14:00'
status: accepted
---
## Context

M1 (transformer foundation & sklearn parity) had several candidate first moves:
outstanding parity bugs, recursive composition of unfitted `SQLTransform`s, the
sklearn transformers themselves, and a general UDF/UDAF/macro execution-model spec.

## Decision

**Order (2026-07-16): outstanding bugs → recursive (fit-cascade) composition →
sklearn transformers.** The general UDF/UDAF/macro **execution-model spec is not on
the critical path** — extract it later from the concrete work if/when actually
needed, rather than writing it up front.

## Consequences

- Rationale: stock sklearn `Pipeline.fit` clones + re-fits each step, so the
  recursive-composition primitive is exactly what the sklearn work sits on —
  building it first de-risks sklearn. Shipped `5ac613e`.
- The execution-model spec stays a captured idea, deprioritized off M1; it isn't
  immediately useful and the concrete composition + sklearn work doesn't need it.
