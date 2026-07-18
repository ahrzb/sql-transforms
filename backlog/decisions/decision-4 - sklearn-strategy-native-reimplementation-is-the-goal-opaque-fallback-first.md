---
id: decision-4
title: 'sklearn strategy: native reimplementation is the goal, opaque fallback first'
date: '2026-07-18 14:00'
status: accepted
---
## Context

We want sklearn-compatible transformers that serve fast at n=1. Two ways to "use"
sklearn: wrap the fitted sklearn object and call `.transform()` at serve time
(correct but slow — a Python call + array materialization on the request path,
breaks fusion), or reimplement each transformer natively in our engine (fast, but
one at a time). We need both partial coverage *now* and a fast end state.

## Decision

**Native reimplementation is the end goal; the opaque fallback makes partial
coverage shippable in the meantime (2026-07-16).** Sequence:
- **Fallback phase — fallback-first:** stand up the whole compose-in surface with every
  transformer backed by the real sklearn object (opaque node). Proves structure +
  parity harness without engine reimplementation.
- **Native-swap phase — native per transformer:** swap each fallback to a native engine impl,
  one at a time, diffed against the fallback oracle, in tier order.

## Consequences

- The opaque-transform capability (decision-3) is the fallback mechanism, not the
  end state.
- Native impls are validated against the same differential matrix their Python
  fallback passes, so a native/sklearn divergence fails loud (TASK-5, TASK-6).
- The (a) compose-in direction (our estimators into a *stock* sklearn pipeline) is
  deferred; the near-term target is running fitted sklearn objects inside *our*
  engine.
- Transformer priority order lives in the SKLEARN_TRANSFORMERS catalogue.
