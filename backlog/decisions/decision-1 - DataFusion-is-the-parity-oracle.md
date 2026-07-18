---
id: decision-1
title: DataFusion is the parity oracle
date: '2026-07-18 14:00'
status: accepted
---
## Context

Two engines run the same rewritten SQL: `transform` (DataFusion, batch) and
`infer`/`infer_batch` (the native `InferFn` interpreter, row-at-a-time). The README
promises they return identical values — batch time == serving time. When they
disagree, we need one unambiguous source of truth to fix toward.

## Decision

**DataFusion is the oracle.** `transform` *is* DataFusion, and the differential
harness gates on it. Where `infer` (native) disagrees with `transform`, **native is
wrong** — match DataFusion's behavior; never codify current native behavior as
correct. This holds for values on the normal numeric path (error *types* are a
separate non-goal — see decision-2).

## Consequences

- Every parity bug is defined as "native diverges from DataFusion," and the fix
  target is DataFusion's output (see TASK-1 and the opaque follow-ups).
- Dev process: on finding a divergence, pin it with a strict `xfail`-on-rust test
  first (so a fix flips it and forces the marker's removal), then fix.
- The differential corpus is the safety net; every divergence lives in a coverage
  gap, which is why the suite can be green while bugs exist.
