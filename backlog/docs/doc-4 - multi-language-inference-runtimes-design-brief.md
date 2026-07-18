---
id: doc-4
title: Multi-language inference runtimes (design brief)
type: other
created_date: '2026-07-18 15:52'
---

# Multi-language inference runtimes (design brief)

**Status: OUT OF SCOPE — parked as design, not in the work queue (AmirHossein 2026-07-19).**
Was a milestone (archived) with the 7-step sketch below (never ticketed / archived if they were).
Multi-quarter. Entry point is scoping/planning, not implementation. Captured from Fermi (Investigator) 2026-07-18;
Substrait feasibility validated with **real artifacts**, not speculation. Rests on the
two-engine framing (native engine = one-of-N), an open question not being ratified in
the short term.

## Goal

Serve trained SQL-transforms from **any backend language** (Go / Java / C# / Node) with
**no Python, no DataFusion, and no FFI at inference time**. A Rust-core-plus-FFI hub was
explicitly rejected — FFI / native-artifact friction kills backend adoption. Pure-native
per language; WASM-from-Rust for JS only. The native Rust engine becomes **one-of-N**,
not a hub.

## Architecture (validated pieces marked ✓)

- **Interchange is a serialized LOGICAL PLAN, not SQL.** Runtimes consume a plan
  document + Parquet fitted tables → no runtime needs a SQL parser or query engine.
  sqlglot stays the SQL front-end (parse / validate / window-rewrite); Substrait is
  downstream of it, not a replacement.
- ✓ **Substrait is emittable TODAY** from datafusion 54.0.0 python — zero new
  deps/Rust/FFI: `Producer.to_substrait_plan(lp, ctx)` + `Serde.serialize_bytes`.
  Planning is schema-only (no data). Proven sizes: projection 199B; the real rewritten
  serving shape (projection + LEFT JOIN to state tables) 293B; GROUP BY/count 180B.
- ✓ **Empirical limit:** `unnest`/explode is NOT emittable by the df 54.0.0 producer
  ("Unsupported plan type: Unnest"). This draws a clean dividing line:
  - **Composable on Substrait** (shared across all runtimes, free from a plan
    interpreter): numeric/cast/CASE/coalesce, window-agg (rewritten to LEFT JOIN),
    scalar one-hot (a join, no explode), aggregation. + tiny custom scalar primitives
    (COO-construct, hash) implemented once per runtime.
  - **Opaque per-runtime** (need explode → not producible): tfidf, array multi-hot —
    map onto the existing opaque-transform concept (decision-3), corpus-gated.
  So the boundary is **"fixed-fanout composes / variable-expansion is opaque"** — and
  it's empirically grounded, not a guess.
- Runtimes are thin **pre-resolved tree-walk interpreters** (columns→slot indices,
  literals inlined, lookups by id). Small-batch serving is boundary-bound (per the
  benchmark milestone) → a row executor is near-optimal; don't columnarize small-batch
  input. Columnar executor added later for big batch, same IR. **Decouple IR (pin it)
  from executor (vary per language/regime).**
- **Parity: a GENERATIVE differential corpus** (fuzz exprs/SQL → DataFusion → freeze
  expected), run in every runtime's CI. Doubles as the producer-coverage sweep. This is
  the single point of failure for N independent reimplementations — coverage cannot
  depend on hand-written cases.

## Step sketch (NOT yet tickets — scope first)

1. **Producer-coverage spike** (cheap, highest-signal): push the full rewrite output
   (every supported function/cast) through the producer; catalog coverage gaps (unnest
   known-blocked). Output decides the IR choice (step 2).
2. **Decide + define the frozen inference IR**: Substrait-profile vs homegrown
   tiny-protobuf IR (both give free N-language deser via protoc). Pre-resolved, small
   primitive set, transforms as compositions.
3. **Serving artifact format**: plan doc + Parquet fitted tables referenced by id.
4. **Generative differential corpus + cross-language conformance harness**
   (foundational — before any runtime).
5. **Reference runtime** (Rust, one-of-N) + WASM target for JS.
6. **First native runtime** — pick by demand (likely Go or Node).
7. **Opaque-primitive contract**: tokenize / hash / COO-construct / sparse-expansion,
   per-runtime, corpus-gated.

## Sequencing (lazy, demand-driven)

Ship the IR/interchange spec (steps 1→2→3) + the corpus (step 4) + the Rust reference
runtime (step 5) **first**. Add each backend-language runtime (step 6) only when a real
workload needs that stack. The corpus + spec are the sunk infra that makes the Nth
runtime cheap; the runtimes themselves are demand-driven. **Do not commit to all four
languages upfront.**

## Open decisions (for scoping)

1. Substrait-profile vs homegrown IR — **spike-gated by step 1**.
2. First backend language — demand-driven.
3. How this sequences against transformer-refs / opaque-transform work: **the
   tfidf/multi-hot feature-output task (TASK-17) and the opaque-primitive step both lean
   on the opaque-transform mechanism** — a real dependency to order.

## Relationships

- Two-engine framing (native = one-of-N) is load-bearing here; open question, not
  ratified short-term.
- Shares the opaque-transform mechanism (decision-3) with the tfidf/multi-hot
  feature-output task (TASK-17).
- The conformance harness (doc-5, decision-6) is a precursor pattern for the generative
  corpus (step 4).
