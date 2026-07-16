# Roadmap

The ordered path from where we are to the near-term goal in [VISION.md](VISION.md):
**a well-adopted, compose-in sklearn transformer alternative for low-latency
serving.** Milestones are sequenced; the checkboxes are the progress bar.

**How this relates to the other docs — each fact lives in exactly one place:**
- [VISION.md](VISION.md) — the destination (why / what), task-free.
- **This file** — ordering, milestones, and progress. Each checklist item is a
  *label that links to a [BACKLOG.md](BACKLOG.md) item*; it does not restate scope.
- [BACKLOG.md](BACKLOG.md) — the source of truth for each task's detail.
- [SQL_SUPPORT.md](SQL_SUPPORT.md) — the per-feature SQL support matrix.

When a task completes: tick its box here **and** remove/archive the BACKLOG item —
one motion, so the two never drift.

Legend: `[x]` done · `[ ]` todo.

## Near-term track — toward the serving goal

### M0 — Differential test harness ✅
The parity oracle the rest of the track leans on: `transform` (DataFusion) and
`infer` (Rust `InferFn`) proven to agree across the expression/join surface, so a
regression in either engine is caught mechanically. Done — tasks 1–4 merged. One
real bug it surfaced is tracked under Maintenance below.

### M1 — Transformer foundation & sklearn parity
*Serves VISION hook 1 (composes in) + the correctness bar of hook 2.* Get
sklearn-compatible transformers composing into a stock pipeline and producing
bit-identical output. Correctness and coverage first; speed is M3.
- [ ] [Transformer execution model — UDF/UDAF, macros, composition](BACKLOG.md#transformer-execution-model--procedures-udfudaf-macros-composition) — the conceptual backbone the rest builds on
- [ ] [First slice: compose SQLTransforms via `{transform}(col)` references](BACKLOG.md#first-slice-compose-sqltransforms-via-transformcol-references) — t-string prerequisite cleared (Python floor now **3.14** ✅); ready to brainstorm/spec
- [ ] [sklearn integration — functionality & parity](BACKLOG.md#sklearn-transformer-integration--functionality--parity) — transformers + `Pipeline`/`ColumnTransformer` glue + assembly-parity harness + Python fallback

### M2 — Benchmark baseline
*Gate before any optimization.* Stand up the measurement harness — single-row
latency *distribution* + batch throughput, single- and multi-threaded — so the M3
optimizations are chosen by evidence, not hunch.
- [ ] [Benchmark inference-path optimizations before building them](BACKLOG.md#benchmark-inference-path-optimizations-before-building-them)

### M3 — Rust-optimized serving path
*Serves VISION hook 2 (fast at n = 1).* Delete the dict/DataFrame from the request
path: parse in Rust, native transforms, a single contiguous feature buffer into
`model.predict`. Depends on M1 (parity net) and M2 (measure first).
- [ ] [Rust-optimized serving inference path](BACKLOG.md#rust-optimized-serving-inference-path)

### M4 — Feature-contract surface
*Serves VISION hook 3 (the moat).* Emit the typed, validated, provenance-carrying
output schema — a mismatch errors at the boundary, and every feature traces back to
the raw column that produced it.
- [ ] Typed / validated / provenance feature contract — *(no BACKLOG item yet; scope once M1 lands)*

## Maintenance & correctness
Quality items not on the critical path to the goal, but worth clearing.
- [ ] [Unify batch vs inference error semantics](BACKLOG.md#unify-batch-vs-inference-error-semantics)
- [ ] [LEFT lookup-join output nullability bug](BACKLOG.md#left-lookup-join-output-nullability-found-by-the-differential-harness) — surfaced by M0's harness

## Later / parked
- [ ] [`CASE WHEN` + outer joins in authored SQL](BACKLOG.md#case-when-and-outer-joins-in-authored-sql) — SQL surface; prioritize by what authoring actually needs
- [ ] [`ORDER BY` / window frames](BACKLOG.md#order-by--window-frames-running-cumulative-moving-aggregates) — research spike; may not fit a row-at-a-time inference engine
- **Feature-store expansion** — the post-adoption goal; M4's contract is its groundwork, so it's a natural next step rather than a pivot.
- ~~Codegen / compiled inference path~~ — considered, likely won't do (superseded by the `InferFn` interpreter).
