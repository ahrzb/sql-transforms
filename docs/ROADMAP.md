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
regression in either engine is caught mechanically. Done — tasks 1–4 merged. The
one real bug it surfaced (LEFT-join nullability) is now fixed (see M1 below).

### M1 — Transformer foundation & sklearn parity
*Serves VISION hook 1 (composes in) + the correctness bar of hook 2.* Get
sklearn-compatible transformers composing into a stock pipeline and producing
bit-identical output. Correctness and coverage first; speed is M3.

**Ordering (decided 2026-07-16):** outstanding bugs → **recursive (fit-cascade)
composition** of unfitted `SQLTransform`s → sklearn transformers. Rationale: the
recursive composition primitive is exactly what stock sklearn `Pipeline.fit`
needs (it clones + re-fits each step), so building it first de-risks the sklearn
work that sits on top. The general **UDF/UDAF/macro execution-model spec is *not*
on the critical path** — it isn't immediately useful; extract it from the concrete
composition + sklearn work later, if/when it's actually needed.

Done — foundation:
- [x] [First slice: frozen composition `{a.transform}(col)`](BACKLOG.md#compose-sqltransforms-via-transformcol-references--follow-up-slices) — shipped on master; the frozen-reuse primitive the recursive path extends.
- [x] LEFT lookup-join output nullability bug — **fixed & merged** (`7cb7d3c`): threaded outer-nullability through `resolve_tables`, widening the outer side's columns to nullable in `effective_schemas`. Unblocks `OrdinalEncoder` unknown-category handling; the `strict` xfail retired to a normal passing regression test.
- [x] Identifier-quoting bug (composition inline + PARTITION BY joins) — **fixed & merged** (`c056ec3`): carried the `.quoted` flag through the composition inline + remap, and quoted the PARTITION BY join keys to match the quoted GROUP BY. Two regression tests proving `transform`==`infer` parity on quoted columns.
- [x] [Rich (recursive) type system and UNNEST](BACKLOG.md#rich-recursive-type-system-and-unnest) — **first slice shipped** (`4809470`): recursive `Value`/`Base` spine, struct + list, `UNNEST` (struct→columns, list→rows via `RelNode::Unnest`), schema-driven marshalling, struct equality/join-keys. +17 differential parity tests (159→176). Fast-follow types (temporal/decimal/map/…) + deferred edges tracked in BACKLOG, non-blocking. Also M4 feature-contract groundwork.
- [x] [Recursive (fit-cascade) composition — unfitted `{a}(col)`](BACKLOG.md#compose-sqltransforms-via-transformcol-references--follow-up-slices) — **shipped** (`5ac613e`, suite 188): outer `SQLTransform` references an unfit `SQLTransform` via `{a}(col)`, whose window-state fits into the composite during `.fit()` (sklearn-staged); arbitrary nesting/chaining `{a}({b}(x))`, outer aggregates over the cascade, free mixing with the frozen path; single-output auto-unwraps to scalar; clone contract (refs never mutated). `transform`/`infer` parity across the matrix. Multi-output fan-out / multi-input / unfit-composite refs still deferred (error explicitly). This is the primitive sklearn `Pipeline` composition builds on.

Active — in order:
1. [ ] [sklearn integration — functionality & parity](BACKLOG.md#sklearn-transformer-integration--functionality--parity) — **fallback-first (decided 2026-07-16)**, built on the recursive composition now shipped. Phase A (below) stands up the compose-in surface with real-sklearn-fallback internals; **Phase B** then swaps each transformer to a native engine implementation, one at a time, diffed against the fallback oracle, in **transformer-tier order** (see [SKLEARN_TRANSFORMERS.md](SKLEARN_TRANSFORMERS.md): Tier 0 `StandardScaler`/`SimpleImputer`/`OrdinalEncoder`/`OneHotEncoder` → Tier 1 scalers + `TargetEncoder` → Tier 2). Most Tier 0/1 map onto already-shipped engine machinery (window aggs, `PARTITION BY`, `LookupJoin`, struct/`UNNEST`). n = 1 serving-path speed is the separate M3.

   **Phase A slices (decided 2026-07-16):**
   1. [ ] **A1 — thin vertical.** Estimator interface (`fit`/`transform`/`get_feature_names_out`/`get_params`/`set_params`/cloneable) on ONE transformer (`StandardScaler`), internals = real sklearn fallback, driven end-to-end through a stock sklearn `Pipeline`. **Stands up the [per-transformer differential parity harness](BACKLOG.md#per-transformer-differential-parity-harness)** (StandardScaler through the param + edge-case matrix). Ships hooks 1+3 on its own. Designs the `get_feature_names_out`/provenance **contract shape** (the hook-3 surface) — joint look before it hardens. NB: A1 is single-transformer parity; the full *assembly* oracle lands in A2.
   2. [ ] **A2 — `ColumnTransformer` glue.** Column routing + horizontal concat; the real end-to-end **assembly-parity** oracle — bit-identical width + column order + values vs stock `ColumnTransformer.transform()`. **Must include a multi-output transformer** (OneHot fallback is the natural one) so variable-width concat + feature-name expansion is actually exercised — single-output-only would pass on a degenerate case and hide the offset/naming risk. Provenance/feature-names must survive routing + concat in order.
   3. [ ] **A3 — breadth.** Remaining transformers fallback-backed; `get_feature_names_out` provenance uniform across all; sets up Phase B's per-transformer native swaps. (Unknown-category stays sklearn's job in fallback mode; native handling is Phase B.)

Deferred off the critical path:
- [ ] [Transformer execution model — UDF/UDAF, macros](BACKLOG.md#transformer-execution-model--procedures-udfudaf-macros-composition) — the "conceptual backbone," but not immediately useful; spec to be *extracted from* the concrete work above rather than written up front.

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

## Later / parked
- [ ] [`CASE WHEN` + outer joins in authored SQL](BACKLOG.md#case-when-and-outer-joins-in-authored-sql) — SQL surface; prioritize by what authoring actually needs
- [ ] [`ORDER BY` / window frames](BACKLOG.md#order-by--window-frames-running-cumulative-moving-aggregates) — research spike; may not fit a row-at-a-time inference engine
- **Feature-store expansion** — the post-adoption goal; M4's contract is its groundwork, so it's a natural next step rather than a pivot.
- ~~Unify batch vs inference error semantics~~ — **won't do**: error-type parity across engines is a non-goal; only output *values* must match. Div/mod-by-zero raising a clean `ValueError` (Rust) vs a raw DataFusion `Exception` (batch) is accepted by design. See BACKLOG.
- ~~Codegen / compiled inference path~~ — considered, likely won't do (superseded by the `InferFn` interpreter).
