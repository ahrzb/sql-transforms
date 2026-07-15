# Backlog

Deferred work. When a task is pushed out of the current scope — from a spec, a
plan, a review, or a conversation — it lands here with enough context to pick up
cold later. This is the parking lot; [VISION.md](VISION.md) stays focused on what
the project is and how it works *today*, and [SQL_SUPPORT.md](SQL_SUPPORT.md)
tracks feature-by-feature support status.

Each item: what, why deferred, and where to start.

## Open items

### Unify batch vs inference error semantics
`transform` (DataFusion) and `infer`/`infer_batch` (Rust `InferFn`) return
identical values on the normal numeric path, but an integer division/modulo by
zero raises a clean `ValueError` from the Rust path and a raw DataFusion
`Exception` ("DataFusion error: Arrow error: Divide by zero error") from the batch
path. Tracked by the strict-`xfail` test
`test_transform_raises_clean_valueerror_on_div_by_zero`. **Start:** catch
DataFusion's error in `_batch.run_batch` and re-raise the same clean `ValueError`
the interpreter raises; the xfail flips to a pass when done.

### sklearn-style transforms
Bring sklearn-style transforms (scaling, encoding, binning) onto the current
fit/state/`InferFn` pipeline — or decide they're out of scope for v0. The README
still advertises a `sklearn.*` function surface that predates the Rust rewrite and
is not wired into the current pipeline. **Start:** decide in/out of scope against
the two project goals (see [../MEMORY.md](../MEMORY.md) / project goal); if in,
design how a `sklearn.standardize(col)`-style call maps onto fit-time state + the
rewrite.

### `CASE WHEN` and outer joins in authored SQL
Decide if/how `CASE WHEN` and `LEFT`/`RIGHT`/`FULL OUTER` joins matter for real
feature-engineering SQL before investing — neither is supported in the Rust
interpreter today. **Start:** prioritize by what authoring (goal 1) actually
needs; `CASE WHEN` also needs Layer-1 interpreter support, not just a rewrite
change.

### `ORDER BY` / window frames (running, cumulative, moving aggregates)
`AGG(col) OVER (ORDER BY ...)` and explicit `ROWS`/`RANGE BETWEEN` frames —
running sums, cumulative means, moving windows. Currently rejected with
`NotImplementedError` (`WindowAgg.has_order`). **Fundamentally harder:** these are
order-dependent and stateful across rows, so they do not fit the "freeze a value
at fit, broadcast at inference" model that `OVER ()` and `PARTITION BY` share —
inference would need streaming/sequence state. **Start:** treat as a research
spike, not a small feature; decide whether it's even in scope for a row-at-a-time
inference engine before investing.

## In progress

### `PARTITION BY` window aggregates
Per-partition learned state (target/categorical encoding) via LEFT-joined
unique-keyed state tables; unseen partition → NULL; transform stays strictly
1-to-1. Includes a Rust LEFT-lookup-join addition, and **folds in aggregate result
typing** (state value columns keep real Arrow types — int/float/str/bool, nullable
— instead of float coercion; enables integer count/ordinal encodings). Designed in
[superpowers/specs/2026-07-15-partition-by-design.md](superpowers/specs/2026-07-15-partition-by-design.md)
— move back to Open items only if it's shelved.

## Considered — likely won't do

### Codegen / compiled inference path
An older README roadmap item, superseded by the Rust `InferFn` interpreter. Keep
parked unless interpreter overhead becomes a *measured* bottleneck; revisit only
with a benchmark in hand.
