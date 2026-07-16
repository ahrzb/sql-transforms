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

### sklearn transformer integration — functionality & parity
Ship sklearn-compatible transformers that **compose into** a user's existing
sklearn pipeline — implement the estimator interface (`fit`/`transform`/
`get_feature_names_out`, `get_params`/`set_params`, cloneable) so ours are
first-class citizens inside a stock `Pipeline`/`ColumnTransformer`, mixing with
sklearn's own transformers one at a time, and produce output that matches sklearn
exactly. This item is about *correctness and coverage*, not speed: the simplest
implementation that's bit-identical wins, even if it isn't yet the zero-copy path.
It delivers the parity harness the optimized path (below) is validated against, and
is shippable on its own — correct-but-not-yet-fast (via Python fallback) is a real
milestone that de-risks the semantics. Supersedes the old README `sklearn.*`
surface and the earlier "decide in/out of scope" question — it's in scope as the
primary serving goal (see [VISION.md](VISION.md), "Positioning" + "Serving without
the intermediate"). **Scope:**
- **Two integration directions, compose-first:** (a) *compose* — our transformers
  are sklearn estimators the user drops into their own `Pipeline`/`ColumnTransformer`
  (primary; the incremental, low-friction adoption path, one transformer at a time,
  coexisting with sklearn); (b) *consume* — accelerate a whole already-fitted
  sklearn pipeline handed to us (secondary). Estimator-interface compliance is what
  makes (a) work and is the gating requirement.
- Transformer coverage, ranked by "what a served request touches" (not raw
  popularity): `SimpleImputer` + `StandardScaler` (numeric) and `OrdinalEncoder` +
  `OneHotEncoder` (categorical, co-first — target audience is mixed
  numeric/categorical, incl. recommendation with high-cardinality IDs). Other
  scalers (MinMax/Robust/MaxAbs) are near-free follow-ons once one scaler exists;
  `TargetEncoder` close behind for high-card categoricals.
- The real unlock is the structural glue, not the leaves: `Pipeline` (sequencing)
  and `ColumnTransformer` (column routing + output concatenation). Build these
  alongside the first leaves — bare transformers can't run a realistic pipeline.
- Unknown-category handling is a *designed-in* requirement, not a flag: cold-start
  unseen IDs are the common case in serving/recommendation, not an edge case.
  Match sklearn's `handle_unknown` / `drop` / infrequent-category semantics
  exactly.
- **Acceptance test = end-to-end assembly parity**: the full feature vector (width
  + column order + values) must be bit-identical to
  `ColumnTransformer.transform()`, because the downstream model consumes it
  positionally and a mislabeled column is a *silent* wrong prediction. Per-
  transformer correctness in isolation is not sufficient. This harness is the
  item's main deliverable — it's also the oracle the optimized path is tested
  against.
- First-class **Python fallback** per transformer (run the real sklearn object) so
  partial coverage ships and the native surface can grow incrementally. Note
  fallback is not free at serving time — it drags the DataFrame back onto the
  request path (see the Rust-optimized item + benchmark item).
- Open sub-question carried over: whether/how the SQL authoring surface
  (`sklearn.standardize(col)`-style, goal 1) maps onto this. Both integration
  directions work with fitted sklearn-estimator objects; the SQL authoring
  front-end is a separate question.

### Transformer execution model — procedures (UDF/UDAF), macros, composition
The conceptual backbone the two items above build on (from a 2026-07-15 design
discussion). Capture, not yet a spec.

- **A transformer = UDAF(s) + one UDF, never a single primitive.** A UDAF is
  `N→1` (aggregate training rows → *state*); a transform is `N→N` (per-row). So a
  UDAF is only ever the *fit* half. `StandardScaler` = UDAF{`mean`,`stddev`} → state
  + UDF{`(x-mean)/scale`}. The tell that something needs both: the output mixes an
  aggregate with the raw row value — which is precisely a **window aggregate**, e.g.
  `StandardScaler` ≡ `(x - AVG(x) OVER ()) / STDDEV(x) OVER ()`, exactly the shape
  the existing `fit`→state→rewrite pipeline already compiles. The interesting
  variable is the **state shape** the UDAF emits: scalar (scaler/imputer), a *list*
  (OneHot categories → `array_agg`/`distinct` UDAF), a *code-map* (Ordinal), or a
  *per-group table* (TargetEncoder = the shipped `PARTITION BY`).
- **Transformers are macros over the window-agg/scalar SQL surface.** *Static*
  macros expand to fixed SQL regardless of data (scaler, imputer). *Fit-parameterized*
  macros expand only after their UDAF runs — `onehot(x)` becomes one
  `CAST(x = 'cat_i' AS INT)` per learned category; `ordinal(x)` needs the code-map.
  Most of the numeric library is one-line macro definitions, not new engine code.
- **Procedure registry = UDFs + UDAFs, each SQL-defined or Rust-built-in, one
  contract.** SQL-defined-now / Rust-built-in-later is per-primitive and matches the
  functionality-vs-optimized split (promote a hot UDF to Rust without touching the
  transformers using it; the SQL definition stays as its parity oracle). Placement
  in *our two engines*: **UDAFs are fit-time only → register as DataFusion
  UDAF/UDWF** (DataFusion 54 supports `udf`/`udaf`/`udwf`, incl. Rust-backed via
  PyCapsule); one impl, done. **UDFs** are either **SQL-expressible** (arithmetic,
  `COALESCE`, `=`, `CAST` — runs on *both* `transform` (DataFusion) and `infer`
  (Rust `InferFn`) for free) or a **genuinely new scalar op** needing a *dual* impl
  (InferFn Rust built-in **and** a DataFusion UDF for batch), kept in lockstep by the
  differential harness. First cut should stay entirely in the SQL-expressible +
  DataFusion-UDAF lane (no new InferFn Rust ops).
- **Composition: fuse at inference, stage at fit.** A fitted SQLTransformer used
  inside another (== a `Pipeline` step) merges by *expression inlining over frozen
  state*: post-fit every transformer's rewrite is a scalar expression over
  `__THIS__` + frozen `__STATE__`, so nesting substitutes the inner's expression
  into the outer → **one fused per-row expression, single `InferFn` pass, no
  intermediate materialized** (the serving thesis, end-to-end; a 5-stage pipeline
  collapses to one expression at `n=1`). **Fit cannot flatten** — the outer's
  aggregates are over the inner's *transformed* output, and flattening would need a
  window aggregate inside another window aggregate's argument (illegal SQL), so fit
  cascades like sklearn `fit_transform`: fit stage → transform training forward →
  fit next. Mechanical requirement: name-scope each stage's `__STATE__` tables so
  they don't collide when inlined.
- **Respect `.transform` vs `.fit_transform`.** `fit`/`fit_transform` = the staged
  cascade (unfrozen stages fit on the running transformed training data); `transform`
  = frozen fused application, no fitting. Both must be bit-identical to the equivalent
  sklearn Pipeline.
- **Frozen state reuse.** State present → the transformer is frozen (transform-only
  in a cascade); absent → it fits. Composition never silently re-fits a fitted
  component (enables pretrained/shared encoders reused across pipelines). Caveat: a
  *stock* sklearn `Pipeline.fit` clones + re-fits every step (estimator contract), so
  reusing pre-fit state inside sklearn's own Pipeline needs the frozen-estimator
  mechanism (sklearn 1.6 `FrozenEstimator`, or our `frozen=True` no-op `fit`); within
  our own `Pipeline`/`ColumnTransformer` equivalents we honor it directly.
- **Open, for the spec:** MVP slice (candidate: StandardScaler + OneHotEncoder +
  our Pipeline + parity harness — smallest slice hitting fan-out, unknown-category,
  and fuse/stage composition); the concrete UDF and UDAF signatures (input cols +
  params → state schema; input cols + state → output feature names + values) that the
  registry, parity harness, and both impls agree on; and whether a SQL-defined UDF is
  a raw `:param` template string or a structured InferFn-AST builder.

### Compose SQLTransforms via `{transform}(col)` references — follow-up slices
**✅ First slice (frozen path) shipped** — on master (through `bb22526`).
`{a.transform}(col)` inlines a fitted transform's frozen scalar expression, fused
into one per-row expression with exact `transform`/`infer` differential parity;
the outer taking its own window aggregate over the inlined column works; a bare
`{a}` on a fitted object and `{a.transform}` on an unfit object both error
explicitly. Identifier handling locked to DataFusion-faithful verbatim quoting —
but with a known gap now tracked separately (see "Identifier quoting not preserved
…" below). **The live remaining work is the "Deferred to follow-up slices" list at
the end of this entry** (fit-cascade, fan-out, multi-input); everything between
here and there is kept as the design reference those slices build on.

The first implementable step of the execution model above, and the primitive
everything else (our `Pipeline`, sklearn composition) is built on: let one
`SQLTransform` reference **another `SQLTransform` object** inside its SQL, applied to
a column, and combine the two correctly. Target syntax — a template/t-string where an
embedded transform is invoked on a column:
`SQLTransform(t"SELECT {scaler}(age) AS age_scaled FROM __THIS__")`, with `scaler` a
`SQLTransform` interpolated in. `{scaler}(age)` = apply `scaler`'s transform to column
`age`.

**Reference forms encode fit intent (the API's key decision):**
- **`{a}(col)`** — composes `a` as a *fittable* step; `a` participates in the outer's
  `fit_transform` cascade. **Errors if `a` is already fitted** — a bare reference to a
  fitted object is ambiguous (reuse its state, or re-fit it?), so force the user to be
  explicit. This is the fit-cascade path.
- **`{a.transform}(col)`** — uses `a`'s **frozen** transform; **no fitting happens**
  (errors if `a` is *not* fitted). The `.transform` at the call site makes "no
  fitting" unmissable. This is the frozen-reuse path.

**First-slice scope = the frozen path (`{a.transform}`) only.** It's dramatically
cheaper: a frozen inner's window aggregates are already `__STATE__` constants, so
`{a.transform}(col)` inlines to a **plain scalar** expression (no live window
function). The outer then fits + rewrites as a normal `SQLTransform` in **one flat
pass** — even the outer's own aggregates over the inner output
(`AVG({a.transform}(age)) OVER ()`) are legal flat SQL, because there's no nested
window aggregate. No staging, no cascade, no training-transform passes. `{a}`
(fit-cascade) is designed into the syntax now but implemented in the next slice; in
this slice a bare `{a}` raises "fit-cascade composition not yet implemented — fit `a`
and use `{a.transform}`".

Mechanics for the frozen path:
- **Arity — single-input, single-output referenced transforms only.** `{a.transform}
  (col) AS name` maps one input column to one output column (scaler / imputer shape).
  Multi-output *fan-out* (OneHot → N cols; needs output-naming/placement +
  column-count-from-state) and multi-input transforms are **deferred to a follow-up
  slice**. This is the smallest thing that proves inline + remap + frozen state-merge
  + outer-fit-flat.
- **Input remapping:** the referenced transform reads exactly one `__THIS__` column;
  `(age)` substitutes the outer's `age` for that input column throughout `a`'s frozen
  expression.
- **Inline:** substitute `a`'s frozen rewritten scalar expression for the reference,
  remapped to `col`.
- **State merge:** union `a`'s `__STATE__` tables into the outer's, **name-scoped**
  per referenced transform so they don't collide (e.g. `__STATE__@a`).
- Honors `.transform` vs `.fit_transform` end-to-end.

**First-slice done =** a fitted `scaler` (single-in/out `SQLTransform`) referenced as
`{scaler.transform}(age)` inside an outer `SQLTransform` — including the outer taking
its own aggregate over the inlined column (`… / AVG({scaler.transform}(age)) OVER ()`)
— fits + transforms/infers correctly, bit-identical between `transform` (DataFusion)
and `infer` (Rust); a bare `{scaler}` on a fitted object raises the explicit
fit-cascade-not-implemented error; and `{scaler.transform}` on an *unfit* object errors.

Open (this slice):
- **API surface — t-string (gate RESOLVED):** the Python floor is now **3.14**
  (`chore: bump Python floor to 3.14`), so PEP 750 t-strings are available natively.
  The bump was verified clean: builds on `abi3-py314`, full suite green on 3.14.6, and
  the one real 3.14 incompatibility — `typing.Union` became a class, breaking
  `call_method1("__getitem__", …)` — is fixed in `src/schema.rs` (`get_item`). No CI
  matrix exists to gate. This unblocks the intended surface: a t-string doesn't eval
  to a `str` — it produces a `Template` exposing literal parts and interpolations
  *separately*, so an embedded `SQLTransform` arrives as the **real object**, not a
  stringified repr to parse back out, making `{scaler}(age)` a genuine structural
  hand-off. Residual API question: the concrete `SQLTransform(t"…")` constructor shape
  (accept a `Template`, walk its interpolations to bind each embedded transform).
- **Reference mechanism:** embed by Python object (t-string interpolation, the
  intended form) vs by name in a registry — confirm object-embedding is the surface.
- **`__STATE__` name-scoping token:** the concrete collision-safe naming for merged
  state tables (`__STATE__@a` is illustrative) and how the rewrite/validation refer
  to it.

**Referenced transformers are definitions, never mutated (both forms).** The
composite owns *all* fitted state; a reference is always **read-only on `a`**:
`{a.transform}` reads `a`'s existing frozen state; `{a}` reads `a`'s *definition* and
fits it fresh **into the composite's own name-scoped state** (`__STATE__@a`), leaving
`a` untouched (still unfit afterward). This is sklearn's clone contract — `Pipeline.fit`
clones each step and fits the clone, never the original — so the same `a` can be
referenced by many composites without interference, and fitting one composite never
leaks state into `a` or into another composite.

Deferred to follow-up slices (designed-around now, not built):
- **Fit-cascade (`{a}(col)` on an unfit transform)** — the staged fit + nested-window
  problem (an inner still-live `OVER (...)` can't inline into another window agg's
  argument; fit must stage: fit inner → transform training forward → fit outer),
  writing the learned state into the *composite* (not back to `a`, per above). The
  frozen path avoids the staging entirely.
- **Multi-output fan-out** referenced transforms (OneHot) — output naming/placement +
  column-count-from-state.
- **Multi-input** referenced transforms — positional/named binding for
  `{transform}(a, b)`.

### Identifier quoting not preserved in composition inline + PARTITION BY joins
Two spots still emit column identifiers **unquoted**, breaking case-sensitive
(quoted) column names — the same class Task 3 fixed for the SELECT-list rewrite,
where the rule is **preserve the user's quoting verbatim** (DataFusion folds
unquoted → lowercase, quoted → exact):
- **Composition inline** (`sql_transform/_compose.py`): `inline_references`'s
  `rewrite()` rebuilds columns with `exp.column(col, table="__THIS__")` and
  `_single_col_arg` returns the bare `.name`, both dropping the `quoted` flag. So
  composing over a quoted mixed-case column fails at fit:
  `SQLTransform(t'SELECT {scaler.transform}("Age") AS s FROM __THIS__').fit(...)` →
  `ValueError: No field named __this__.age`. Fails loudly (never silently wrong),
  and only on the quoted/capitalized-column edge — all current composition tests
  use lowercase `age`, so it shipped green.
- **PARTITION BY join** (`sql_transform/_rewrite.py:69`): the join condition
  `f"__THIS__.{c} = {table}.{c}"` renders partition keys unquoted. Not reachable
  via composition today (partitioned inners are rejected as multi-input), but a
  case-sensitive `PARTITION BY` column would hit it.

**Start:** carry the column's `.quoted` flag through both spots exactly as
`_rewrite.py:57-62` already does for the SELECT list — build the column with
separate `this`/`table` identifiers, the column preserving `<node>.this.quoted`
and the `__THIS__`/state-table qualifier left unquoted; have `_single_col_arg`
return the arg `exp.Column` node so the remap keeps the call-site column's
quoting. Add a differential composition test over a capitalized column to close
the coverage hole. One identifier-verbatim fix covers both spots.

### Rust-optimized serving inference path
Make the preprocessing above *fast at n = 1*: keep the dict/DataFrame off the
request path entirely, parse request JSON in Rust into typed values, run native
(non-fallback) transforms, and hand `model.predict` a single contiguous feature
buffer (near-zero-copy numpy view) with no per-feature Python objects on either
boundary. This is the payoff behind the serving thesis — the functionality item
proves the vector is *right*; this one makes it *fast*. **Depends on:** the
functionality & parity item above (needs the parity harness as a correctness net)
and the benchmark item below (measure before optimizing). **Why separate:**
correctness and representation-performance are different risks and sequence
differently. **Scope:**
- Native implementations of the hot-path transformers operating on the Rust value
  representation — no Python/pandas intermediate anywhere on the request path.
- Contiguous feature-buffer output via the buffer protocol; contiguous typed input
  parsed in Rust. Both boundaries stay object-free.
- Which specific low-level tactics to apply (thread-local scratch arena, GIL
  release/threshold, JSON parser choice) and in what order is **gated by the
  benchmark item** — deletion of the DataFrame is the primary win; the arena and
  GIL work are second-order and only justified if a profile puts them on the
  critical path. Don't pre-commit to them here.

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

### Benchmark inference-path optimizations before building them
Several candidate optimizations for the online-inference path are currently
*hunches*, not measured wins: a thread-local bump arena for per-row scratch;
extracting Python values into an owned Rust type and releasing the GIL
(`allow_threads`) during compute; and — the load-bearing one — parsing request
JSON in Rust so the dict/DataFrame never touches the request path (see
[VISION.md](VISION.md), "Serving without the intermediate"). **Why deferred:**
building all three and attributing wins afterward is backwards. Each targets a
different cost, and two are probably aimed at the wrong path — the arena and
GIL-release mostly help the *batch/throughput* path, not the single-object
latency path this project optimizes for. Measure first; don't commit effort to an
optimization until a baseline shows it's the bottleneck. **Start:** stand up a
baseline harness before any of this lands, capturing the four corners —
single-row latency *distribution* (p50/p99, not mean) and batch throughput
(rows/sec), each single- and multi-threaded. Known traps to design the harness
around:
- GIL-reacquire contention only appears with *concurrent* callers; a
  single-threaded microbenchmark reports the handoff as free and misleads.
- For the latency path, profile one `infer(row)` call first — expected top costs
  are the JSON/dict boundary and any per-call setup (program lowering, param
  extraction) not amortized to construction, *not* allocation. Confirm or kill
  that before touching the arena.
- Releasing the GIL for a single tiny object is expected to *hurt* p99 (handoff >
  compute); if pursued, gate it behind a batch-size threshold, and treat
  process-level parallelism / predict-side GIL release as the likelier
  concurrency answer.

### LEFT lookup join output nullability (found by the differential harness)
A raw `InferFn` `LEFT JOIN` onto a static table whose value columns are declared
**non-nullable** raises a pydantic `ValidationError` on an unmatched key instead of
returning NULL. The row-level executor already produces the correct `NULL`; only
the synthesized *output model* is wrong — `resolve_tables` in `src/plan.rs` drops
the `outer` flag (via `..`) when building `effective_schemas`, so
`src/types.rs::resolve_column_type` types the LEFT-joined column non-nullable and
`OutputRow(**row)` rejects the runtime `None`. Masked in the `SQLTransform`
PARTITION BY path (which widens state columns to nullable in `_state.py`) and in
`test_interpreter.py::test_left_lookup_join_miss_returns_null` (bare `pa.table`
defaults to nullable) — the differential harness's `static({"y": "int"})` builds
`nullable=False`, exposing it. Directly blocks `OrdinalEncoder` unknown-category
handling (unseen category = a LEFT-lookup miss). **Start:** widen an outer join's
lookup-side columns to nullable when computing `effective_schemas`. Tracked by the
`strict=True` xfail `tests/test_diff_relational.py::test_left_lookup_join_hit_and_miss`,
which flips to a pass (forcing the decorator's removal) when fixed.

## Considered — likely won't do

### Codegen / compiled inference path
An older README roadmap item, superseded by the Rust `InferFn` interpreter. Keep
parked unless interpreter overhead becomes a *measured* bottleneck; revisit only
with a benchmark in hand.
