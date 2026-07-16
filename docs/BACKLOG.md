# Backlog

Deferred work. When a task is pushed out of the current scope — from a spec, a
plan, a review, or a conversation — it lands here with enough context to pick up
cold later. This is the parking lot; [VISION.md](VISION.md) stays focused on what
the project is and how it works *today*, and [SQL_SUPPORT.md](SQL_SUPPORT.md)
tracks feature-by-feature support status.

Each item: what, why deferred, and where to start.

## Open items

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
the intermediate").

**Sequencing (decided 2026-07-16): Python-fallback-first, then optimize per
transformer.**
- **Phase A — compose-in surface, fallback-backed.** Stand up the whole structural
  surface end to end with every transformer backed by the **real sklearn object**
  (the Python fallback): the estimator interface, the `Pipeline`/`ColumnTransformer`
  glue, and the assembly-parity harness. Ships hooks 1 (composes in) + 3 (typed
  contract) and proves the *structure* is right — does ours actually slot into a
  stock pipeline, does column routing + concatenation yield a bit-identical vector —
  before any engine reimplementation. The fallback is trivially parity-correct (it
  *is* sklearn), so it doubles as the oracle for Phase B. Explicitly **not** a speed
  win (fallback is slower than raw sklearn — extra indirection + still builds the
  DataFrame); it's the correctness/structure milestone.
- **Phase B — native per transformer.** Replace each transformer's internals with a
  **native** engine implementation (SQLTransform/expression, no sklearn call), one at
  a time, each diffed against the fallback until bit-identical, then flipped over.
  Same public API throughout, so speed arrives transparently, transformer by
  transformer. Ordering within B: `StandardScaler` as the thin vertical (scalar
  state, no fan-out) to shake out the native path, `OneHotEncoder` next for the hard
  cases (fan-out + unknown-category).
- The **n = 1 serving-path optimization** (delete the dict/DataFrame, Rust-parse
  input, contiguous feature buffer) is the *separate* M3 item on top of Phase B's
  native transforms — not part of this item.

**Scope:**
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

**Deprioritized off the M1 critical path (2026-07-16).** The general UDF/UDAF/macro
registry isn't immediately useful — the concrete work (recursive composition, then
sklearn transformers) doesn't need it up front. Treat the spec as something to
*extract from* that concrete work once two contrasting transformer shapes exist,
not to write ahead of it. The notes below stay as captured design thinking.

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

### Rich (recursive) type system and UNNEST
**✅ First slice shipped** — on master (`4809470`). Recursive `Value`/`Base` spine,
struct + list types, schema-driven Python↔`Value` marshalling (nested output
models), `s.x` field access, `unnest(struct)`→columns, `unnest(list)`→rows
(`RelNode::Unnest`), struct equality + join-keys. +17 differential parity tests
(159→176), no regressions. **Live remaining work = the fast-follow types and the
deferred edges below — none block anything.**

Foundation for the composition output model, fan-out transformers, and the M4
feature contract. **Supersedes the narrower "Rust struct-support" ticket** — its
five bullets (struct `Value`, field access, output-model synthesis, DataFusion
parity) are subsumed here. Rather than bolt structs onto the closed scalar type
layer, replace that layer with a **recursive, extensible, schema-driven** one so
`InferFn` can carry the full pyarrow type surface. Spec:
[rich type system design](superpowers/specs/2026-07-16-rich-type-system-design.md).

**Why the pivot (2026-07-16):** composition needs structs; DataFusion has no
`struct.*` — it uses **`UNNEST`** (`unnest(struct)`→columns, `unnest(list)`→rows),
so we match that; and the engine should carry real feature-data types (also M4
feature-contract groundwork). Build the type *layer* properly, not one type.

**First slice** (reference semantics are DataFusion's throughout, differential-harness
enforced):
- Recursive `Value` (`Struct`/`List`, `src/expr.rs`) + recursive `Base`
  (`src/types.rs`) — a **structural** change touching every `match Base` arm
  (`compatible`, `field_type_to_python`, `arrow_type_to_base`, …); non-container
  regressions staying green is the main risk.
- Struct/list construction (`named_struct` / `[…]`) + `s.x` field access on aliased
  struct **columns** (not `(expr).field` — DataFusion rejects that) (`src/expr_build.rs`).
- `unnest(struct)` → columns: build-time projection expansion (cardinality-preserving).
- `unnest(list)` → rows: **the hard novel piece** — a cardinality change (1 row → N),
  modeled as a new `RelNode::Unnest` relational operator; empty/NULL list → 0 rows.
  Land it last; split to an immediate fast-follow if it proves large.
- Schema-driven Python↔`Value` marshalling both boundaries (dynamic in/out) +
  pyarrow struct/list schema reading.

**Fast-follows the spine enables (deferred, not this slice):** temporal
(timestamp/date), decimal, map, dictionary, binary — localized additions once the
recursive spine lands.
**Open items (in spec):** sqlparser `UNNEST` AST shape (function vs. dedicated
node); `unnest(list)` empty/NULL cardinality re-verify; struct field-order
round-trip; table-alias vs struct-field-name precedence in `s.x`.

**Deferred edges (rich-type slice) — all fail loud, none block:**
- **Ordered comparison (`<`,`>`,`<=`,`>=`) on structs/lists.** `=`/`!=` are
  implemented (structural equality, `src/expr.rs` `comparison`); DataFusion does
  **lexicographic ordering** for `<`/`>` on structs and lists
  (`named_struct('x',1) < named_struct('x',2)` → `true`; `[1,2] < [1,3]` → `true`),
  which we don't — a full fix needs real lexicographic `Ord` on
  `Value::Struct`/`List`. Clean runtime error today (`compare_values`' scalar-only
  `as_f64` fallback). Pick up if a real query needs ordered struct/list comparison.
- **Static-table struct/list values stay `Value::Object` (`src/lookup.rs`).** A
  struct/list column in a static lookup table isn't marshalled into the recursive
  `Value`, so a struct join-key against a static table **never matches**. Scalar
  keys unaffected.
- **Null-struct field access — accepted divergence.** Field access on a NULL struct
  yields InferFn `NULL` vs DataFusion's quirky `0`. Values-only parity is the bar
  and DataFusion's `0` is the odd one, so this is accepted, not a fix.
- **`unnest` naming edges.** `unnest(x) AS <existing col>` raises a spurious
  ambiguity error; an unaliased `unnest(list)` column is named `"unnest"`.
  Cosmetic/edge; revisit if it bites real queries.

### Compose SQLTransforms via `{transform}(col)` references — follow-up slices
**✅ First slice (frozen path) shipped** — on master (through `bb22526`).
`{a.transform}(col)` inlines a fitted transform's frozen scalar expression, fused
into one per-row expression with exact `transform`/`infer` differential parity;
the outer taking its own window aggregate over the inlined column works; a bare
`{a}` on a fitted object and `{a.transform}` on an unfit object both error
explicitly. Identifier handling locked to DataFusion-faithful verbatim quoting
(the earlier quoting gap in the inline + PARTITION BY paths is fixed, `c056ec3`).
**✅ Second slice (fit-cascade) shipped** — on master (`5ac613e`, suite 188).
An outer `SQLTransform` can reference an **unfit** `SQLTransform` via `{a}(col)`;
the ref's window-aggregate state is fit **into the composite** during the
composite's `.fit()` (sklearn-staged), with arbitrary nesting/chaining
(`{a}({b}(x))`), outer aggregates over the cascade, and free mixing with the frozen
path (`{a}({b.transform}(x))`). `transform`/`infer` parity holds across the matrix.
Single-output `{a}(col)` auto-unwraps to a scalar (consistent with the frozen path);
the ref is never mutated (clone contract). Design/decisions in the
[fit-cascade spec](superpowers/specs/2026-07-16-fit-cascade-composition-design.md).

**Live remaining work = the "Deferred to follow-up slices" list at the end of this
entry** — now just multi-output fan-out, multi-input refs, and unfit-*composite*
refs (all error explicitly today). They re-enter with the sklearn transformers that
need them (OneHot fan-out, multi-input encoders). Everything between here and the
deferred list is kept as the design reference those slices build on.

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

Deferred to follow-up slices (error explicitly today, not built):
- **Multi-output fan-out** referenced transforms (OneHot) — output naming/placement +
  column-count-from-state; unpacks via `unnest({a}(col))` on the shipped struct type.
- **Multi-input** referenced transforms — positional/named binding for
  `{transform}(a, b)`.
- **Unfit-*composite* references** — a `{a}(col)` where `a` is itself a composite
  (deeper recursion). Frozen-composite refs already work; unfit-composite is the
  deferred deeper case.
- **Minor, guarded** (found in fit-cascade review, `5ac613e`): a referenced
  transform whose *inner definition* has >1 distinct `PARTITION BY` set would collide
  on `fit_into_scope`'s single-scope state key — raised as a loud
  `NotImplementedError` (can't silently corrupt). Unreachable for this slice's
  single-in/out refs; revisit when partitioned or multi-input refs land.

### Error attribution — failure → authored SQL span → composite transformer part
As nested/composite transformers land, a runtime interpreter failure (div/mod by
zero, bad cast, unknown-category lookup miss, type error) points at a node in the
*fused* per-row expression — which has lost track of where it came from.
Composition fuses N transformers' rewrites into one flat expression over `__THIS__`
+ name-scoped `__STATE_R{i}__` states, and nesting (`{a}({b}(x))`) inlines
b→a→outer, so a failing node can originate several inlining layers deep. **Goal:**
attribute a failure back through the chain — *failing op → the span of the authored
SQL that produced it → the specific transformer (and, for a composite, which
referenced part at which nesting depth)*. Turn "ValueError: division by zero" into
something like "division by zero in `{scaler}` applied to `age` — from `x /
STDDEV(x) OVER ()` in the scaler's definition."

**Why:** composition makes failures un-attributable exactly when there are the most
places to look. This is the debuggability half of VISION hook 3 (provenance) — a
prediction/pipeline you can't locate a fault in isn't handoff-able. Now live as of
the fit-cascade slice (nesting exists), so this is worth doing next to any further
composition work.

**Readiness done (`5ac613e`), the feature is not.** The fit-cascade slice kept
*all* inlining centralized in `inline_references` — one choke point where origin
tags (referenced transformer / ref scope `__STATE_R{i}__` + authored SQL span) can
later thread through a single place, instead of scattering the work. That's the
cheap part, and it's landed.

**The remaining work needs Rust.** The composite's rewritten SQL reaches `InferFn`
as a **string**, so a build-time tag on the sqlglot AST does *not* survive to the
interpreter — the tag has to be propagated *through* the Rust engine and surfaced
back out on error. **Start:** thread the origin tag from `inline_references` into the
program the interpreter runs, carry it on the executing node, and render the
attributed location when the interpreter raises. Distinct from the "error-type
parity across engines" non-goal below — that's about *which exception type* each
engine raises; this is about *locating* a failure in the authored source, and
applies equally to both engines.

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

## Considered — likely won't do

### Codegen / compiled inference path
An older README roadmap item, superseded by the Rust `InferFn` interpreter. Keep
parked unless interpreter overhead becomes a *measured* bottleneck; revisit only
with a benchmark in hand.

### Unify batch vs inference error *types* — non-goal (accepted divergence)
`transform` (DataFusion) and `infer`/`infer_batch` (Rust `InferFn`) must return
identical *values* on the normal numeric path — that parity is non-negotiable and
the differential harness enforces it. But matching the **error type/message** each
engine raises is an explicit **non-goal** (decision 2026-07-16): the two engines
genuinely carry different failure information, and reconciling it buys nothing a
user relies on. Concretely, integer div/modulo-by-zero raising a clean `ValueError`
from the Rust path vs a raw DataFusion `Exception` ("DataFusion error: Arrow error:
Divide by zero error") from the batch path is **accepted by design**, not a gap to
close. (Was previously an open "unify error semantics" item; descoped here.)
Locked in by the positive test
`test_div_by_zero_raises_on_both_engines_with_different_error_types`, which asserts
both engines error on integer div/mod-by-zero with *different* hierarchies (Rust
infer → clean `ValueError`; DataFusion batch → its own non-`ValueError` `Exception`).
