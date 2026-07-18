# Backlog

Deferred work. When a task is pushed out of the current scope — from a spec, a
plan, a review, or a conversation — it lands here with enough context to pick up
cold later. This is the parking lot; [Vision](<../backlog/docs/doc-3 - Vision.md>) stays focused on what
the project is and how it works *today*, and [SQL_SUPPORT.md](SQL_SUPPORT.md)
tracks feature-by-feature support status.

Each item: what, why deferred, and where to start.

> **Two homes, no overlap (2026-07-18).** Actionable, pickup-able tasks now live in
> **Backlog.md** (`backlog/tasks/*.md`; `backlog board` / `backlog task list`, or the
> `backlog` MCP server). **This prose file stays the reasoning archive** — the *why*,
> the decision history, verified source citations, and deferred/strategic context that
> doesn't fit an atomic task. Rule: **actionable work → Backlog.md; context & decisions
> → here.** Tasks link back with `--ref docs/BACKLOG.md`; don't duplicate detail across
> the two. Seeded tasks: TASK-1 (native parity bugs), TASK-2/3 (opaque-transform
> follow-ups), TASK-4 (codegen bugs), TASK-5/6 (native-swap + assembly).

## Open items

### Native engine (`InferFn`) parity bugs — `transform` ≠ `infer`
> **✅ RESOLVED (2026-07-18) — all 10 fixed & merged** (`rust-parity-bugs` → `b1a10bf`;
> suite 211, 0 xfailed). Each was pinned first as a strict xfail-on-rust in
> `tests/test_diff_rust_bugs.py` (now on master, repros confirmed against **live**
> DataFusion), then fixed to match the oracle and the marker flipped. Tracking: **TASK-1
> (Done)**. **One residual sub-case remains** — the codegen merge surfaced that the #1
> float-display fix missed the `[1e-5, 1e-4)` band (native `'1e-5'` vs DF `'0.00001'`),
> pinned as `test_float_display_small_decimal_band` → **TASK-7**. Minor/unspecified:
> `SUBSTR` with a *negative length* (DF unprobed; impl returns `''`) — no divergence
> surfaced, flagged for completeness. Detail below retained as the historical record.
Five divergences (reported 2026-07-17) where the native `infer` path disagrees with
the DataFusion `transform` path on the same input — each **violates the README's
core promise that the two return identical values** (a user gets a different answer
at serving time than at batch time). Found while validating assumptions for the
codegen engine; **pre-existing and unrelated to codegen**. They survive because the
differential harness only pins the surface it *covers* — every one sits in a
coverage gap, which is why the suite is green. **Causes below were verified against
the source** (not just the report); the runtime values are as-reported and unverified.

**Class A — value divergences** (both engines run the query, return different values):
1. **`CAST(<float> AS VARCHAR)` renders wrong** — reported `infer` → `'1'`,
   `transform` → `'1.0'`. Cause **confirmed**: `display_value` (`src/expr.rs:114`)
   uses `f64::to_string`. Also affects `CONCAT`. Related: `1e300` expands to a
   300-digit string vs DataFusion's `'1e300'`.
2. **`ROUND(<int>)` returns the wrong type** — reported `infer` → int `3`,
   `transform` → float `3.0`. Cause **confirmed**: `eval_builtin`'s `"round"` arm
   (`src/expr.rs:550`) passes `Value::Int(i)` through unchanged. A *type*
   divergence, not just formatting — and `src/types.rs:268` (`"abs" | "round"`)
   clones the arg's base type, so it propagates into the synthesized output model.
3. **`NULLIF(1, 1.0)` doesn't null** — reported `infer` → `1`, `transform` → `NULL`.
   Cause **confirmed**: `"nullif"` (`src/expr.rs:575`) compares with `Value`'s
   variant-tagged `PartialEq`, so `Int(1) != Float(1.0)`; DataFusion coerces
   numerically.

**Class B — surface gaps** (`infer` rejects what `transform` evaluates):
4. **Unary minus unsupported** — `SELECT -a` / `SELECT -1` → `Unsupported
   expression` from `infer`. Cause **confirmed**: `convert_expr`
   (`src/expr_build.rs:39-42`) handles only `UnaryOperator::Not`.
5. **`||` string concat unsupported** — `Unsupported operator: ||` from `infer`.
   Cause **confirmed**: `convert_binary_operator` (`src/expr_build.rs:205`) has no
   `StringConcat` arm.

**⚠ Pinning status (updated 2026-07-17).** `tests/test_diff_rust_bugs.py` **now
exists — but only on the codegen worktree/branch** (`worktree-codegen-inferfn`, not
on master), where the codegen dev pinned `ROUND(int)` (#2) and the newly-found
`COALESCE(int,float)` typing bug as strict `xfail`-on-rust. **On master no pinning
test exists yet** for any of these. **Adding the pinning tests on master is part of
this work**, and it should not depend on the codegen branch landing.

**Start:** each bullet names its cause + site; fix independently. **DataFusion is the
oracle** (`transform` *is* DataFusion, and the harness gates on it — canonical:
**decision-1**) — match DataFusion, don't codify current native behavior. Pin each with
a strict `xfail` on the rust backend first so a fix flips it to a failure and forces
the marker's removal. Task: **TASK-1**.

**Update (2026-07-17) — codegen adversarial fresh-eyes review found 5 more native
divergences on uncovered edge cases** (all verified against both engines; the
differential corpus never hit them). These are *additional* native `InferFn` bugs
(oracle = DataFusion), independent of the codegen engine's own status:
6. **`COALESCE(int, float)` mis-types** — reported `infer` → int `3` / `ValidationError`,
   `transform` → float `3.0` (numeric supertype). **Same root as #3**: the
   `NULLIF`/`COALESCE` output type is taken from `args[0].base` instead of the common
   supertype. Codegen fixed its side (`_function_type` → common supertype) + pinned
   xfail-on-rust; native still to fix.
7. **`SUBSTR` with start ≤ 0** — `SUBSTR('hello',0,3)` → DF `'he'`, engines `'hel'`;
   `SUBSTR('hello',-2,5)` → DF `'he'`, engines `'hello'`. DF uses Postgres windowing
   (positions < 1 consume the length). Realistic; silent wrong string.
8. **`NaN = NaN`** → DF `True`; both engines raise `Cannot compare NaN`. Reachable via
   float ÷ 0.
9. **`CAST(str AS BOOL)`** — DF accepts `'t'`/`'1'`/`'yes'` → `True`; engines accept
   only `'true'`.
10. **`CAST(str AS INT/FLOAT)` with surrounding whitespace** — `CAST(' 42 ' AS BIGINT)`
    → DF **errors**; both engines strip whitespace and return `42`.

Realism (dev's read): #7 (substr) is the one worth fixing; #6 is a clean type bug;
#8/#9/#10 are low-realism edge cases. All uncovered — the engine is green as-is.

### sklearn transformer integration — functionality & parity
*Strategy (native-goal + opaque-fallback, fallback-first) canonical record:*
**decision-4**. Tasks: **TASK-5** (native-swap), **TASK-6** (assembly surface).
> **⚠ REFRAME IN PROGRESS (2026-07-16).** AmirHossein narrowed the near-term target
> to the **(b) fallback-execution-node** direction — running an already-fitted
> sklearn estimator we have no native version of as an **opaque black-box step
> inside our engine** (materialize input columns out of the Arrow/row representation
> → call the fitted object's `.transform()` → read the result back in → continue).
> Correct (it *is* sklearn) and inefficient (a Python call + materialized array on
> the request path, breaks fusion) — but it makes partial coverage shippable and
> lets native impls replace fallbacks one at a time. The **(a) compose-in direction**
> (our own transformers as sklearn estimators dropped into someone else's pipeline)
> is **deferred** — see the "Estimator-interface compliance" item below. The
> **fallback-phase slice decomposition below assumed compose-in-first and is being
> re-sequenced** around the fallback node; Dev is brainstorming the concrete
> fallback-node shape with AmirHossein. Treat the fallback-phase slices as stale until that
> lands. The native-swap phase (native per transformer) intent and the transformer
> prioritization are unaffected.

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
primary serving goal (see [Vision](<../backlog/docs/doc-3 - Vision.md>), "Positioning" + "Serving without
the intermediate").

**Sequencing (decided 2026-07-16): Python-fallback-first, then optimize per
transformer.**
- **Fallback phase — compose-in surface, fallback-backed.** Stand up the whole structural
  surface end to end with every transformer backed by the **real sklearn object**
  (the Python fallback): the estimator interface, the `Pipeline`/`ColumnTransformer`
  glue, and the assembly-parity harness. Ships hooks 1 (composes in) + 3 (typed
  contract) and proves the *structure* is right — does ours actually slot into a
  stock pipeline, does column routing + concatenation yield a bit-identical vector —
  before any engine reimplementation. The fallback is trivially parity-correct (it
  *is* sklearn), so it doubles as the oracle for the native-swap phase. Explicitly **not** a speed
  win (fallback is slower than raw sklearn — extra indirection + still builds the
  DataFrame); it's the correctness/structure milestone.
- **Native-swap phase — native per transformer.** Replace each transformer's internals with a
  **native** engine implementation (SQLTransform/expression, no sklearn call), one at
  a time, each diffed against the fallback until bit-identical, then flipped over.
  Same public API throughout, so speed arrives transparently, transformer by
  transformer. Ordering within B: `StandardScaler` as the thin vertical (scalar
  state, no fan-out) to shake out the native path, `OneHotEncoder` next for the hard
  cases (fan-out + unknown-category).
- The **n = 1 serving-path optimization** (delete the dict/DataFrame, Rust-parse
  input, contiguous feature buffer) is the *separate* serving-path item on top of the
  native-swap phase's native transforms — not part of this item.

The fallback phase is sliced into ordered slices — **thin-vertical → `ColumnTransformer`-glue
(incl. a multi-output transformer) → breadth** — the authoritative
breakdown + tick state live in the transformer-foundation section of [ROADMAP.md](ROADMAP.md).

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

The prioritized transformer list (tiers + native-machinery status + parity gotchas)
lives in [the sklearn transformer plan](<../backlog/docs/doc-2 - sklearn-transformer-implementation-plan.md>).

### Feature-output model — records / dense / sparse
The output side of the serving use case (from Fermi/Investigator, 2026-07-18; **folded
into the transformer-foundation phase**, tasks TASK-13…17). Today the infer path returns pydantic records;
sklearn interop needs numeric matrices, and text/categorical needs **sparse** output.
Three output contracts: (1) pydantic records (current), (2) dense float64 `(n,k)`,
(3) sparse CSR.

Connective design decisions (the tasks carry the rest):
- **Sparse feature = a per-row COO struct column** `struct<indices: list<int32>,
  values: list<float64>>`. It's 1:1 with rows, so sparseness lives *inside the cell*,
  not in row cardinality — that's what lets **one SELECT = a mixed dense+sparse feature
  set**. Materializes to scipy CSR for free (concat arrays → data/indices, per-row
  lengths → indptr).
- **Dimension N + unseen-key handling come from the FITTED transform**, not the cell or
  a type-level policy. N pins `shape=(n,N)` so batch width can't drift — that drift is a
  **silent model-misalignment bug** (a batch missing the last vocab term materializes
  narrower). Hence the **width-invariant assert** (the sparse-column task, TASK-14).
- **One SELECT → dense⊕sparse via type-directed decompose+assemble** (the assembler task,
  TASK-16). sklearn
  `ColumnTransformer` is the **internal** assembly target (hstack + densify), *not* a
  user API — users write SQL, we compile the ColumnTransformer.
- Fitted domains (vocab/idf, categories) ride the existing static_tables/lookup
  mechanism — no new artifact store.
- **The tfidf / array multi-hot task (TASK-17) is the opaque one** — needs explode, so it
  maps onto the shipped opaque-transform mechanism (decision-3). The sparse-column /
  scalar-one-hot tasks are fixed-fanout and composable. This is the same
  fixed-fanout-composes / variable-expansion-is-opaque boundary the multi-language
  runtimes (doc-4) are built on.

**Usability signal (2026-07-18, House Prices):** the column numeric/categorical roles
still live in Python for the sklearn handoff, so with the literal-SQL form the column
list is **duplicated (SQL + Python)** — a real papercut on wide (80-col) datasets.
Motivates the assembler task (TASK-16) owning column routing (compile the `ColumnTransformer` from
the SQL, so roles aren't re-declared in Python). Not a separate task — an assembler design
constraint.

### Opaque transform support — Part 1 (engine capability) → Part 2 (SQL surface)
*Split rationale (why Part 1/Part 2) canonical record:* **decision-3**. Follow-up
tasks: **TASK-2** (Part 1), **TASK-3** (Part 2). Both parts have shipped; this entry
retains the status + deferred-direction detail.
The near-term fallback-node work, **split into two tasks (AmirHossein, 2026-07-17)**,
Part 1 first. **Supersedes/splits** the bundled spec
[opaque-transform-refs-design](superpowers/specs/2026-07-17-opaque-transform-refs-design.md)
(`f213d6c`) and its predecessor mixed-pipeline draft (`963eea6`).

**Part 1 — the native engine can use transformers — ✅ SHIPPED (`4d5c85c`, 2026-07-17).**
Both engines invoke an **opaque, already-fitted Python transformer** — an sklearn
transformer *or* a whole fitted `Pipeline` — during one query: marshal the row's
values out (aligned by `feature_names_in_`), call `.transform()`, marshal back through
a declared `out_schema`. Differential parity green (`transform`=DataFusion oracle ==
`infer`=native); suite 197 passed, `cargo test` 2 passed. Pure engine capability,
independent of how it's expressed in SQL — native where we have it, opaque where we
don't. Merge not pushed (local master).

**Part 1 follow-ups — surfaced by review, belong to Part 2 (2026-07-17):**
- **(a) Out-of-projection transformer calls — latent parity gap.** When a transformer
  call isn't in the query's projection, native fails loud (`Unknown function`) while the
  DataFusion oracle (globally-registered UDF) still executes it. Harmless today (Part 1
  only calls in-projection) but diverges once Part 2 generates reserved names across
  clauses. **Fix:** a build-time guard that rejects/resolves transformer refs across
  *all* clauses, or register/resolve consistently so both engines agree.
- **(b) Single-field 1-in/1-out transformers — missing parity coverage.** The oracle
  marshals output via `y[:, i]`, assuming a 2-D transform result; a 1-D (single-output)
  return would break that indexing. No 1-output transformer ships in Part 1 — **add a
  differential parity case before one does.**
- **(c) `out_schema` vs natural-dtype invariant — enforce or reconcile.** The declared
  `out_schema` must equal the transform's natural output dtype. The engines reach the
  declared type by *different coercion layers* (DataFusion UDF: pyarrow cast; native
  infer: pydantic model-validate), which agree only when no real coercion happens.
  Documented at both call sites (`ed09a47`) but convention, not enforced. **Fix:**
  validate `out_schema` against the fitted transformer's actual output dtype at
  registration, *or* reconcile the two coercion layers so a mismatch can't diverge.

**Part 2 — the SQL / authoring surface — ✅ SHIPPED (`6a270d4`, 2026-07-18).** A fitted
sklearn transformer can be referenced as an opaque `{ref}` in an authored t-string —
`SQLTransform(t"SELECT {svd}({w2v}(inp)) AS out FROM __THIS__")` — with single refs and
nested threading `f(g(x))`; both engines (`transform`=DataFusion, `infer`=native)
differentially equal. Python-only — reused Part 1's Rust `Expr::Transform` + DataFusion
UDF unchanged. Suite 201. Spec `2026-07-18-transformer-refs-design.md`, plan
`2026-07-18-transformer-refs.md`; the superseded `2026-07-17-opaque-transform-refs-design.md`
was removed.

**Deferred (recorded in the spec): aggregate over a transformer's output** (and the
full fit-staging machinery). It needs **inline struct-field access in the DataFusion
serve query**, which DataFusion doesn't do — **the derived-table wall the split
predicted is real.** In-scope cases sidestep it by passing whole structs through and
never doing field access. This is the deferred direction; revisit if
aggregate-over-output is actually needed (it's what fit-cascade-across-a-transformer
would require).

**Part 2 follow-ups — from the whole-branch review (opus; verdict ready-to-merge, no
Critical/Important):**
1. **Redundant fit-time work.** The single-transformer path runs `.transform()` on the
   training data **twice** — once in `_derive_schemas` to sniff the `out_schema` dtype,
   once in `_materialize` whose output is then discarded (no outer consumer). Cold-path
   and correct, just wasteful. **Fix:** reuse the `_derive_schemas` probe output and
   skip `_materialize` when a ref has no outer consumer.
2. **Opaque error messages on out-of-scope paths** (both error *safely* — never wrong
   output — but the messages are cryptic): (a) aggregate over transformer output → raw
   DataFusion `Invalid function '__compose_0__'`; (b) unfitted transformer → `must be a
   SQLTransform or its .transform` `TypeError`. **Fix:** explicit pre-checks with
   friendly messages.
3. **Negative / contract test coverage** (missing, cheap `pytest.raises`): mixed
   leaf+nested args `{t}(a, {g}(b))`; aggregate-over-output; column vs
   `feature_names_in_` mismatch; unfitted ref. **Plus** a regression test for
   `transformer + PARTITION BY <transformer-input-col>` coexistence — parity holds
   because `rewrite_sql` qualifies the `named_struct` columns to `__THIS__`, but it's
   currently guarded only by a downstream side-effect, so lock it in.
4. **Confirmatory only (low value):** a 3+ level nesting test.

**Why the split was right (confirmed):** the bundled spec's *surface* half dragged real
engine complexity in for cosmetic reasons — the lowering wanted a **derived table** (to
bind the struct once and project its fields with clean names, since DataFusion won't let
you alias `unnest` output or do inline field access), and supporting a derived table in
the row engine means **adding a projection node inside the `RelNode` plan tree** (today
`Plan { projection, input }` projects only at the top level). Part 2 shipped by
**avoiding** that wall (whole-struct passthrough, no field access) and deferring the
cases that hit it — exactly the containment the split bought.

### Estimator-interface compliance of our transformers (compose-in / hook 1) — deferred
Make *our* transformer objects pass sklearn's `check_estimator` conformance
(`fit`/`transform`/`get_feature_names_out`/`get_params`/`set_params`/clone/
`n_features_in_`/tags, etc.) so they drop into a **stock sklearn
`Pipeline`/`ColumnTransformer`** and coexist with sklearn's own transformers — the
**(a) compose-in** direction, VISION hook 1. **Deferred (2026-07-16):** it only
matters once we surface our own estimator objects into *someone else's* sklearn
pipeline; the near-term target is the other direction — the fallback execution node
that runs fitted sklearn estimators inside *our* engine (see the reframe banner on
the sklearn-integration item above). Fix later. **Start (when picked up):** run
`sklearn.utils.estimator_checks.check_estimator` against our transformer base and
close the gaps; this is also where the `get_feature_names_out` provenance contract
(hook 3) gets pinned for external consumers.

### Per-transformer differential parity harness
The oracle every sklearn transformer is validated against — the transformer analogue
of the differential test harness (expression/join). For each transformer, a
**parametrized parity matrix** asserts our `transform` output is bit-identical to the
real sklearn object's, across (a) the parity-sensitive params (see the per-transformer
notes in [the sklearn transformer plan](<../backlog/docs/doc-2 - sklearn-transformer-implementation-plan.md>) — e.g. `StandardScaler`
population-vs-sample std, `OneHotEncoder` `handle_unknown`/`drop`/infrequent, exact-vs-
approx quantiles for `RobustScaler`) and (b) input edge cases: nulls, unseen
categories, single row vs batch, int/float/string dtypes, empty/degenerate columns.
**Why its own item:** it's the mechanism behind the native-swap phase's per-transformer native
swaps — each native impl is diffed against the same matrix its Python fallback
already passes, so a native/​sklearn divergence fails loud. **Distinct from** the
end-to-end **assembly**-parity harness (the ColumnTransformer-glue slice, the whole `ColumnTransformer`
vector): this one is *leaf* correctness per transformer, that one is *assembly*
correctness; both are needed and this one feeds that one. **Start:** stand it up in
the thin-vertical slice alongside the first transformer (`StandardScaler`) — one transformer through the
matrix — then grow the matrix per transformer as coverage lands. Reuses the existing
differential-harness patterns (`tests/test_diff_*`).

### Transformer execution model — procedures (UDF/UDAF), macros, composition
The conceptual backbone the two items above build on (from a 2026-07-15 design
discussion). Capture, not yet a spec.

**Deprioritized off the transformer-foundation critical path (2026-07-16).** The general UDF/UDAF/macro
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
  (native `InferFn`) for free) or a **genuinely new scalar op** needing a *dual* impl
  (InferFn native built-in **and** a DataFusion UDF for batch), kept in lockstep by the
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

Foundation for the composition output model, fan-out transformers, and the
feature contract. **Supersedes the narrower "Rust struct-support" ticket** — its
five bullets (struct `Value`, field access, output-model synthesis, DataFusion
parity) are subsumed here. Rather than bolt structs onto the closed scalar type
layer, replace that layer with a **recursive, extensible, schema-driven** one so
`InferFn` can carry the full pyarrow type surface. Spec:
[rich type system design](superpowers/specs/2026-07-16-rich-type-system-design.md).

**Why the pivot (2026-07-16):** composition needs structs; DataFusion has no
`struct.*` — it uses **`UNNEST`** (`unnest(struct)`→columns, `unnest(list)`→rows),
so we match that; and the engine should carry real feature-data types (also
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
and `infer` (native); a bare `{scaler}` on a fitted object raises the explicit
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
interpreter — the tag has to be propagated *through* the native engine and surfaced
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
feature-engineering SQL before investing — neither is supported in the native
interpreter today. **Start:** prioritize by what authoring (goal 1) actually
needs; `CASE WHEN` also needs Layer-1 interpreter support, not just a rewrite
change. **Demand signal (2026-07-18):** a usability test (House Prices) hit this as
the blocker to ordinal-encoding quality ladders (Ex/Gd/TA → 5/4/3) — currently
un-expressible, forced model-side. Real authoring need, not hypothetical.
**Ticketed: TASK-27** (CASE WHEN only; outer joins stay parked here as the separate
concern).

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
[Vision](<../backlog/docs/doc-3 - Vision.md>), "Serving without the intermediate"). **Why deferred:**
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
An older README roadmap item, superseded by the native `InferFn` interpreter. Keep
parked unless interpreter overhead becomes a *measured* bottleneck; revisit only
with a benchmark in hand.

**Update (2026-07-17) — a codegen engine now exists (fact, not yet a ratified
direction).** The codegen dev built it on `worktree-codegen-inferfn` (`sql_transform/
_codegen/`, plan `2026-07-17-codegen-inferfn.md`, 11 tasks): suite 397 passed / 14
skipped (containers) / 3 xfailed (pinned native divergences), parity target =
DataFusion oracle. The revisit-condition above ("benchmark in hand") is satisfied
(the earlier n=1 boundary-bound benchmark), and an engine has been written — but
**whether codegen is adopted as a maintained/default path vs. the native interpreter
is AmirHossein's pending framing call; this note records the artifact, not a
decision.** Its adversarial review surfaced **2 codegen-only parity divergences**
(native already matches the oracle here — no native ticket). **✅ Both fixed & merged into
codegen (`131fa0b`); TASK-4 Done.** Retained for the record:
- **float→string for |x| < 1e-4** — `CAST(1e-5 AS VARCHAR)` → DF `'0.00001'`, native
  `'0.00001'`, codegen `'1e-05'`; `1e-6` → DF `'1e-6'`, codegen `'1e-06'`. Realistic
  (small feature values); DF's exact float formatting is a known rabbit hole.
- **integer arithmetic overflow** — `9223372036854775807 * 2` → DF/native wrap to `-2`,
  codegen (Python bigint) `18446744073709551614`. Rare in feature transforms; fix
  touches all arithmetic ops.
Also needs codegen-side fixes for the shared bugs #7–#10 above if oracle parity on
the codegen path is wanted (it already fixed #2 ROUND and #6 COALESCE typing).

### Unify batch vs inference error *types* — non-goal (accepted divergence)
**Decision:** error-type parity across engines is a non-goal; only output *values*
must match. Canonical record: **decision-2** (`backlog/decisions/`). Locked in by
`test_div_by_zero_raises_on_both_engines_with_different_error_types`.
