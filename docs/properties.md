# Properties of the system

The laws this system holds — semantic guarantees, stated as invariants.
Companion to `docs/kpis.md`: KPIs are what we *measure*; properties are
what must remain *true*. Each entry says where the property is argued
(spec/draft) and where it is pinned (test). A change that repeals one of
these is a design decision and goes through a draft, not a diff.

Layers follow the pipeline: marginalization → fit → serving → engine.

---

## Marginalization (the rewrite)

**P1 — The binding-time split.** A query is a chain of strict projections
over `__THIS__`. Everything whole-table (window aggregates, scalar/EXISTS
subqueries, transformer windows) is *static* — computed at fit, frozen into
params tables; everything row-wise is *dynamic* — survives verbatim into
`serving_sql`. An expression that survives untouched appears identically in
both texts, so the two cannot disagree on it.
*Spec:* 2026-07-29-sql-projection-marginalization-design. *Pinned:*
`_marginalize_test.py` rewrite goldens.

**P2 — The one window rule.** A window is admitted iff its value is a
function of row-visible values (partition keys, plus order values under
RANGE/GROUPS peers). Physical position is the one thing a join key cannot
carry: `row_number`, `ntile`, `lag`/`lead`, bounded ROWS frames, EXCLUDE
are refused *forever*, not "later".
*Spec:* window-widening design. *Pinned:* refusal tables in
`_marginalize_test.py`.

**P3 — Params multiplicity.** Every admitted window's value is constant
within its key tuple, so DISTINCT over (keys, values) collapses to exactly
one params row per key tuple, and each LEFT JOIN matches at most one row —
serving never duplicates or drops a row (`shape="map"` is provable).
*Spec:* marginalization design ("Multiplicity holds by construction").

**P4 — NULL keys are ordinary keys.** PARTITION BY groups NULLs into one
partition, so params joins use `IS NOT DISTINCT FROM`, never `=`; a NULL
partition key is a real params row, on every path (DuckDB and Confit).
*Pinned:* `_projection_test.py::test_standard_scaler_with_null_keys...`,
`confit/tests/test_params_joins.py`.

**P5 — Chain flattening is substitution.** CTEs/derived tables resolve to
a base-first level list and flatten by expression substitution; output
names are frozen as explicit aliases before rewriting so substitution can
never change a column's name.
*Spec:* projection-chains-fit-plan design. *Pinned:* chain goldens.

**P6 — Subqueries are provably uncorrelated.** `FROM __THIS__` carries no
alias and no other relation is in scope, so there is *no syntax* to
reference the outer row — scalar subqueries run verbatim as fit steps.
IN/ANY/ALL (per-row membership) refuse.

**P7 — Refusals are construction-time and named.** Everything refused is
refused at `SQLProjection(...)`/`DuckDBInferFn(...)` construction with an
error naming the construct — never at fit, never at serve, never silently.
*Carve-out (DRAFT-24):* a transform's codomain T is **learned**, so
refusals that depend on it — an addressed field that does not exist, a
declared width that disagrees with the fitted one, a per-group codomain
disagreement, a width-k call used inside an expression — are raised at
**fit**, by name. Everything syntactic stays construction-time; nothing
moves to serve time.
(The corpus's FAILED bucket pins the "never silently" half — control C5.)

**P8 — the `__` prefix is reserved.** In `sql_transform.model` every
synthesized name lives under `__` (`__param_0`, `__param_fit`,
`{name}__x{token}`), so an identifier there is refused at construction —
relation, CTE or alias. `__FIT__` and `__THIS__` are the exception: they are
the two parameters, and the only `__` names an author writes. Implemented
2026-08-08 (TASK-75); before that the prefix was reserved in name only and a
user relation called `__param_0` silently beat the frozen parameter.

**P8 (old implementation) — `__cf_` is reserved.** The prefix (case-insensitive) is refused in
input SQL and declared schemas; all synthesized names live under it, so
generated and authored names cannot collide.

**P9 — The oracle is the parser and the printer.** SQL is parsed and
printed by DuckDB itself (`json_serialize_sql`/`json_deserialize_sql`);
interpreted node shapes are validated by pydantic views (drift fails as
one named error), carried nodes pass through opaquely, and synthetic nodes
are cloned from oracle-serialized templates.

## Fit

**P10 — The fit plan is an inspectable DAG.** `Marginalized.plan` is a
topologically ordered list of named steps; SQL steps are plain SQL runnable
by hand, `kind="fit"` steps produce a params table (join keys +
`__cf_est`) plus fitted instances. Every intermediate is registered by
name and inspectable after fit.

**P11 — Fit is deterministic.** Fit runs at `SET threads = 1` (DuckDB's
parallel window aggregation is not bit-deterministic for floats — measured
1/500 fuzz drift). Same training table ⇒ bit-identical params, on any
machine.

**P12 — Transformers fit clone-per-group.** The registry object is never
mutated: each partition fits a `clone()` (deepcopy fallback), keyed by an
instance id that is *data* in the params table. Captured scope objects are
snapshotted at construction — later mutation of the variable cannot change
a constructed projection.

## Serving

**P13 — One artifact, two bindings.** A fitted projection is exactly three
pieces — `serving_sql` + Arrow params tables + UDF objects. `transform`
binds them to DuckDB (batch); `infer`/`infer_batch` bind the *same three
pieces* to Confit. No binding has private state (control C3 is this
property, measured).

**P14 — The one NULL story.** Unseen group ⇒ LEFT JOIN miss ⇒ NULL
instance id ⇒ NULL output. NULL-ness always flows through join data, never
through a lookup convention. Distinct from it: an id *present but missing
from instances* is a broken artifact and raises — never NULL.
*Pinned:* `_transformers_test.py`, `_serving_test.py`,
`confit/tests/test_udfs.py`.

**P15 — UDFs are pure, deterministic, declared.** A UDF is a pure function
of its arguments (per-group weights arrive as data via the id argument,
P14); it declares `takes`/`returns` in the engine type vocabulary
("i1"|"i64"|"f64"|"str"); the scalar `__call__` (plain values, None=NULL,
tuple or None out) is THE semantic contract every binding must match;
determinism is the price of admission (the round-trip control assumes it).
The default `apply_batch` loops the scalar form: boundary amortized, math
not vectorized, so batch ≡ row bit-exactly.
*Spec:* udf-protocol-serving-calls design; DRAFT-22.

**P16 — Width rules.** Output width and field *names* are knowable only
post-fit and are declared on the fitted UDF (`returns`/`return_names`).
**A call with declared field names is struct-valued at EVERY width; only
field reads are scalars.** An addressed field (`t(...).name`, validated
at fit — the P7 carve-out) serves as a field read over the ONE call,
evaluated once per row on both paths (DuckDB CSEs the identical pure
calls into one struct-returning invocation; confit binds each read to one
SSA lane of one shared-site ecall). A bare transformer call — any width,
as an item or inside an expression — **refuses at construction** by name:
it has no scalar reading, and no struct output boundary exists until
DRAFT-25's nested outputs. Bundles (`struct_pack(...)`) are destructured
at construction into positional feature arguments: no STRUCT or LIST
value ever crosses the serving output boundary — every output column is
scalar, and the struct exists transiently inside DuckDB's expression
evaluation only. The engine's `list | None` boundary remains for direct
`DuckDBInferFn(udfs=...)` users with UNNAMED width-k externs.
*Amended 2026-08-04 (struct-valued calls, the subtraction loop):* the two
owned rules with no oracle reading are deleted — loop 1/4's width-1
scalar-valuedness (and its field-read collapse) and loop 3's flat
alias-prefixed bare-item expansion (`AS e` → `e_pca0`). Authored
spellings that used them now refuse or re-spell as field reads; an unseen
group is NULL per field read. DRAFT-25 restores struct-level outputs (and
the NULL-struct distinction) properly.
*Amended 2026-08-05 (fit/transform split):* the `tfm(x) OVER w` sugar is
**deleted** — the one construct with no oracle reading (a window
aggregate returns one value per partition; the sugar returned per-row
values). The surface: bare `tfm(bundle).field` is the single sugar
(global fit-transform); any other fit scope spells the split —
`tfm_transform(tfm_fit(bundle) OVER (PARTITION BY ...), bundle).field` —
where `tfm_fit` is a true window aggregate (same θ per partition, θ =
`Struct<type, id>`) and `tfm_transform` a true scalar (fit-here-apply-
there: the transform bundle may differ in values, name-keyed against the
fit bundle). A registered transformer `x` reserves `x_fit`/`x_transform`.
Fit-side contract: a fit is a multiset aggregate (order-blind,
seed-fixed, author-signed — P15's family); declared order-sensitive fits
will name their order in-call (a later slice); there is NO determinism
promise for fit reproducibility in v0. Not-yet-landed fit clauses refuse
by name: FILTER, in-call ORDER BY (ordered fits), frames, window-clause
ORDER BY (running fits), θ export.
*Spec:* 2026-08-05-fit-transform-split-design.

**P16a — Names are the type; matching is name-keyed.** A fitted transform
is `S → T` between named structs: S's field names and types come from the
bundle (the caller's binding), T's are learned at fit — an author
declaration (`Named`) is authoritative, sklearn's `get_feature_names_out`
advisory, canonical `f0..` the fallback. Addressing an output by *position*
is never possible, because a refit can renumber lanes (measured:
OneHotEncoder gaining a category shifts every lane after it) — so a field
that disappears refuses by name rather than silently rewiring. A codomain
that differs per group is not a function type and refuses.
*Spec:* DRAFT-24.

**P17 — Serving rows have the training table's shape.** The serving row
model derives from the training table's arrow schema (real types);
unmappable columns are opaque and allowed unless referenced. (v0 contract;
narrowing to referenced columns is a known future widening.)

## Engine (Confit)

**P18 — Two outcomes, parameterized.** Serve bit-for-bit identical to
DuckDB — with the declared udfs registered, when any — or refuse at build
with a named error. There is no third mode. (Control C2 measures this.)

**P19 — The backends cannot drift where they share code.** Everything with
nontrivial semantics executes through helpers shared by the interpreter
and cranelift (extern calls included: one `call_extern` enforces the
declared return shape for both); the 500-seed random-IR differential
guards the rest.

**P20 — Statics are frozen at build.** Params/static tables materialize
into probe maps at construction; nothing re-reads them at serve time. A
declared-non-nullable NULL, a duplicated unique key, or a shape-violating
UDF result is a named build error or trap — never a wrong value.

---

## Agreed direction, not yet law

Composition (FROM-position templates, frozen-artifact inlining,
`as_udf()`), native typed UDF entries (DRAFT-23), step semantics for
order-keyed windows (DRAFT-21), and the leakage/cross-fitting question
(DRAFT-20) are *directions* recorded in drafts — they become properties
here only when they land with pins.
