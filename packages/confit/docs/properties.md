# Properties of the system

The laws this system holds — semantic guarantees, stated as invariants.
Companion to the [success measures](specs/success-measures.md): KPIs are what we
*measure*; properties are what must remain *true*. Each entry says where the
property is pinned (test).

Layers follow the authored transform: fit → serving → engine.

The [system specification](../../../docs/specs/README.md) organizes the current
contracts by topic. The [authoring contract](../../sql-transform/docs/contract.md)
defines authoring syntax, fit behavior and the public artifact.
These properties record checks, not a second authoring specification.

---

## Authoring

**P1 — Fit binds the first parameter.** An authored transform computes
`(F, T) -> R`. Fit binds `F` and writes params for later execution on `T`.
Independent fit queries freeze into params tables. Decorrelation rewrites
supported correlated fit subqueries into fit queries and queries over params.
`SQLTransform` supports general composition; `SQLProjection` adds the row-local rule.
Confit checks serving admission separately.
*Pinned:* the original-query gate in `_marginal_projection_test.py::gate`,
the freezing tests in `_walk_test.py`, and correlation tests in `_correlate_test.py`.
The [authoring contract](../../sql-transform/docs/contract.md) defines the admitted forms.

**P2 — Window marginalization is bounded convenience.** Admitted window values
depend on partition keys and, where needed, order values.
The fit-time carrier executes the original window expressions together.
It preserves their numerical reading rather than replacing windows with grouped aggregates.
Physical-position windows (`row_number`, `ntile`, `lag`/`lead`), bounded ROWS
frames and EXCLUDE refuse.
*Pinned:* `_marginal_test.py::test_refusals_fire_pre_rewrite_in_the_authors_vocabulary`,
`::test_the_carrier_keeps_the_original_reduction_order`, and the original-query gate.

**P3 — Params multiplicity.** Every admitted window's value is constant
within its key tuple, so DISTINCT over (keys, values) collapses to exactly
one params row per key tuple, and each LEFT JOIN matches at most one row —
serving never duplicates or drops a row (`shape="map"` is provable).

**P4 — NULL keys are ordinary keys.** PARTITION BY groups NULLs into one
partition, so params joins use `IS NOT DISTINCT FROM`, never `=`; a NULL
partition key is a real params row, on every path (DuckDB and Confit).
*Pinned:* `_marginal_projection_test.py::test_standard_scaler_with_null_keys_and_null_inputs`,
`confit/tests/test_params_joins.py`.
*Scope — this is a **window** rule.* A key derived from a correlation
predicate instead mirrors whichever operator the author wrote: `=` rejects
NULL keys at fit and never ships that group, `IS NOT DISTINCT FROM` keeps it.
Applying P4 there invents an answer for a key `=` cannot reach — measured.
Both rules require the params key to group rows exactly as the join predicate does.
A coercion or collation can break that requirement.
*Pinned:* `_correlate_test.py::test_the_join_predicate_mirrors_the_operator...`.

**P5 — Retired: automatic chain flattening.** Window marginalization no longer
rewrites CTE or derived-table projection chains. Author explicit fit/request
SQL or compose general `SQLTransform` members.

**P6 — Retired: alias-based uncorrelatedness.** Authored SQL can express
supported correlated fit subqueries. Bounded window marginalization keeps
uncorrelated scalar/EXISTS subqueries over the fit data.
The [authoring contract](../../sql-transform/docs/contract.md) defines both boundaries.

**P7 — Refusals occur at the applicable boundary.** The SQLProjection constructor
checks syntax and the projection's row-local form. Fit checks learned fields and widths,
group schemas, nested-column shadowing and params multiplicity.
DuckDB binding can fail when a fit or batch query executes.
`FittedProjection.compile()` checks Confit admission separately and returns a fresh function.
Data-dependent serving traps remain governed by the oracle contract.
No boundary permits silent changes to the authored computation.

Read request metadata from `fitted.schema` and serving state from
`fitted.sql`, `fitted.params` and `fitted.udfs`.
Read backend metadata from the compiled Confit function, not the estimator.
The current authoring contract governs these fields and refusal timing.

**P8 — The `__` prefix is reserved.** Authored relations, common table expressions,
and output aliases cannot use this prefix, except for `__FIT__` and `__THIS__`.
Internal fit names use it. Window derivation also uses names chosen to avoid
authored-name collisions.
*Pinned:* `_reserved_test.py`.

**P8b — Private ordinals do not enter the public artifact.** Raw estimator
fit refuses fit columns named `__cf_fit_row` or `__cf_row`, case-insensitively.
Batch projection input refuses `__cf_row`.
Derived window names avoid authored-name collisions; there is no blanket
`__cf_` prefix refusal.
See the [authoring contract](../../sql-transform/docs/contract.md).

**P9 — The oracle is the parser and the printer.** SQL is parsed and
printed by DuckDB itself (`json_serialize_sql`/`json_deserialize_sql`);
interpreted node shapes are validated by pydantic views (drift fails as
one named error), carried nodes pass through opaquely, and synthetic nodes
are cloned from oracle-serialized templates.
*Corollary — carrying is not optional.* Measured on 1.5.5,
`json_deserialize_sql` requires exactly one field: dropping `BASE_TABLE.type`
is rejected, and dropping `BASE_TABLE.table_name` is **accepted and prints
different SQL**. Dropping a serialized field can therefore change the query instead of raising an error.
Each typed node retains every serialized field.
Unknown node types retain their complete contents and typed children.
The walk can then find `__FIT__` reads under unknown node types.
The AST data types in `_nodes.py` implement this rule.
The serialized shapes in `_shapes.json` record the DuckDB version's fields.
*Corollary — an identifier means what the oracle binds.* DuckDB folds every
identifier, quoted ones too (unlike Postgres), so wherever the walk compares
one name to another it folds: CTE keys, the supplied connection's catalog,
and a correlated reference's qualifier. It also binds *less* than it lists —
internal views and ATTACHed databases are not on the search path, so they are
not names the catalog may claim *unqualified*; qualified, they are the
connection's own by construction, since everything captured from the frame is
registered under a bare name. The boundary is the caller's frame: Python's
namespace is case-sensitive and is looked up, not bound, so `codes` and
`Codes` stay two variables.

## Fit

**P10 — Fit follows dependencies.** Fit executes dependent SQL steps after
their inputs exist. A narrow DISTINCT pick over a frozen carrier runs before
fit releases that carrier. Only serving-live learned params remain in the artifact.
An authored query can explicitly retain fit data.
The public artifact does not expose an intermediate plan or its released tables.
See the [authoring contract](../../sql-transform/docs/contract.md);
the planning and execution code is in `_plan.py` and `_program.py`.

**P11 — Projection execution controls thread count.** Projection fit, probes,
and batch execution use `threads=1`, then restore the observed setting.
The original-query gate uses the same thread count on both DuckDB paths.
This does not promise reproducible estimator fits or bit-exact results across machines.

**P12 — Raw estimators fit clone-per-group.** Each fitted partition owns a
clone of the captured prototype, with a deepcopy fallback.
Its params ID selects that instance.
Rebinding a captured caller name does not replace the captured object.
Projection fit materializes captured relations into the stored Arrow params mapping.

## Serving

**P13 — One public artifact, separate admission.** A Confit-servable projection
has four public serving fields: `.sql`, `.schema`, `.params` and `.udfs`.
`.sql` is unordered serving SQL; `.schema` is the filtered, nullable request schema.
`.params` stores normalized captured statics and serving-live learned tables.
DuckDB batch execution and each fresh Confit build use the same params and UDFs.
Replacing an entry in the stored params mapping affects subsequent execution
and builds, not an already built function.
Batch success does not prove Confit admission.
See the [authoring contract](../../sql-transform/docs/contract.md) and control C3.

**P14 — The one NULL story.** Unseen group ⇒ LEFT JOIN miss ⇒ NULL
instance id ⇒ NULL output. NULL-ness always flows through join data, never
through a lookup convention. Distinct from it: an id *present but missing
from instances* is a broken artifact and raises — never NULL.
*Pinned:* `_serving_test.py::test_unseen_group_is_null_row_at_a_time`,
`_raw_test.py`, `confit/tests/test_udfs.py`.
*Scope — also a **window** rule.* An unseen key in a lifted correlation takes
whatever **DuckDB's own correlated subquery** returns for a key that is not
there — `0` for `count`, NULL for `avg`, whatever the author's `CASE` says.
There is no list of which aggregates those are, and there cannot be one:
`count_if(x)` is NULL on empty input and `count(x) FILTER (WHERE x)` is 0, the
same count spelled twice.

Nor is it the aggregate's value on an empty *input*, which is a different
number for three of DuckDB's 68 aggregates — `entropy` is 0.0 on empty and NULL
on a miss, because the count-bug repair is applied to `count` alone. P9 settles
which is right: DuckDB defines the reference for this authoring computation.
The probe uses a guaranteed join miss, not an empty scan.
It measures the correlated subquery's own missing-key result.
A separate match count distinguishes misses from present groups with NULL results.
`COALESCE` cannot make that distinction.
*Pinned:* `_correlate_test.py::test_an_unseen_key_takes_the_subquerys_own_empty_value`,
`::test_a_present_group_whose_value_is_null_is_not_a_miss`,
`::test_the_miss_value_is_duckdbs_own_not_the_aggregates_empty_input`, and
`::test_a_false_guard_means_the_group_is_empty_not_that_the_answer_is_null`.

**P15 — UDFs are pure, deterministic, declared.** A UDF is a pure function
of its arguments (per-group weights arrive as data via the id argument,
P14); it declares `takes`/`returns` in the engine type vocabulary
("i1"|"i64"|"f64"|"str"); the scalar `__call__` (plain values, None=NULL,
tuple or None out) is THE semantic contract every binding must match;
determinism is the price of admission (the round-trip control assumes it).
The default `apply_batch` loops the scalar form: boundary amortized, math
not vectorized, so batch ≡ row bit-exactly.

**P16 — Width rules.** Output width and field *names* are knowable only
post-fit and are declared on the fitted UDF (`returns`/`return_names`).
**A call with declared field names is struct-valued at EVERY width; only
field reads are scalars.** An addressed field (`t(...).name`, validated
at fit — the P7 carve-out) serves as a field read over the ONE call,
evaluated once per row on both paths (DuckDB CSEs the identical pure
calls into one struct-returning invocation; confit binds each read to one
SSA lane of one shared-site ecall).
The engine's `list | None` boundary also serves direct
`DuckDBInferFn(udfs=...)` users with unnamed width-k externs.
An unseen group is NULL per field read.
The [authoring contract](../../sql-transform/docs/contract.md#4-projection-only-estimators-and-scalar-udfs)
defines raw BIGINT IDs, struct outputs, FILTER, ordering, and fit-source admission.

**P16a — Names are the type; matching is name-keyed.** A fitted transform
is `S → T` between named structs: S's field names and types come from the
bundle (the caller's binding), T's are learned at fit — an author
declaration (`Named`) is authoritative, sklearn's `get_feature_names_out`
advisory, canonical `f0..` the fallback. Addressing an output by *position*
is never possible, because a refit can renumber lanes (measured:
OneHotEncoder gaining a category shifts every lane after it) — so a field
that disappears refuses by name rather than silently rewiring. A codomain
that differs per group is not a function type and refuses.

**P17 — Serving metadata declares actual request widths.** The fitted
projection's nullable schema contains the request fields its serving SQL reads.
Confit checks those declared widths independently.

## Engine (Confit)

**P18 — Two outcomes, parameterized.** Serve bit-for-bit identical to
DuckDB — with the declared udfs registered, when any — or refuse at build
with a named error. There is no third mode. (Control C2 measures this.)

*Reference and comparison rules:* the [oracle contract](oracle/README.md).
[C2: engine parity](specs/success-measures.md#engine-parity-c2) specifies the control,
including its narrow approved bounds; known-limitations.md §5 records the
optimizer-off choice's user-visible cost.

**P19 — The backends cannot drift where they share code.** Everything with
nontrivial semantics executes through helpers shared by the interpreter
and cranelift (extern calls included: one `call_extern` enforces the
declared return shape for both); the 500-seed random-IR differential
guards the rest.

**P20 — Statics are frozen at build.** Params/static tables materialize
into probe maps at construction; nothing re-reads them at serve time. A
declared-non-nullable NULL, a duplicated unique key, or a shape-violating
UDF result is a named build error or trap — never a wrong value.

**P21 — An exact answer is a function of the query and its declared
inputs.** A reading that also depends on hidden table history, an
unselected evaluation path, or an uncontrolled run is not an exact target:
it is refused, or served under a narrower ruled contract, never frozen as
whichever answer one run happened to give. The ruled exception is
confit's own serving-row order.

*Argued:* [claim: nondeterminism-axiom](oracle/03-nondeterminism.md#decision-rule).
*Pinned:* `tests/known_divergences/test_trap_elision.py::test_duckdbs_is_null_elision_is_not_a_function_of_the_query_or_the_rows`
(the optimizer-on reading depends on insert history, so the oracle is
optimizer-off) and `tests/test_arrow_schema_api.py::test_a_static_tables_only_query_refuses_at_build`
(nothing is frozen at construction: a query with no request row refuses).

