# Serving contract

The [goal](../goal.md) defines the target. This document supplies the API and
scope details; the [oracle](../oracle/README.md) defines SQL compatibility.
Current refusals are not automatically permanent product exclusions.
The [oracle policy decision](../decisions/oracle-policy.md) adopts the scope
classification and output-nullability rule below.

## API and output shape

The public engine is `confit.DuckDBInferFn`; `confit.BUILD_PROFILE` reports the
build profile. The exact interface is in `packages/confit/confit/_engine.pyi`.
`confit.oracle` and `confit.compare` are measurement utilities, not serving backends.

| Input or member | Meaning |
|---|---|
| `sql` | SQL to specialize |
| `row_tables` | Request-table names and Arrow schemas; declared widths matter |
| `static_tables` | Named Arrow tables frozen at construction |
| `udfs` | Declared callable objects, described below |
| `shape` | Output multiplicity proved at construction |
| `output_schema` | Output names, Arrow types, field order, and truthful nullability metadata |
| `infer_rows(rows)` | Dict-or-object rows in, dict rows out |
| `infer_arrow(batch)` | Arrow table in, Arrow table out |
| `backend`, `boundary` | Execution and Python-boundary diagnostics, not caller-selected modes |

Nullability must be sound: a field declared non-null must not produce NULL on a
valid successful execution. Conservative nullable metadata is permitted; its flag
need not equal DuckDB's inferred flag. This is an adopted requirement, not a claim
of complete enforcement. The [comparison contract](../oracle/05-the-comparison-contract.md#output-names-and-schemas)
records the current checks.

| Shape | Output per input row | Sequence |
|---|---|---|
| `map` | Exactly one | Input order |
| `filter` (default) | Zero or one | Input-order subsequence |
| `many` | Zero or more, explicitly requested | One output block per input row |

**exclusion: multiplicity-by-default.** Multiplying rows requires `shape="many"`;
`map` also rejects queries that may drop rows. Shape is proved at construction,
not inferred from a sample batch. Within-block join ordering follows the
[ordering contract](../oracle/03-nondeterminism.md), not a promise to reproduce
DuckDB's hash-join traversal.

The target execution backends are `cranelift` and `interpreter`; the Python
boundaries are `marshaller` and `generic`. The remaining legacy `constant` path
is an implementation gap documented in the [reference notes](../oracle/01-what-the-oracle-is.md),
not a third target backend.

Construction refusal is a `ValueError` identifying the unsupported construct.
The [refusal specification](../oracle/04-verdicts-agreement-abstention-refusal.md)
distinguishes intended prefixes, scope grounds, and current diagnostic gaps.
Construction success does not rule out data-dependent runtime traps.

## Scope and restrictions

**exclusion: whole-relation-shapes.** Splitting or combining a request batch must
not change a request row's answer. Grouping, aggregation, sorting, distinctness,
limits, and windows that depend on sibling request rows are outside the model.
Aggregating frozen rows matched by one request row is inside it, as is reducing
a list stored in that row. Queries reading no request table are outside it.
These boundaries follow the [scope decision](../decisions/trustworthy-fold.md).

Syntax is not the scope test. The frontend also refuses CTEs, subqueries, set
operations, `HAVING`, `QUALIFY`, named windows, `OFFSET`/`FETCH`/`TOP`, multiple
statements, table functions in `FROM`, `rowid`, `FULL OUTER JOIN`, and scalar calls
with aggregate/window modifiers. A CTE, for example, can describe a row-local
computation. These forms are excluded when they violate row locality; retaining
blanket bans on otherwise row-local uses still needs a product justification.

The inventory below records existing restrictions and their grounds. The adopted
classification does not blanket-approve every numeric budget, syntax ban, or
served divergence; each needs its own justification.

| Restriction | Rule or current behavior | Ground and status |
|---|---|---|
| **exclusion: per-row-general-work** | Refuse constructs requiring general compilation or binding per row; regex patterns, replacements, options, and group indexes must meet construction-time requirements | Specialization-inherent under the goal; complete enforcement not established |
| **exclusion: resource-ceilings** | 1 GiB string-builder budget, regex program-size guard, and 2 GiB Arrow-batch ceiling associated with 32-bit string offsets; known violations refuse construction, data-dependent violations may fail at runtime | Resource restrictions, not DuckDB nondeterminism; individual budgets and enforcement need their stated evidence |
| **exclusion: optimizer-on-answers** | Target the optimizer-off oracle, not optimizer rewrites that can elide a trapping expression | Fixed reference choice, decided 2026-08-17 |
| **exclusion: statistics-dependent-kernels** | The measured `ILIKE`/embedded-NUL case serves NUL-transparent behavior instead of reproducing a kernel choice affected by other rows | Measured non-row-local dependency; a served difference whose disposition remains in the ledger, not a constructor refusal |

Batch dependence defines the semantic boundary. Explicit multiplicity is a
separate API choice. Neither justifies blanket bans on otherwise row-local syntax.

Evidence is partial: `tests/test_shape_contract.py` checks multiplicity;
`tests/known_divergences/test_string_budget.py` checks literal-count refusal;
`tests/known_divergences/test_trap_elision.py` checks the optimizer gap. `tests/test_known_limitations.py`
covers some refusals, not the entire inventory. No executable Arrow-ceiling
check is identified here. The [divergence ledger](../oracle/07-the-divergence-ledger.md)
records the corresponding measured differences and unresolved dispositions.
Paths in this paragraph are relative to `packages/confit/`.

### Scope classification

**ask: exclusion-ratification — RULED.** Classify cases as:

1. **Outside the model:** depends on other request rows, or reads no request table.
2. **Unimplemented:** potentially valid within the model; a gap, not a permanent exclusion.
3. **Explicit product/resource restriction:** deliberately excluded for a stated reason.
4. **Invalid input:** malformed SQL or an inconsistent caller declaration, not a missing feature.

Do not count invalid declarations as ordinary unsupported SQL when measuring
product acceptance. Individual restrictions still need evidence; the classification
does not make the current refusal inventory a permanent product definition.

Implementation priorities are separate: **ask: next-query-classes** and the gap
inventory live in the [baseline report](../reports/2026-09-02-goal-baseline.md).

### Restriction inventory by class

Each refusal family in [known limitations](../known-limitations.md) and the campaign's
refusal classes, sorted into the four classes above. A row states the class and its
ground; it does not ratify the restriction, and a class-2 row is a gap to close, not a
product boundary. "DuckDB rejects" was measured on DuckDB 1.5.5 (2026-09-26).

| Family | Class | Ground |
|---|---|---|
| Aggregation, `GROUP BY`/`HAVING`, windows, `QUALIFY` over request rows | 1 outside | answer depends on sibling request rows |
| `ORDER BY`, `LIMIT`/`OFFSET`/`FETCH`/`TOP`, `DISTINCT` over request rows | 1 outside | order, count or identity across the batch |
| Queries reading no request table (static-only), table functions as the driving relation | 1 outside | no request row to specialize on; the static-only path still serves today (claim: one-door-bypass) |
| `FULL OUTER JOIN`, `rowid` | 1 outside | emits rows no request row produced / identifies a row by batch position |
| CTEs, subqueries (incl. derived tables), set operations, named windows used row-locally | 2 unimplemented | DuckDB serves them; blanket syntax bans, not scope |
| Row-local `USING`/`NATURAL` self-joins, more than one join under `shape='many'` | 2 unimplemented | named follow-ups |
| Decimal expressions and decimal-literal arithmetic (`UNSHIPPED` decimals) | 2 unimplemented | m-8 lattice phase 5 |
| `f32` row columns, lists, whole-struct output, bracket field access, `HUGEINT`/unsigned | 2 unimplemented | type-lattice width not built |
| `decimal256` static columns | 4 invalid | DuckDB refuses them at Arrow registration |
| Narrow-integer overflow trap on the row path | 2 unimplemented | m-8 phase 3 |
| All-NULL `CASE`/`COALESCE`/`least`/`greatest`, bare `NULL` as `repeat`'s string (BLOB) | 2 unimplemented | DuckDB binds them; the BLOB type is not built |
| `^`, prefix `~`, `#`, `NOT GLOB`; `COLUMNS(...)` in expressions; `* EXCLUDE (t.key)` on `USING`; mixed string/number `BETWEEN`/`IN` | 2 unimplemented | parser precedence or binding not reproduced; refused rather than served wrong |
| Regex reject list and fuzzer-found regex classes | 2 unimplemented | RE2 semantics not reproduced by rust-regex on these constructs; DuckDB-self-inconsistent cases stay refused for that reason |
| Non-constant regex patterns, replacements, options, group indexes | 3 restriction | specialization-inherent: per-row compilation (exclusion: per-row-general-work) |
| 1 GiB string-builder budget, regex program-size guard, 2 GiB Arrow batch | 3 restriction | resource (exclusion: resource-ceilings) |
| Default-shape refusals: duplicate join keys, inner joins and `WHERE` under `shape='map'` | 3 restriction | explicit multiplicity API; `shape='many'`/`'filter'` opt in |
| Static-only row limits | 3 restriction | nondeterministic frozen answer (TASK-128); also class 1 as a static-only query |
| Scalar calls with `OVER`/`FILTER`/`IGNORE NULLS`/`WITHIN GROUP`; `SIMILAR TO … ESCAPE`; `QUALIFY` without a window | 4 invalid | DuckDB rejects |
| `bind error:` refusals: unknown or ambiguous column, constant cast or constant overflow DuckDB errors on at plan time | 4 invalid | wrong against the declared schema, or DuckDB rejects |
| Static table not provided; UDF declarations inconsistent with their use (width-1 list return, argument width) | 4 invalid | inconsistent caller declaration |
| `parse error:` on malformed SQL | 4 invalid | not SQL |
| `parse error:` on DuckDB-valid SQL outside the accepted dialect | 2 unimplemented | dialect surface, not scope |

Acceptance measurements count class 4 apart from classes 1-3, as the scope rule above
requires. The campaign reports refusals by DuckDB's outcome for the same query (claim:
refusal-outcome-reporting): a refusal whose query DuckDB rejects is class 4 by
measurement, whatever its message prefix.

## UDF and model boundary

Confit owns the UDF protocol, external calls, structured result access, and
native tree scoring. `sql_transform` owns fitting, clone-per-group behavior,
estimator packing, and comparison with sklearn.

UDFs must be deterministic. Each declares `name`, an Arrow `takes` schema, an
Arrow `returns` type, and a scalar `__call__` returning output lanes as a tuple,
or `None` for an all-NULL result. `instances` marks a fitted transformer and adds
an implicit leading nullable-BIGINT instance argument, outside `takes`.

Struct returns have named lanes; fixed-size-list returns have unnamed lanes.
A width-one list must instead declare its scalar element type. A UDF exposing
`tree_tables()` supplies `(nodes, models, compare_grid)` for native scoring
without a Python call on the row path. The protocol is specified by
`packages/confit/confit/_engine.pyi` and exercised by `test_udfs.py` and
`test_tree_predict.py`.

**claim: udf-parity-is-still-the-oracle.** When DuckDB can execute the operation,
register the same UDF for oracle comparison. A UDF is not an exemption from SQL
parity. Builtin-name collisions refuse because the engines would resolve the
call differently (`tests/test_udfs.py::udf_check` and
`::test_a_udf_may_not_take_a_builtin_name`). Where DuckDB cannot execute a model
operation, the independent reference and bounds are specified under
[C4: transformer parity](success-measures.md#transformer-parity-c4).
