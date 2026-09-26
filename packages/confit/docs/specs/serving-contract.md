# Serving contract

The [goal](../goal.md) defines the target. This document supplies the API and
scope details; the [oracle](../oracle/README.md) defines SQL compatibility. The
rationale for the scope classification and output-nullability rule is in the
[oracle policy](../decisions/closed/oracle-policy.md).

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
need not equal DuckDB's inferred flag. The [comparison contract](../oracle/05-the-comparison-contract.md#output-names-and-schemas)
lists the checks.

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

`backend` is `cranelift`, `interpreter`, or `constant`; `boundary` is
`marshaller`, `generic`, or `constant`. `constant` is the static-tables-only
path: DuckDB evaluates the query once at construction and the answer is frozen.

Construction refusal is a `ValueError` identifying the unsupported construct.
The [refusal specification](../oracle/04-verdicts-agreement-abstention-refusal.md)
defines the prefixes and scope grounds. Construction success does not rule out
data-dependent runtime traps.

## Scope and restrictions

**exclusion: whole-relation-shapes.** Splitting or combining a request batch must
not change a request row's answer. Grouping, aggregation, sorting, distinctness,
limits, and windows that depend on sibling request rows are outside the model.
Aggregating frozen rows matched by one request row is inside it, as is reducing
a list stored in that row. Queries reading no request table are outside it.
The rationale is in the [scope decision](../decisions/closed/static-only-queries.md).

Syntax is not the scope test. The frontend also refuses CTEs, subqueries, set
operations, `HAVING`, `QUALIFY`, named windows, `OFFSET`/`FETCH`/`TOP`, multiple
statements, table functions in `FROM`, `rowid`, `FULL OUTER JOIN`, and scalar calls
with aggregate/window modifiers. A CTE, for example, can describe a row-local
computation; such a use is inside the model even though the frontend refuses it.

| Restriction | Rule | Ground |
|---|---|---|
| **exclusion: per-row-general-work** | Refuse constructs requiring general compilation or binding per row; regex patterns, replacements, options, and group indexes must be construction-time constants | Specialization-inherent under the goal |
| **exclusion: resource-ceilings** | 1 GiB string-builder budget, regex program-size guard, and 2 GiB Arrow-batch ceiling associated with 32-bit string offsets; known violations refuse construction, data-dependent violations fail at runtime | Resource restrictions, not DuckDB nondeterminism |
| **exclusion: optimizer-on-answers** | Target the optimizer-off oracle, not optimizer rewrites that can elide a trapping expression | Fixed reference choice |
| **exclusion: statistics-dependent-kernels** | The measured `ILIKE`/embedded-NUL case serves NUL-transparent behavior instead of reproducing a kernel choice affected by other rows | Measured non-row-local dependency; a served difference recorded in the ledger, not a constructor refusal |

Batch dependence defines the semantic boundary. Explicit multiplicity is a
separate API choice.

Evidence: `tests/test_shape_contract.py` checks multiplicity;
`tests/known_divergences/test_string_budget.py` checks literal-count refusal;
`tests/known_divergences/test_trap_elision.py` checks optimizer-elided traps.
`tests/test_known_limitations.py` covers some refusals, not the entire inventory.
The [divergence ledger](../oracle/07-the-divergence-ledger.md) records the
corresponding measured differences. Paths in this paragraph are relative to
`packages/confit/`.

### Scope classification

Every refusal falls in one class:

1. **Outside the model:** depends on other request rows, or reads no request table.
2. **Inside the model, refused by this engine:** valid within the model; not a product exclusion.
3. **Explicit product/resource restriction:** deliberately excluded for a stated reason.
4. **Invalid input:** malformed SQL or an inconsistent caller declaration, not a missing feature.

Product-acceptance measurements count class 4 apart from classes 1-3.

### Restriction inventory by class

Each refusal family in [known limitations](../known-limitations.md) and the campaign's
refusal classes, sorted into the four classes above. "DuckDB rejects" is measured on
DuckDB 1.5.5.

| Family | Class | Ground |
|---|---|---|
| Aggregation, `GROUP BY`/`HAVING`, windows, `QUALIFY` over request rows | 1 outside | answer depends on sibling request rows |
| `ORDER BY`, `LIMIT`/`OFFSET`/`FETCH`/`TOP`, `DISTINCT` over request rows | 1 outside | order, count or identity across the batch |
| Queries reading no request table (static-only), table functions as the driving relation | 1 outside | no request row to specialize on; the static-only path serves them by one construction-time DuckDB evaluation (claim: one-door-bypass) |
| `FULL OUTER JOIN`, `rowid` | 1 outside | emits rows no request row produced / identifies a row by batch position |
| CTEs, subqueries (incl. derived tables), set operations, named windows used row-locally | 2 inside, refused | DuckDB serves them; syntax refusals, not scope |
| More than one join under `shape='many'` | 2 inside, refused | named rejection |
| Decimal expressions and decimal-literal arithmetic | 2 inside, refused | exact decimal arithmetic not reproduced |
| `f32` row columns, lists, whole-struct output, bracket field access, `HUGEINT`/unsigned | 2 inside, refused | type not served |
| `decimal256` static columns | 4 invalid | DuckDB refuses them at Arrow registration |
| All-NULL `CASE`/`COALESCE`/`least`/`greatest`, bare `NULL` as `repeat`'s string (BLOB) | 2 inside, refused | DuckDB binds them; the engine has no BLOB type |
| `^`, prefix `~`, `#`, `NOT GLOB`; `COLUMNS(...)` in expressions; `* EXCLUDE (t.key)` on `USING`; mixed string/number `BETWEEN`/`IN` | 2 inside, refused | parser precedence or binding not reproduced; refused rather than served wrong |
| Regex reject list and fuzzer-found regex classes | 2 inside, refused | RE2 semantics not reproduced by rust-regex on these constructs; DuckDB-self-inconsistent cases are refused for that reason |
| Non-constant regex patterns, replacements, options, group indexes | 3 restriction | specialization-inherent: per-row compilation (exclusion: per-row-general-work) |
| 1 GiB string-builder budget, regex program-size guard, 2 GiB Arrow batch | 3 restriction | resource (exclusion: resource-ceilings) |
| Default-shape refusals: duplicate join keys, inner joins and `WHERE` under `shape='map'` | 3 restriction | explicit multiplicity API; `shape='many'`/`'filter'` opt in |
| Static-only row limits | 3 restriction | nondeterministic frozen answer; also class 1 as a static-only query |
| Scalar calls with `OVER`/`FILTER`/`IGNORE NULLS`/`WITHIN GROUP`; `SIMILAR TO … ESCAPE`; `QUALIFY` without a window | 4 invalid | DuckDB rejects |
| `bind error:` refusals: unknown or ambiguous column, constant cast or constant overflow DuckDB errors on at plan time | 4 invalid | wrong against the declared schema, or DuckDB rejects |
| Static table not provided; UDF declarations inconsistent with their use (width-1 list return, argument width) | 4 invalid | inconsistent caller declaration |
| `parse error:` on malformed SQL | 4 invalid | not SQL |
| `parse error:` on DuckDB-valid SQL outside the accepted dialect | 2 inside, refused | dialect surface, not scope |

The campaign reports refusals by DuckDB's outcome for the same query (claim:
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
the same UDF is registered for oracle comparison. A UDF is not an exemption from SQL
parity. Builtin-name collisions refuse because the engines would resolve the
call differently (`tests/test_udfs.py::udf_check` and
`::test_a_udf_may_not_take_a_builtin_name`). Where DuckDB cannot execute a model
operation, the independent reference and bounds are specified under
[C4: transformer parity](success-measures.md#transformer-parity-c4).
