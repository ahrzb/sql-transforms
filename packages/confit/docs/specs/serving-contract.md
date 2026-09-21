# Serving contract

The [goal](../goal.md) defines the target. This document supplies the API and
scope details; the [oracle](../oracle/README.md) defines SQL compatibility.
Current refusals are not automatically permanent product exclusions.

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
| `output_schema` | Output names, Arrow types, and field order |
| `infer_rows(rows)` | Dict-or-object rows in, dict rows out |
| `infer_arrow(batch)` | Arrow table in, Arrow table out |
| `backend`, `boundary` | Execution and Python-boundary diagnostics, not caller-selected modes |

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

The following restrictions retain their existing evidence and proposed grounds.
The general exclusion classification awaits **ask: exclusion-ratification**;
explicit reference and scope decisions above are not reopened by that question.

| Restriction | Rule or current behavior | Ground and status |
|---|---|---|
| **exclusion: per-row-general-work** | Refuse constructs requiring general compilation or binding per row; regex patterns, replacements, options, and group indexes must meet construction-time requirements | Specialization-inherent under the goal; complete enforcement not established |
| **exclusion: resource-ceilings** | 1 GiB string-builder budget, regex program-size guard, and 2 GiB Arrow-batch ceiling associated with 32-bit string offsets; known violations refuse construction, data-dependent violations may fail at runtime | Proposed resource grounds; these are Confit limits, not DuckDB nondeterminism |
| **exclusion: optimizer-on-answers** | Target the optimizer-off oracle, not optimizer rewrites that can elide a trapping expression | Reference choice decided 2026-08-17; proposed exclusion ground `scope-by-product-decision` |
| **exclusion: statistics-dependent-kernels** | The measured `ILIKE`/embedded-NUL case serves NUL-transparent behavior instead of reproducing a kernel choice affected by other rows | Proposed specialization-inherent ground; a served difference, not a constructor refusal |

The proposed ground for whole-relation shapes and explicit multiplicity is
`scope-by-product-decision`. The batch-dependence boundary itself is already
decided; blanket syntax bans need justification beyond that boundary.

Evidence is partial: `tests/test_shape_contract.py` checks multiplicity;
`tests/known_divergences/test_string_budget.py` checks literal-count refusal;
`tests/known_divergences/test_trap_elision.py` checks the optimizer gap. `tests/test_known_limitations.py`
covers some refusals, not the entire inventory. No executable Arrow-ceiling
check is identified here. The [divergence ledger](../oracle/07-the-divergence-ledger.md)
records the corresponding measured differences and unresolved dispositions.
Paths in this paragraph are relative to `packages/confit/`.

### Open scope decision

**ask: exclusion-ratification.** Ratify permanent restrictions and their grounds
without turning missing features into exclusions. Which syntax bans are independent
product choices? Which limits need checks? How should invalid SQL and caller-declaration
errors be classified and counted in acceptance rates? Those errors do not fit the
three proposed grounds for product exclusions.

Implementation priorities are separate: **ask: next-query-classes** and the gap
inventory live in the [baseline report](../reports/2026-09-02-goal-baseline.md).

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
