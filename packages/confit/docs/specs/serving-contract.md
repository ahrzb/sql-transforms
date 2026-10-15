# Serving contract

The [goal](../../../../loops/confit/goal.md) defines the target. This document supplies the API and
scope details; the [oracle](../oracle/README.md) defines SQL compatibility. The
rationale for the scope classification and output-nullability rule is in the
[oracle policy](../../../../loops/confit/decisions/closed/oracle-policy.md).

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

`backend` is `cranelift` or `interpreter`; `boundary` is `marshaller` or
`generic`.

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
The rationale is in the [scope decision](../../../../loops/confit/decisions/closed/static-only-queries.md).

Confit also refuses these forms: a CTE read twice, the subqueries that the
[restriction inventory](#restriction-inventory-by-class) does not list as
served, set operations, `HAVING`, `QUALIFY`, named windows,
`OFFSET`/`FETCH`/`TOP`, multiple statements, table functions in `FROM`,
`rowid`, `FULL OUTER JOIN`, and calls of scalar functions with `FILTER`,
`OVER`, `IGNORE NULLS` or `WITHIN GROUP`. Some of these forms are inside the
model, because the answer for one request does not depend on other requests.
For example, a CTE read twice can describe a row-local computation. So the
syntax of a form does not decide whether it is inside the model.

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
| Queries reading no request table (static-only), table functions as the driving relation | 1 outside | no request row to specialize on; refused at construction |
| `FULL OUTER JOIN`, `rowid` | 1 outside | emits rows no request row produced / identifies a row by batch position |
| Derived tables and CTEs read once, over the request table, row-local, driving FROM (joins to static tables beside them included), at any depth | 2 inside, served | one pipeline stage per level ([design](2026-09-26-row-local-subqueries-design.md)) |
| A scalar subquery over one static table that selects an aggregate or an expression over aggregates, keyed by equalities with the outer query and filtered by terms that read only the static table ([claim: scalar-subquery-as-join](#scalar-subqueries-over-a-static-table)) | 2 inside, served | an aggregate, or the expression, over the static rows that one request matches, read through a join to a table that confit computes at build |
| A scalar subquery of another form, for example with a DOUBLE `sum` or `avg`, or with an item that calls a UDF. The [claim](#scalar-subqueries-over-a-static-table) lists the served forms. | 2 inside, refused | refused by name; a DOUBLE `sum` or `avg` waits on a [decision record](../../../../loops/confit/decisions/open/double-sum-bound.md) about its parity bound |
| A CTE read twice or joined beside another relation, static-only derived tables and CTEs, `WITH RECURSIVE`, other subqueries, set operations, named windows used row-locally | 2 inside, refused | DuckDB serves them; syntax refusals, not scope |
| An expression over a constant-NULL derived-table column | 2 inside, refused | DuckDB keeps it SQLNULL-typed across the level; confit does not model that type |
| Under `shape='many'`: (1) a self-join after another join; (2) a join or a scalar subquery over a static table, in a query that also has a derived table or a CTE | 2 inside, refused | refused by name; without a derived table or a CTE, any number of joins serves, with a self-join only as the first join |
| A comma join whose WHERE has a term that can trap before a key term, or a key term that can trap out of order | 2 inside, refused | DuckDB evaluates the WHERE terms in written order over the cross product, and confit probes with the key terms first; refused by name; key terms first, or `JOIN ... ON`, serve |
| Casts into DECIMAL from DOUBLE/VARCHAR/BOOLEAN, DECIMAL join key expressions against a non-DOUBLE key, `IN`/`BETWEEN` capped at 38 digits | 2 inside, refused | named rejection; decimal expressions otherwise serve exactly |
| `f32` row columns, lists, `UHUGEINT` | 2 inside, refused | type not served |
| A struct value in a derived table or a CTE, a CASE over structs of different types, `IS NULL` over a struct that the query builds | 2 inside, refused | DuckDB serves them. A derived table carries scalar columns only. Confit does not convert struct types to a common type, and does not build the fields only to test the struct. |
| `decimal256` static columns | 4 invalid | DuckDB refuses them at Arrow registration |
| All-NULL `CASE`/`COALESCE`/`least`/`greatest`, bare `NULL` as `repeat`'s string (BLOB) | 2 inside, refused | DuckDB binds them; the engine has no BLOB type |
| `^`, prefix `~`, `#`, `NOT GLOB`; `COLUMNS(...)` in expressions; `* EXCLUDE (t.key)` on `USING`; mixed string/number `BETWEEN`/`IN` | 2 inside, refused | parser precedence or binding not reproduced; refused rather than served wrong |
| Regex reject list and fuzzer-found regex classes | 2 inside, refused | RE2 semantics not reproduced by rust-regex on these constructs; DuckDB-self-inconsistent cases are refused for that reason |
| Non-constant regex patterns, replacements, options, group indexes | 3 restriction | specialization-inherent: per-row compilation (exclusion: per-row-general-work) |
| 1 GiB string-builder budget, regex program-size guard, 2 GiB Arrow batch | 3 restriction | resource (exclusion: resource-ceilings) |
| Default-shape refusals: duplicate join keys, inner joins and `WHERE` under `shape='map'` | 3 restriction | explicit multiplicity API; `shape='many'`/`'filter'` opt in |
| Scalar calls with `OVER`/`FILTER`/`IGNORE NULLS`/`WITHIN GROUP`; `SIMILAR TO … ESCAPE`; `QUALIFY` without a window | 4 invalid | DuckDB rejects |
| `bind error:` refusals: unknown or ambiguous column, constant cast or constant overflow DuckDB errors on at plan time | 4 invalid | wrong against the declared schema, or DuckDB rejects |
| Static table not provided; UDF declarations inconsistent with their use (width-1 list return, argument width) | 4 invalid | inconsistent caller declaration |
| `parse error:` on malformed SQL | 4 invalid | not SQL |
| `parse error:` on DuckDB-valid SQL outside the accepted dialect | 2 inside, refused | dialect surface, not scope |

The campaign reports refusals by DuckDB's outcome for the same query (claim:
refusal-outcome-reporting): a refusal whose query DuckDB rejects is class 4 by
measurement, whatever its message prefix.

### Scalar subqueries over a static table

**claim: scalar-subquery-as-join.** Confit serves a scalar subquery over one
static table `s` that selects an aggregate:

```sql
SELECT t.id, (SELECT sum(s.v) FROM s WHERE s.k = t.k) AS total FROM __THIS__ AS t
```

The aggregate is `count(*)`, `count` of a column or of a constant that is not
NULL, `min` or `max` of a column, `sum` of an integer, BOOLEAN or DECIMAL
column, or `avg` of an integer or DECIMAL column. On a platform other than
x86_64 Linux, confit serves `avg` only for SMALLINT and for a DECIMAL with a
precision of 4 or less (see below). The SELECT item can also be an expression
over such aggregates, for example `max(s.v) + 1` or
`coalesce(sum(s.v), 0) + count(*)`. Then the expression reads the columns of
`s` only inside its aggregates. If it reads a column outside them, the oracle
stops with a binder error. Confit refuses an expression that holds a subquery
or calls a SQL function, a UDF or a tree model.

Each term of the WHERE has one of two forms. A key compares a column of `s`
with an expression of the outer query, its outer expression, by `=` or
`IS NOT DISTINCT FROM`. Each other term reads only `s`, such as `s.v > 5`.
The WHERE can also be absent. The outer expression of the first key in the
written order can trap (see below). Confit refuses the subquery if the outer
expression of a key after the first can trap, or if a term that reads only
`s` can trap. It also refuses an outer expression that reads a table that the
outer query joins, and a term that reads only `s` but holds a subquery or
calls a SQL function.

At build, confit keeps the rows of `s` on which each term that reads only `s`
is TRUE. It puts these rows into groups with equal keys and computes the
aggregate of each group. The keys are the columns of `s` that the WHERE
compares with the outer query. If no term compares one, all the rows that
confit keeps make one group. If the item is an expression, confit computes it
over the aggregates of each group. A request reads the value of the group
that it matches, through a LEFT join. If a request matches no group, it reads
0 for `count` and NULL for the other aggregates. If the item is an
expression, the request reads the expression over these values. Confit
computes the value of this expression once, at build.

The oracle evaluates the WHERE of the subquery in written order, for each
pair of a distinct outer value and a row of `s`. It evaluates the argument of
the aggregate for the matched rows only. It evaluates the item once for each
distinct outer value, over the aggregates of the matched rows. If no row
matches, it evaluates the item over the aggregates of no rows. The pins at the
end of `tests/test_scalar_subquery.py` record these facts. The order of
evaluation matters only where an evaluation can trap. In a served subquery,
only the outer expression of the first key can trap. The argument of the
aggregate, if it has one, is a column or a constant. Only the running total of
a `sum` or an `avg` can overflow. Confit computes it at build, and it refuses
the build if the total can overflow (the build refusals at the end of this
section list this case). An expression item can also raise an error, for
example `max(s.v) + 1` if `s.v` is a TINYINT column with the value 127.
Confit evaluates the expression at build, so a served subquery does not trap
in it. Confit refuses the build if the expression raises an error:

- If the expression raises an error over the aggregates of no rows, for
  example `ln(count(*))`, confit refuses the build, whatever the rows of `s`
  are. The oracle evaluates the expression for each request that matches no
  group.
- If the expression raises an error over the aggregates of a group, confit
  refuses the build. The build refusals at the end of this section list this
  case too. The oracle traps for each request that matches this group. If the
  subquery has no key, the oracle traps each time it runs the query, even for
  a batch with no requests.

The oracle evaluates the outer expression of the first key only if a row of
`s` passes the terms written before that key. The planner of the oracle
moves the upper bound of `x BETWEEN lo AND hi` to the end of the WHERE, also
with the optimizer off. So if such a term comes before the key, only its
lower bound decides if the oracle evaluates the outer expression. At build,
confit counts the rows of `s` that pass these terms. For each request, it
evaluates the outer expression only if this count is above 0. The oracle also
runs a subquery in the SELECT list for each row that the outer WHERE passes,
and a subquery in the WHERE for each row that reaches the WHERE. The oracle
does this also for a subquery inside a CASE, whatever the condition of the
CASE is. Confit evaluates the outer expression of the first key for the same
rows. It refuses the subquery in two cases:

- The subquery is in the WHERE, and the outer query has a comma join whose
  key is a term of the WHERE. The oracle runs the subquery for every pair of
  the comma join, and confit only for the pairs that the key matches.
- The query runs under `shape='many'`. There confit evaluates every join
  before the WHERE.

The answer of `avg` is a DOUBLE. The oracle adds the values that are not NULL
without rounding, and it counts them. If a group has no such value, the answer
is NULL. The oracle divides only at the end, once for each group.

For an integer type, the divisor is the count. For a DECIMAL with k digits
after the point, the divisor is the count times the DOUBLE nearest to 10^k. The
type of the argument sets how the oracle divides:

- For SMALLINT and for a DECIMAL with a precision of 4 or less, it divides in
  `double`. The answer is the exact quotient, rounded once to a DOUBLE.
- For TINYINT, INTEGER, BIGINT, HUGEINT, the unsigned integer types and a
  DECIMAL with a precision of 5 or more, it divides in the C type
  `long double`.

On x86_64 Linux, `long double` is the 80-bit extended format of the x87
floating-point unit, with a significand of 64 bits. There the oracle computes
the answer in four steps:

1. It rounds the exact sum to 64 bits.
2. It rounds the divisor to 64 bits.
3. It divides the result of step 1 by the result of step 2, and it rounds the
   quotient to 64 bits.
4. It rounds this quotient to a DOUBLE.

Each rounding goes to the nearest value, and a tie goes to the even value. So
the answer can differ from the exact quotient rounded once to a DOUBLE.

Confit computes the same four steps exactly with integers, in the module
`src/duckdb/avg.rs`. It holds each value as an integer significand and a power
of two. Thus its answer does not depend on the floating point of the machine
that runs confit. On a platform other than x86_64 Linux, `long double` has
another format, for example IEEE quadruple precision on 64-bit ARM Linux.
There confit refuses to build an `avg` that the oracle divides in
`long double`.

These facts come from the source code of DuckDB 1.5.5, in its file
`extension/core_functions/aggregate/algebraic/avg.cpp`, and from measurements
on x86_64 Linux. The unit tests of `src/duckdb/avg.rs` hold measured answers.

If an `avg` reads a BOOLEAN or VARCHAR column, the oracle stops with the error
`No function matches the given name and argument types 'avg(BOOLEAN)'` (or
`'avg(VARCHAR)'`). Confit stops the build with the same error message.

Confit also refuses the build, with an error that names the reason, in three
cases that depend on the rows of `s`:

- The sum of the absolute values of a group is past the range of the type in
  which the oracle keeps the sum. For `sum`, this is the range of its result
  (HUGEINT, a 128-bit integer, or a DECIMAL with 38 digits). For `avg`, this
  is the range of HUGEINT. Then a partial sum can overflow in some order of
  the additions.
- A DOUBLE `min` or `max` finds two values that compare as equal but have
  different bits, such as `-0.0` and `0.0`, or two NaN values. Then the answer
  of the oracle depends on the order of the rows.
- An expression item raises an error over the aggregates of a group.

A SELECT item can read the alias of an earlier item. The oracle rejects such
a read if the earlier item holds a subquery. Its error starts with
`Alias "c" referenced in a SELECT clause - but the expression has a subquery`,
and confit gives the same error. A WHERE can read such an alias in the oracle
and in confit.

Evidence: `tests/test_scalar_subquery.py`, and campaign cases. The campaign
uses a generator of random SQL (`fuzz/gen.py`). For each seed whose remainder
after division by 43 is 33, the generator tries to add this form to the case.
If it adds the form, it tags the case `scalar-subquery`. About one subquery in
ten that it adds selects an expression over aggregates. It adds up to three
subqueries to a case, so about one case in six of those tagged
`scalar-subquery` has such an expression.

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
without a Python call on the row path. `confit.functions` spells the
protocol as classes: `SqlFunction` (a SQL body, registered on DuckDB as a
macro; a call is replaced by its body with the arguments substituted, as
DuckDB binds a macro), `ExternFunction` (a callable) and `Ensemble` (packed
tree tables plus their reference walk), each with a `register` that is its
oracle definition (`tests/test_functions.py`, `tests/test_sql_functions.py`).
An object exposing `sql_body` is a SQL function. It can also expose
`sql_lets` and `sql_let_body`, and confit then reads these two attributes
instead of `sql_body`. `sql_lets` is a list of SQL texts. Each text is a value
that the body reads more than once. `sql_let_body` and each later text read
value `i` as `__cf_let(i)`. Confit expands each value once for each call, so a
recurrence whose steps read the step before twice does not double in size at
each step. The definition does not change: `sql_body` is `sql_let_body` with
each `__cf_let(i)` replaced by the text of value `i`, and DuckDB runs
`sql_body` (`tests/test_sql_function_lets.py`). Confit reads each value as one
operand, as if it were in parentheses. So each text must also be one operand
where it stands in `sql_body`. `SqlFunction` writes each value as one operand. The protocol is specified by
`packages/confit/confit/_engine.pyi` and exercised by `test_udfs.py` and
`test_tree_predict.py`.

**claim: udf-parity-is-still-the-oracle.** When DuckDB can execute the operation,
the same UDF is registered for oracle comparison. A UDF is not an exemption from SQL
parity. Builtin-name collisions refuse because the engines would resolve the
call differently (`tests/test_udfs.py::udf_check` and
`::test_a_udf_may_not_take_a_builtin_name`). Where DuckDB cannot execute a model
operation, the independent reference and bounds are specified under
[C4: transformer parity](success-measures.md#transformer-parity-c4).

### Tree tables

A UDF that exposes `tree_tables()` is scored natively. sql-transform's
`TreeBasedTransform` is one packer ([tree models](../../../sql-transform/spec/python/tree-models.md)).

#### The compare grid

Both of the above are sklearn's semantics, and neither is universal — so a
transform says which floating-point grid its comparisons live on, as the third
member of `tree_tables()`:

```python
def tree_tables(self):
    return nodes, models, "float32"
```

A class
wrapping a library that compares in float64 skips the threshold rewrite and
declares `"float64"`, and the engine then converts its integer features
exactly rather than narrowing them.

Without the field the narrowing would fire for every model, which would make
the wire format quietly sklearn-specific: a float64-grid packer would get its
integer features narrowed anyway, silently losing precision above `2**24` that
it had every right to keep. The threshold rewrite is skippable by a packer;
the conversion is not, so the engine has to be told.

**It is required, not defaulted.** The packer that would get this wrong is
exactly the one that never thought about it, and a default would be the same
trap with an extra step.

**It belongs to the TRANSFORM, not to an instance inside it.** `score(id, ..)`
takes the instance id from a row, so it is a runtime value, while the
conversion is chosen once when the query is lowered. A per-instance grid could
only be honoured with a per-row branch.

That opcode adds no float32 TYPE — its lane is f64 out, the same way
`ftoi.nearest` is a rounding mode and not an integer type. The engine still
computes in exactly `i64` / `f64` / string / bool, and no cast lands on the
row path.

#### Writing a packer

For a library `TreeBasedTransform` does not know, write your own transform
class. The whole protocol is four attributes and one method — the engine never
sees sklearn, and never calls `__call__` on a class that has `tree_tables`.

```python
class XGBTransform:
    name = "score"
    # names + types in one declaration; the features bind by position
    takes = pa.schema([("price", pa.float64()), ("sqft", pa.float64())])
    returns = pa.float64()
    instances = {0: booster}    # presence is what adds the leading id argument

    def tree_tables(self):
        return nodes_table, header_table, "float64"   # or "float32" if your
        # thresholds were rewritten onto sklearn's grid — see above
```

**`nodes`** — one row per node, grouped by model then tree:

| column | type | |
|---|---|---|
| `model_id` | int64 | dense from 0; a model's rows are contiguous |
| `tree_id` | int64 | a tree's rows are contiguous |
| `node_id` | int64 | dense from 0 **within each tree** |
| `feature` | int32 | `-1` marks a leaf |
| `threshold` | float64 | `feature <= threshold` goes left |
| `left`, `right` | int32 | tree-local node ids, `-1` on a leaf |
| `missing_left` | bool | where `NaN` goes, per node |
| `value` | float64 | the leaf's contribution |

**`models`** — one row per model:

| column | type | |
|---|---|---|
| `model_id` | int64 | dense from 0 |
| `base` | float64 | seeds the accumulator for `sum`, added after for `mean` |
| `agg` | string | `"sum"` or `"mean"` |
| `link` | string | `"identity"` or `"sigmoid"` |

int32 and int64 are both accepted for the id, child and feature columns.
A NULL anywhere in either table is a build error naming the row.

Decoding walks the pyarrow buffers directly — a 100k-node forest costs no
Python objects.

#### Build-time refusals

Each names the offending row or field, before any data flows:

- a child index out of range, or a child that does not follow its parent
  (that ordering is what makes traversal provably terminate);
- a node unreachable from its tree's root;
- a leaf with children, or a split node missing one;
- `feature` beyond the declared width;
- a node id out of dense order;
- an unknown `agg` or `link` spelling;
- non-dense or non-contiguous `model_id`, a model with no nodes, an empty
  model table;
- instance ids that are not dense from 0;
- a call passing the wrong number of arguments, or a non-numeric one;
- two declared UDFs whose names collide case-insensitively, whichever kinds
  they are;
- a declared UDF whose name is a builtin (`least`, `round`, `upper`, …): the
  binder matches the builtin catalogue before it consults the declared UDFs,
  while DuckDB lets a registered function shadow its own builtin, so the two
  engines would answer the same SQL differently;
- a `tree_tables()` that raises or does not return
  `(nodes, models, compare_grid)`.

#### Which backend runs it

The kernel is native Rust either way: the interpreter calls it directly,
Cranelift emits a call to an `extern "C"` shim over the same routine. Scoring
cost is identical; only the surrounding row code differs.

Note that `DuckDBInferFn` **discards the Cranelift compile error and falls
back to the interpreter silently**. If you are measuring, assert the engine:

```python
assert fn.backend == "cranelift"
```

`SPECIALIZER_FORCE_INTERP=1` pins the interpreter.
