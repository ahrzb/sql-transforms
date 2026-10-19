# Serving contract

The [goal](../../../../mandates/confit/README.md#objective) defines the target. This document supplies the API and
scope details; the [oracle](../oracle/README.md) defines SQL compatibility. The
rationale for the scope classification and output-nullability rule is in the
[oracle policy](../../../../proposals/closed/oracle-policy.md).

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
The rationale is in the [scope decision](../../../../proposals/closed/static-only-queries.md).

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
| Aggregation, `GROUP BY`/`HAVING`, windows, `QUALIFY` over request rows | 1 outside | answer depends on sibling request rows, also for a join to a static table with `GROUP BY` ([ruling](../../../../proposals/closed/join-group-by-per-request.md)) |
| `ORDER BY`, `LIMIT`/`OFFSET`/`FETCH`/`TOP`, `DISTINCT` over request rows | 1 outside | order, count or identity across the batch |
| Queries reading no request table (static-only), table functions as the driving relation | 1 outside | no request row to specialize on; refused at construction |
| `FULL OUTER JOIN`, `rowid` | 1 outside | emits rows no request row produced / identifies a row by batch position |
| Derived tables and CTEs read once, over the request table, row-local, driving FROM (joins to static tables beside them included), at any depth | 2 inside, served | one pipeline stage per level ([design](2026-09-26-row-local-subqueries-design.md)) |
| A scalar subquery over one static table, keyed by equalities with the outer query and filtered by terms that read only the static table. It selects an aggregate, an expression over aggregates, or an expression over the columns of the one row that a request matches, or of the row that its ORDER BY and LIMIT keep. If one request reads at most 100 rows, confit also serves other terms that compare the static table with the outer query, and terms that can trap ([claim: scalar-subquery-as-join](#scalar-subqueries-over-a-static-table)) | 2 inside, served | an aggregate, or the expression, over the static rows that one request matches, read through a join to a table that confit computes at build, or computed by a scan for each request. A DOUBLE `sum` or `avg` compares within a bound ([claim: float-reduction-bound](../oracle/05-the-comparison-contract.md#floating-point-comparisons)) |
| A scalar subquery of another form, for example with an item that calls a UDF, or with a DOUBLE `sum` or `avg` that is not a whole output column. The [claim](#scalar-subqueries-over-a-static-table) lists the served forms. | 2 inside, refused | refused by name |
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

The aggregate is one of these:

- `count(*)`, and `count` of a column or of a constant that is not NULL;
- `min` or `max` of a column;
- `sum` of an integer, BOOLEAN, DECIMAL or DOUBLE column;
- `avg` of an integer, DECIMAL or DOUBLE column;
- `count_if` or `countif` of a BOOLEAN column, which counts the TRUE values;
- `bool_and` or `bool_or` of a BOOLEAN column;
- `bit_and`, `bit_or` or `bit_xor` of an integer column.

On a platform other than x86_64 Linux, confit serves `avg` only for SMALLINT,
for a DECIMAL with a precision of 4 or less, and for DOUBLE. The oracle divides
these in `double`, which is the same on each platform (see below). A DOUBLE
`sum` or `avg` must be the whole SELECT item of a subquery that is a whole
output column (see below).

The answer of `count_if` is a HUGEINT. The answer of `bit_and`, `bit_or` and
`bit_xor` has the type of the column. If no value of a group is other than
NULL, `count_if` is NULL and not 0, as `sum` is. The boolean and bitwise
aggregates are NULL too. If one of these aggregates reads a column of another
type, the oracle stops with a binder error, for example `No function matches
the given name and argument types 'bool_and(BIGINT)'`. Confit stops the build
with the same error message. These facts come from measurements on the oracle
on 2026-10-16.

An aggregate of a column can read each distinct value of the column once, for
example `count(DISTINCT s.v)`. Two values are one distinct value if the oracle
compares them as equal. So `-0.0` and `0.0` are one DOUBLE value, and all NaN
values are one value. The keyword `ALL`, as in `sum(ALL s.v)`, changes nothing.

An aggregate call can also sort its values with an ORDER BY inside the call,
for example `max(s.v ORDER BY s.j DESC)`. The oracle evaluates the ORDER BY
terms over the rows that pass the WHERE, and a term can trap there. So confit
accepts a term only if it is a column of `s`, because a column cannot trap.
For an aggregate over DISTINCT values, each
term must be the argument of the call. Else the oracle stops with the binder
error `In a DISTINCT aggregate, ORDER BY expressions must appear in the argument
list`, and confit gives the same error. If `count(*)` has an ORDER BY, the
oracle also stops with a binder error, and confit gives the same error.

The order of the values can change the answer of three kinds of aggregate.
Confit handles each kind in a way that does not depend on the order:

- A DOUBLE `sum` or `avg` can change in its last bits. The bound B of the
  comparison contract holds for each order of the additions (see below).
- A `sum` or `avg` can overflow in one order and not in another. Confit
  refuses the build if a partial sum can overflow in some order (see below).
- A DOUBLE `min` or `max` can give `-0.0` or `0.0`, or one of two NaN values.
  Confit refuses the build in this case (see below).

The SELECT item can also be an expression over such aggregates, for example
`max(s.v) + 1` or `coalesce(sum(s.v), 0) + count(*)`. Then the expression reads
the columns of `s` only inside its aggregates. If it reads a column outside
them, the oracle stops with a binder error. Confit refuses an expression that
holds a subquery or calls a SQL function, a UDF or a tree model.

The SELECT item can also be an expression over the columns of `s` with no
aggregate, for example `(SELECT s.v FROM s WHERE s.k = t.k)` or
`(SELECT s.v * 2 + s.j FROM s WHERE s.k = t.k)`. For a request, the oracle
evaluates this item over the row of `s` that passes the WHERE. If no row passes
the WHERE, the value is NULL, and the oracle does not evaluate the item. If more
than one row passes it, the oracle stops with the error
`More than one row returned by a subquery used as an expression`. Confit
evaluates the item at build, over the one row of each group (see below).
Confit also refuses these items:

- an item that reads a column of the outer query, such as `s.v + t.j`;
- an item that reads the whole row of `s`, such as `(SELECT s FROM s WHERE
  s.k = t.k)`, or a field or an element of a column of `s`;
- the item `NULL`, to which the oracle gives the type NULL, which confit does
  not have;
- an item that holds a subquery or calls a SQL function, a UDF or a tree model.

The subquery of such an item can also end with ORDER BY, LIMIT and OFFSET,
for example `(SELECT s.v FROM s WHERE s.k = t.k ORDER BY s.o DESC LIMIT 1)`.
For a request, the oracle sorts the rows that pass the WHERE by the ORDER BY
terms. Then it skips the first m rows for `OFFSET m`, and it keeps the next n
rows for `LIMIT n`. The value is the item over the row that it keeps. The value
is NULL if the oracle keeps no row. If the oracle keeps more than one row, it
stops with the error
`More than one row returned by a subquery used as an expression`. These rules
apply to the clauses:

- Each ORDER BY term is a column of `s` or the item. A term names the item by
  its position `1` or by its alias. If `s` has a column with the same name as
  the alias, a term with this name names the item, not the column, as in the
  oracle.
- Each term can have ASC or DESC, and NULLS FIRST or NULLS LAST. If a term
  does not say where NULL values go, they go last, under ASC and under DESC.
- The oracle sorts a NaN after all other DOUBLE values, and it sorts `-0.0`
  and `0.0` as equal values. It sorts VARCHAR values by their bytes, and FALSE
  before TRUE.
- LIMIT and OFFSET each take a whole number or NULL. `OFFSET NULL` skips no
  row. Without a key (see the next paragraph), `LIMIT NULL` and `LIMIT ALL`
  keep all the rows after the OFFSET.

The subquery has a key if its WHERE compares a column of `s` with the outer
query. With a key, the oracle keeps the rows whose number r in the sorted
order has m < r ≤ m + n. It computes m + n as a BIGINT, and this sum wraps
past the greatest BIGINT to a negative number. The oracle reads `LIMIT NULL`
and `LIMIT ALL` as the greatest BIGINT.

So with a key, `LIMIT NULL OFFSET 1` keeps no row, and the value is NULL for
each request. Without a key, `LIMIT NULL OFFSET 1` keeps all the rows after the
first. Confit gives the same values in both cases. The pin
`test_pin_a_limit_under_a_key_keeps_rows_up_to_a_sum_that_wraps` in
`tests/test_scalar_subquery.py` records them, and the table of
[inherited quirks](../oracle/02-inherited-quirks.md) lists this quirk.

Confit refuses an ORDER BY term of another form, such as an expression or a
column of the outer query. It also refuses ORDER BY and LIMIT in a subquery
whose item has an aggregate, and in EXISTS, IN, ANY, SOME and ALL.

Confit also serves `EXISTS` and `NOT EXISTS` over such a subquery, for example
`EXISTS (SELECT * FROM s WHERE s.k = t.k)`. Each SELECT item of this subquery
is `*`, a constant or a column of `s`. The oracle evaluates these items over
the matched rows. So confit refuses an item that can trap there, such as
`s.v + 1`. It also refuses an aggregate item, such as `count(*)`, and LIMIT.

For a request, `EXISTS` is TRUE if a row of `s` passes the WHERE, and
`NOT EXISTS` is TRUE if no row passes it. Confit computes `EXISTS` as
`count(*) > 0` over the same FROM and WHERE, and `NOT EXISTS` as
`count(*) = 0`. The oracle gives the same values and the same traps for these
forms. Pins at the end of `tests/test_scalar_subquery.py` record this. In the
oracle, `NOT EXISTS (...) IS NULL` means `NOT (EXISTS (...) IS NULL)`, as for
any other `NOT`, and confit parses it in the same way.

Confit also serves `x IN (SELECT s.c FROM s WHERE ...)` and `x NOT IN (...)`
over such a subquery, for example `t.k IN (SELECT s.v FROM s WHERE s.j = t.j)`.
The left side `x` is an expression of the outer query. The SELECT item `s.c`
is one column of `s`. The oracle evaluates the item over the rows that pass
the WHERE of the subquery. So confit refuses an item that is not a column of
`s`, such as `s.v + 1` or `count(*)`, because it can trap there. Confit also
refuses these forms:

- more than one item;
- DISTINCT, GROUP BY, ORDER BY or LIMIT in the subquery;
- an `x` that holds a subquery, or that reads a table that the outer query
  joins;
- an `x` and an item where one is a VARCHAR and the other is not. The oracle
  stops with a binder error for this form.

The oracle compares `x` with the item as `=` does, so `-0.0` equals `0.0`,
and a NaN equals a NaN.

For a request, `x IN (...)` is TRUE if the item of a row that passes the
WHERE of the subquery equals `x`. Else it is FALSE if no row passes this
WHERE. Else it is NULL if `x` is NULL or the item of a row that passes this
WHERE is NULL. Else it is FALSE. `x NOT IN (...)` is NOT over this value.

A key of the WHERE is a term that compares a column of `s` with an expression
of the outer query (see below). Confit computes three counts of the rows that
pass the WHERE:

1. The rows whose item equals `x`. For this count, confit adds the test
   `s.c = x` as one more key, after the terms of the WHERE. So `s.c` is one
   more key column of the groups of this count.
2. All the rows.
3. The rows whose item is not NULL.

Then `x IN (...)` is TRUE if count 1 is above 0. Else it is FALSE if count 2
is 0. Else it is NULL if count 3 is less than count 2. Else it is NULL if `x`
is NULL. Else it is FALSE. The pins at the end of
`tests/test_scalar_subquery.py` record that the oracle gives the same values.

The oracle evaluates `x` only if a row of `s` passes the whole WHERE of the
subquery and the item of this row is not NULL. The whole WHERE includes both
bounds of a BETWEEN. If this WHERE has no key, the rows that pass it are the
same for all the requests. Then confit computes count 3 at build. It evaluates
`x` for count 1 only if count 3 is above 0. It tests whether `x` is NULL only
after count 2 is above 0 and count 3 is not less than count 2, so also only if
count 3 is above 0.

If the WHERE has a key, the oracle evaluates `x` for every request of the
batch once one request matches a row of `s`. This is also true if the item of
this row is NULL. So if `x` can trap, whether the oracle traps for a request
depends on the other requests in the same batch. Confit serves only answers
that do not depend on the other requests, so it refuses an `x` that can trap
over a WHERE with a key.

Confit also serves `x op ANY (SELECT s.c FROM s WHERE ...)` and
`x op ALL (...)` over such a subquery, for example
`t.k < ANY (SELECT s.v FROM s WHERE s.j = t.j)`. Here `op` is one of the
comparisons `=`, `<>`, `<`, `<=`, `>` and `>=`, and `SOME` is the same as
`ANY`. The rules for `x` and for the item `s.c` are those of IN, and confit
refuses the same forms.

The oracle answers `x op ALL (...)` as `NOT (x op' ANY (...))`, where `op'` is
the opposite comparison. For example, `x > ALL (...)` is
`NOT (x <= ANY (...))`. The oracle answers `x = ANY (...)` as `x IN (...)`, so
`x <> ALL (...)` is `x NOT IN (...)`. Confit serves these two forms as IN and
NOT IN.

For another comparison, the value of `x op ANY (...)` follows the rules of IN,
with the test `x op c` in place of `x = c`. So it is TRUE for a request if
`x op c` is TRUE for the item `c` of a row that passes the WHERE of the
subquery. Else it is FALSE if no row passes this WHERE. Else it is NULL if
`x` is NULL or the item of a row that passes this WHERE is NULL. Else it is
FALSE.

Confit computes two of the three counts of IN: count 2, of all the rows that
pass the WHERE, and count 3, of those whose item is not NULL. In place of
count 1, a request reads one more table through a LEFT join. Confit computes
this table at build. For each group of rows with equal keys (see below), it
holds the count of the rows of the group, and the least or the greatest item
of the group, or both. Then the test `x op c` is TRUE for a row of the group
as follows:

- `x < c` or `x <= c` is TRUE for a row if and only if it is TRUE for the
  row with the greatest item.
- `x > c` or `x >= c` is TRUE for a row if and only if it is TRUE for the row
  with the least item.
- `x <> c` is TRUE for a row if and only if it is TRUE for the row with the
  least item or for the row with the greatest item.

These rules are true because the oracle can compare each two values of a
type: one value is less than the other, or they are equal. A NaN is above
every other DOUBLE, and `-0.0` equals `0.0`. The pins at the end of
`tests/test_scalar_subquery.py` record that the oracle gives the same values
as these rules.

The least or the greatest item can be one of two values that compare as
equal but have different bits, such as `-0.0` and `0.0`. For `min` and `max`,
confit refuses to build in this case (see the build refusals at the end of
this section). Under ANY and ALL, only a comparison reads the item, and both
values give the same answer. So confit builds in this case.

The oracle casts `x` and each item to the common type of their two types
before it compares them. For two DECIMAL types that need more than 38 digits
together, this type keeps the digits before the point, and it drops digits
after the point. For example, DECIMAL(9,4) and DECIMAL(38,0) give
DECIMAL(38,0), so the oracle rounds an `x` of 1.5 to 2 before it compares. For
these two types, the comparison operator `<` gives DECIMAL(38,4) instead, so
it keeps the digits after the point. Under ANY and ALL, confit casts both
sides to the common type of ANY. If the cast of an item to this type can
raise an error, confit refuses the form. The oracle casts every item, so an
item that is not the least or the greatest can also raise the error.

The oracle evaluates `x` once a row of `s` passes the whole WHERE of the
subquery, also if the item of this row is NULL. Under IN, the item of this
row must also be not NULL. Confit evaluates `x` only if the count of the rows
of the group is above 0. If the WHERE has a key, the oracle evaluates `x` for
every request of the batch once one request matches a row of `s`. This rule is
the same as for IN. So confit refuses an `x` that can trap over a WHERE with a
key.

Each term of the WHERE has one of three forms. A key compares a column of `s`
with an expression of the outer query, its outer expression, by `=` or
`IS NOT DISTINCT FROM`. A term of the second form reads only `s`, such as
`s.v > 5`. A term of the third form compares `s` with the outer query in
another way, such as `s.lo <= t.x`. Confit serves this form by a scan of `s`
for each request (see "The scan" below). The WHERE can also be absent. The
outer expression of the first key in the written order can trap (see below).
If another term can trap, for example the outer expression of a later key or
a term that reads only `s`, confit serves the subquery by the scan. If no
term of the WHERE reads the outer query and a term can trap, confit refuses
the subquery. The oracle runs such a subquery once for each batch, also for a
batch with no requests. Confit also refuses an outer expression that reads a
table that the outer query joins, and a term that reads only `s` but holds a
subquery or calls a SQL function.

The oracle evaluates the subquery once for each distinct outer value. It counts
`-0.0` and `0.0` as one value, because `=` finds them equal. In two measured
batches with one request of each, the oracle used the value of the first request
for both requests. So if an outer expression gives different answers for `-0.0`
and `0.0`, the answer for a request can depend on the other requests of the
batch. An example is `CAST(t.x AS VARCHAR)`, which gives `'-0.0'` for `-0.0`.
The pin `test_pin_the_subquery_runs_once_for_minus_zero_and_zero` in
`tests/test_scalar_subquery.py` records this.

So confit refuses a key or a term of the third form if its part that reads the
outer query can give different answers for `-0.0` and `0.0`. Confit accepts this
part if it reads each DOUBLE value of the outer query only through these
operations: a comparison, IS NULL, `+`, `-`, `*`, the dividend of `/` or `%`,
CASE, `greatest`, `least`, a cast to a type other than VARCHAR, and a math
function of one argument, such as `round`, `abs` or `sqrt`. For `-0.0` and
`0.0`, each of these operations gives two answers that `=` finds equal, or it
stops with an error for both. For example, `sqrt` gives `-0.0` and `0.0`. A
measurement on the oracle on 2026-10-16 gave these answers. The left side `x`
of IN, ANY and ALL is not part of this rule, because the oracle evaluates `x`
for each request.

**The scan.** If the WHERE has a term of the third form, or a term that can
trap other than the outer expression of the first key, confit computes the
value for each request by a scan of `s`. An example is
`(SELECT sum(s.v) FROM s WHERE s.lo <= t.x AND t.x < s.hi)`. The scan reads
the rows of `s` that pass the terms that read only `s`. It reads only the
rows whose keys equal the keys of the request. Then it keeps each row on which
every other term is TRUE, and it computes the aggregate over these rows. The
scan does not use a key that compares by `IS NOT DISTINCT FROM` to find rows.
It tests such a key on each row, as it tests the other terms. At build,
confit refuses `s` if one request can read more than 100 rows. For example,
it refuses `s` if more than 100 of these rows have the same key, or if the
subquery has no key and more than 100 rows pass. The limit of 100 rows
follows the owner's suggestion of 2026-10-15.

The scan serves a subquery that selects one of these aggregates or an
expression over them, and EXISTS and NOT EXISTS:

- `count(*)`, and `count` of a column or of a constant;
- `count_if` of a BOOLEAN column;
- `sum` of an integer or BOOLEAN column, or of a DECIMAL column with at most
  36 digits;
- `min` and `max` of an integer, DECIMAL, VARCHAR, DOUBLE or BOOLEAN column;
- `bool_and` and `bool_or` of a BOOLEAN column;
- `bit_and`, `bit_or` and `bit_xor` of an integer column.

The answer of each of these aggregates does not depend on the order of the
rows, so an ORDER BY inside the call does not change it. A DOUBLE `min` or
`max` is the one exception. Two DOUBLE values can compare as equal but have
different bits: `-0.0` and `0.0`, or two NaN values. If the answer is one of
two such values, the oracle answers the one that it reads first. So confit
refuses the scan at build if a column that a `min` or `max` reads has two
such values in the rows that one request can read.

In the scan, a `sum` cannot overflow. 100 integer values add up to much less
than the range of the HUGEINT answer, and 100 DECIMAL values with at most 36
digits add up to less than 10^38. Confit refuses the scan in these cases:

- a term of the scan can give different answers for the outer values `-0.0`
  and `0.0` (see above), such as `s.f < 1.0 / t.x`, where `1.0 / -0.0` is
  `-inf` and `1.0 / 0.0` is `inf`;
- the subquery is under IN, ANY or ALL, or it has ORDER BY or LIMIT;
- the SELECT item is another aggregate, or `count`, `count_if`, `sum` or
  `bit_xor` over DISTINCT values, or an expression over such an aggregate;
- the SELECT item is an expression with no aggregate.

A term of a scan can trap. The oracle evaluates each term for the pairs of a
distinct outer value and a row of `s` on which the terms before it are TRUE.
It evaluates the terms in written order, but it splits each
`x BETWEEN lo AND hi` into the comparisons `x >= lo` and `x <= hi`, and it
moves `x <= hi` to the end of the WHERE. Inside one term, the oracle
evaluates the right side of an OR only if the left side is not TRUE.
Measurements on the oracle on 2026-10-16 gave these facts.

Confit finds the first term that can trap, in the order of the oracle. The
terms before it cannot trap, so confit evaluates them in any order. From this
term on, the scan evaluates each term in the order of the oracle, for the
same pairs. If only terms that read only `s` come before the first key, the
search does not include the outer expression of this key. Confit evaluates
this expression as for a subquery without a scan (see below).

If the item is an expression, the scan computes the aggregates and then the
expression over them. Over no rows, `count` is 0 and the other aggregates
are NULL. The expression can trap, for example `max(s.v) + 1`. Confit
evaluates the expression for each request that reaches the subquery, also if
the request keeps no row. So `ln(count(*))` raises an error for a request
that keeps no row.

The oracle evaluates the expression once for each distinct outer value of the
requests that reach the subquery. It does so also if these requests keep no
row. Measurements on the oracle on 2026-10-16 gave these facts. The requests
with the same outer value keep the same rows. So confit gives each of them
the same answer or the same error as the oracle.

Confit binds most terms `x BETWEEN lo AND hi` as these two comparisons, so it
can evaluate `x <= hi` at the end. A few such terms do not bind as two
comparisons, for example `5 BETWEEN 0 AND 2`. If the WHERE has such a term
and the search above finds a term that can trap, confit refuses the subquery.

At build, confit keeps the rows of `s` on which each term that reads only `s`
is TRUE. It puts these rows into groups with equal keys and computes the
aggregate of each group. An aggregate over DISTINCT values reads each distinct
value of the group once. The keys are the columns of `s` that the WHERE
compares with the outer query. If no term compares a column of `s` with the
outer query, all the rows that confit keeps make one group. If the item is an
expression, confit computes it over the aggregates of each group.

If the item has no aggregate, confit computes it over each row of each group,
as the oracle does. Without LIMIT and OFFSET, the value of a group is the item
over its one row. With LIMIT or OFFSET, confit sorts the rows of each group in
the order of the ORDER BY terms, as the oracle does. The value of the group is
then the item over the row that the oracle keeps.

A request reads the value of the group that it matches, through a LEFT join. If
a request matches no group, it reads 0 for `count` and NULL for the other
aggregates. If the item is an expression, the request reads the expression over
these values. Confit computes the value of this expression once, at build. If
the item has no aggregate, a request reads NULL when it matches no group. It
also reads NULL when the subquery has no key and no row of `s` passes the WHERE.

The oracle evaluates the WHERE of the subquery in written order, for each
pair of a distinct outer value and a row of `s`. It evaluates the argument of
the aggregate for the matched rows only. It evaluates the item once for each
distinct outer value, over the aggregates of the matched rows. If no row
matches, it evaluates the item over the aggregates of no rows. The oracle
evaluates an item with no aggregate in another way. It evaluates this item
over each matched row, before ORDER BY, LIMIT and OFFSET. So this item can
raise an error over a row that LIMIT does not keep. The pins at the end of
`tests/test_scalar_subquery.py` record these facts. The order of evaluation
matters only where an evaluation can trap. In a served subquery, only these
parts can trap: the outer expression of the first key, the left side `x` of
IN, ANY or ALL over a WHERE without a key, and a term or the item of a scan.
The argument of the aggregate, if it has one, is a column or a constant.
Without a scan, only the running total of a `sum` or an `avg` can overflow.
Confit computes it at build, and it refuses the build if the total can
overflow (the build refusals at the end of this section list this case). An
expression item can also raise an error, for example `max(s.v) + 1` if `s.v`
is a TINYINT column with the value 127. Without a scan, confit evaluates the
expression at build, so a served subquery does not trap in it. Confit
refuses the build if the expression raises an error:

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
moves the comparison `x <= hi` of `x BETWEEN lo AND hi` to the end of the
WHERE, also with the optimizer off. So if such a term comes before the key,
only `x >= lo` decides if the oracle evaluates the outer expression. At build,
confit counts the rows of `s` that pass these terms. For each request, it
evaluates the outer expression only if this count is above 0. The oracle also
runs a subquery in the SELECT list for each row that the outer WHERE passes,
and a subquery in the WHERE for each row that reaches the WHERE. The oracle
does this also for a subquery inside a CASE, whatever the condition of the
CASE is. Confit evaluates the outer expression of the first key, the left
side `x` of IN, ANY or ALL, and the terms and the item of a scan for the same
rows as the oracle. If one of them can trap, confit refuses the subquery in
two cases:

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

The answer of a DOUBLE `sum` or `avg` depends on the order of the additions.
Confit adds the values of a group that are not NULL in table order. Table order
is the order of the rows in the Arrow table that the caller gives for `s`. The
sum starts at +0.0, as the sum of the oracle does, and each addition rounds
once. For `avg`, confit divides this sum by the count of the values, and the
division rounds once, as in the oracle.

In the measurements of the
[decision record](../../../../proposals/closed/double-sum-bound.md),
the oracle adds the values of a table of 5,000 rows in the same order, and the
two answers are equal bit for bit. Over a table of 300,000 rows, the oracle
adds parts of the table in parallel. Then its answer can differ from the answer
of confit in the last bits, and from its own answer in another run. For an
aggregate over DISTINCT values, the oracle adds the values in the order of a
hash table, also over a table of three rows. So the comparison contract
accepts the answer of confit if its distance from the answer of the oracle is
at most a bound B. For the n values of a group, B is 2(n − 1)·2^-52 times the
sum of their absolute values
([claim: float-reduction-bound](../oracle/05-the-comparison-contract.md#floating-point-comparisons)).

The comparison applies the bound only to a whole output column. So confit
serves a DOUBLE `sum` or `avg` only as the whole SELECT item of a subquery that
is a whole output column, for example
`(SELECT avg(s.f) FROM s WHERE s.k = t.k) AS c`. A derived table or a CTE can
pass this column on with no change. If another part of the query reads the
value, confit refuses the build by name. Examples are an expression such as
`(SELECT sum(s.f) FROM s WHERE s.k = t.k) + 1`, a WHERE or a join condition
that reads the column, and a struct or a list output that holds the column.

Confit also refuses the build, with an error that names the reason, in these
cases that depend on the rows of `s`:

- The sum of the absolute values of a group is past the range of the type in
  which the oracle keeps the sum. For `sum`, this is the range of its result
  (HUGEINT, a 128-bit integer, or a DECIMAL with 38 digits). For `avg`, this
  is the range of HUGEINT. Then a partial sum can overflow in some order of
  the additions. For an aggregate over DISTINCT values, the sum adds each
  distinct value once.
- A group of a DOUBLE `sum` or `avg` has a NaN or an infinite value. The bound
  B of the comparison contract applies only to finite values.
- For a DOUBLE `sum` or `avg`, the sum of the absolute values of a group, plus
  the bound B of the group, is past the range of a DOUBLE. Then a partial sum
  can overflow in some order of the additions.
- A DOUBLE `min` or `max` finds two values that compare as equal but have
  different bits, such as `-0.0` and `0.0`, or two NaN values. Then the answer
  of the oracle depends on the order of the rows. This refusal does not apply
  to the least and the greatest item of ANY and ALL. In a scan, confit
  refuses the build if a column that a `min` or `max` reads has two such
  values in the rows that one request can read (see "The scan").
- At build, an expression item raises an error over the aggregates of a
  group, or an item with no aggregate raises an error over a row of a group.
- The subquery of an item with no aggregate keeps more than one row of a
  group. Without LIMIT and OFFSET, it keeps all the rows of the group. Confit
  refuses the build even if no request matches this group. The oracle stops
  with the error `More than one row returned by a subquery used as an
  expression` only for a request that matches the group. If the subquery has
  no key, the oracle stops with this error each time it runs the query, even
  for a batch with no requests.
- LIMIT and OFFSET keep one row of a group, and a row that ties with this row
  has a different item. Two rows of a group tie if they have the same values
  in all the ORDER BY terms. Without ORDER BY, all the rows of a group tie.
  The oracle does not specify the order of two rows that tie, so it can keep
  either of them. Its answer is then not a function of the query and its
  inputs
  ([claim: nondeterminism-axiom](../oracle/03-nondeterminism.md#decision-rule)).
  In this check, two DOUBLE items are different if their bits are different.
  So `-0.0` and `0.0` are different items, although they sort as equal
  values.

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
`scalar-subquery` has such an expression. About one aggregate of a column in
five reads DISTINCT values. The generator replaces about one in five of the
aggregates `min`, `max` and `sum` of a BOOLEAN or integer column with
`bool_and`, `bool_or`, `count_if`, `countif`, `bit_and`, `bit_or` or
`bit_xor`. About one aggregate of a column in eight has an ORDER BY inside
the call, with one or two columns of `s`. About one subquery in five that is an item of the
outer query, or the operand of IS NULL, is `EXISTS` or `NOT EXISTS` instead.
About one in five of the other subqueries that are an item or the operand of
IS NULL is `x IN (...)` or `x NOT IN (...)` instead. About one IN in two is
`x op ANY (...)`, `x op SOME (...)` or `x op ALL (...)` instead, where `op` is
one of the six comparisons. About two of these in five have NOT before them,
as in `NOT x op ANY (...)`. About one in five of the other subqueries that are
an item or the operand of IS NULL has a SELECT item with no aggregate in place
of its aggregate. These are the subqueries that are not EXISTS, IN, ANY, SOME
or ALL. In about two cases in three, this item is a column of `s`. In the other
cases, it is an expression over the columns of `s`. About one in two of the
subqueries with such an item end with ORDER BY, LIMIT or OFFSET. Such a
subquery has up to two ORDER BY terms. Each term is a column of `s` or, in
about one case in four, the item, by its position or by its alias. The LIMIT
is 1 in about two cases in three, and else 0, 2, NULL, ALL or no LIMIT. About
one of these subqueries in four has an OFFSET of 0, 1, 2 or NULL. For a
subquery with such an item and without these clauses, in about two cases in
three, the generator also removes from `s` each row that has the same values
in the key columns of this subquery as an earlier row, so that a request
matches one row at most.
The generator makes up to two key columns in `s`. It fills about half of the rows of a key column with
values of one column of the request table, so that requests match these rows.
The item is a key column, and `x` is its column of the request table, or, in
about one case in ten, this column + 1.

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
