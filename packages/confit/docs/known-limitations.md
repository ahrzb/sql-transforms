# Known limitations — deliberate, named, and loud

This is the user-facing contract of Confit (`DuckDBInferFn`):
what it refuses to serve, **why**, and what you see when you hit a limit.
Its executable twin is [`packages/confit/tests/test_known_limitations.py`](../tests/test_known_limitations.py) —
every limitation below is asserted there, so lifting one breaks a test and
forces this document to change with it.

**The contract.** For any SQL you hand it, the engine does exactly one of:

1. **Serve it bit-for-bit identical to DuckDB** (verified continuously
   against DuckDB's own test corpus: 550 of 678 statements as of stage B), or
2. **Refuse loudly at BUILD time** — `DuckDBInferFn(...)` raises a
   `ValueError` naming the construct. Nothing is ever silently wrong or
   silently dropped at inference time.

There is no third mode. Every limitation here is a *measured decision*
recorded in a pins spec (`packages/confit/docs/specs/`), not an accident.

**Which DuckDB** (decided 2026-08-17, and it has a user-visible cost — see
§5). "Identical to DuckDB" means DuckDB with its query optimizer off:

```sql
PRAGMA disable_optimizer;
```

That is not a smaller DuckDB. The binder is untouched, so types, constant
folding and bind-time errors are the same; execution-level laziness — an
untaken `CASE` arm, `AND`/`OR` short-circuit in a filter — is the same. What
it removes is the 33 plan-rewrite passes.

The reason is that the optimizer-on reading is not a function of the query.
`statistics_propagation` decides from a column's stored null statistic, so
DuckDB answers the *same query over the same rows* differently depending on
the table's insert history — measured: a table built as `[-128, NULL]` and
then having the NULL deleted answers differently from one built as `[-128]`,
with identical contents. Confit compiles once against a *schema* and serves
many batches; it never sees a table, let alone its history. A target you
cannot compute from the query is not a target.

**What "identical" means for ROW ORDER** (TASK-129, 2026-08-19). Values are
bit-for-bit. A *sequence* is only promised where one is defined: on the row
path by the serving contract (output rows follow input rows -- `map` exactly,
`filter` as a subsequence, `many` as per-input-row blocks in input order,
join order within a block being the documented multiset), and on a
static-tables-only result by a total `ORDER BY`. Anywhere else SQL defines no
order and neither do we -- DuckDB itself returns the same unordered
`GROUP BY` in twelve different row orders over twelve connections (measured).
The campaign fuzzer compares per this rule: sequence-strict self-legs on the
row path (a reversed batch must reverse), sortedness-plus-multiset under
`ORDER BY` (ties are free), multiset otherwise.

**The break is CLOSED** (TASK-124, fixed 2026-08-19). Boolean short-circuit
is now decided per CONTEXT, as DuckDB decides it: selection context (the
WHERE root and every `CASE WHEN` condition, projections included) makes
`AND` lazy left-to-right and `OR` its exact dual, and exits to eager value
context at `NOT`/`IS NULL`/comparisons/function arguments and CASE arms. The
model, its measurements and the design are in
`packages/confit/docs/specs/2026-08-19-selection-context-design.md`; the full
measured matrix runs against the live oracle in
`known_divergences/test_short_circuit.py`.

---

## 1. The specialization bargain (inherent to the engine model)

Confit's speed comes from doing ALL general work at build time:
parse once, bind once, compile once, freeze the static tables into the
code. Anything that would require re-doing general work per row is
rejected, permanently by design:

| Limitation | You'll see | Why |
|---|---|---|
| Regex patterns must be constants (`regexp_matches(s, pattern_col)` rejects) | `unsupported: non-constant regex pattern (compiled at prepare in v0)` | Regexes compile at prepare; DuckDB compiles per row. Per-row compilation is the opposite of specialization. |
| Replacement strings / regex options / extract group indexes must be constants | `non-constant regexp_replace replacement` etc. | Same. |
| Static (join) tables must be provided at build time; under the DEFAULT shapes their keys must be unique | `duplicate map key` | Joins are frozen hash maps baked into the function. Duplicate keys mean 1:N join multiplicity, which SERVES under the opt-in `shape='many'` (TASK-59) — along with cross joins, inequality `ON` predicates, and constant `ON` clauses — with multiset parity vs DuckDB (its join output order is a measured hash-join accident; the engine emits probe order outer, build insertion order inner). Under `filter`/`map` the 1:1 contract is unchanged. NULL *values* serve since TASK-55; NULL *keys* never match. |
| Self-joins require `shape='many'` (and the ON form) | `joining the dynamic table to itself` | Under `'many'` the batch itself becomes the build side (assembled per call, the ON as a per-pair residual) — comma/cross and `ON` self-joins serve with multiset parity. `USING`/`NATURAL` self-joins stay a named rejection (follow-up); the default shapes keep the original error. |
| Exactly one row table drives the query | `the specializer takes exactly one row table`, `must be the dynamic table` | The serving contract is rows-in → rows-out for one entity stream. |

## 2. Out of scope for row-serving (by decision, not difficulty)

**The row-shape contract** (TASK-58): `DuckDBInferFn(..., shape=...)`
declares how many output rows each input row may produce, checked at
build time. `"filter"` (the default) is the engine's native 0..1;
`"map"` statically PROVES exactly-one (`out[i] ↔ in[i]`, the strict
serving guarantee) by rejecting anything that can drop a row — a WHERE
clause, an INNER join (key misses drop), a static-tables-only constant
query; `"many"` (0..N) is the multiplicity opt-in (stage B): duplicate-key
joins, cross joins, and inequality/constant `ON` joins build ONLY under
it (one join per query for now, named restriction) — multiplicity can
never sneak into a serving path by default. Comma and `ON` self-joins
serve under `'many'` too; `USING`/`NATURAL` self-joins are a named
follow-up rejection.

The engine serves **row-at-a-time feature transforms**. Whole-relation
constructs are out of scope because their output shape is not
one-row-in/one-row-out:

- Aggregation: `GROUP BY`, `HAVING`, `sum`/`count`/`avg`/... →
  `aggregate function ... (no aggregation in v0)`
- `ORDER BY`, `LIMIT`/`OFFSET`, `DISTINCT` — row-independent transforms
  don't reorder or deduplicate.
- CTEs (`WITH`), `UNION`/`INTERSECT`/`EXCEPT`, subqueries, multiple
  statements.
- Table functions in FROM (`range(...)`) — there is no base table.
- `rowid` pseudo-column — rows have no stable identity in a stream.
- `FULL OUTER JOIN` — emits rows that no input row produced.

The exception is a **static-tables-only query** (nothing dynamic remains):
it is evaluated once at build by DuckDB itself and frozen, so aggregation,
`ORDER BY` and DuckDB dialect beyond sqlparser all serve there. One rule
governs everything under it: the frozen answer may be frozen only when it is
a function of the query text and the static tables. Anything that lets scan
order, thread scheduling or a draw pick among valid answers refuses by name.

**Selection by position refuses** — `LIMIT`, `OFFSET`, `FETCH`, `TOP`,
`USING SAMPLE`, `DISTINCT ON`, `QUALIFY`, and the row-position window
functions (`row_number`, `ntile`, `lead`, `lag`, `first_value`, `last_value`,
`nth_value`) — anywhere in the statement, `ORDER BY` or not (row limits
decided 2026-08-19). Which rows survive is not a function of the query:
measured over a 60k-row static table fed through a tying `GROUP BY`, the same
statement under five DuckDB settings a build machine picks for itself
(default, `threads` 1/2/8, `preserve_insertion_order=false`) answered a
`LIMIT` in a derived table **four ways**, one in a CTE **five**, `DISTINCT ON`
**five**, `QUALIFY` over `row_number()` **five**, `row_number()` over tied
keys **five**, and `USING SAMPLE` differently on **all twelve of twelve fresh
connections**. Freezing whichever answer the build-time run happened to get
would make two builds of the same function disagree with each other. You'll
see: `row limit (LIMIT/OFFSET) on a static-tables-only query`,
`DISTINCT ON on a static-tables-only query`, and their kind.

A **tie-producing `ORDER BY` refuses**, by the same rule: two rows that tie
on the sort keys are left in an order the query does not state, so freezing
whichever sequence this build's run produced would let two builds disagree
(measured: five sequences under those same five settings). Ties are measured
at build, by DuckDB, over the result it has just frozen — so an `ORDER BY`
whose keys separate every row serves exactly as before, and zero-row and
one-row results cannot tie. `NULL` and `NaN` are ordinary tie-capable values,
and the tie is read off the KEYS, so two rows that carry equal values
everywhere refuse too. A key the output does not carry is added to the
query's own projection and measured there; where it cannot be (a `DISTINCT`
would collapse a different tuple), the query refuses rather than guess.
You'll see: `tie-producing ORDER BY on a static-tables-only query`.

A **non-deterministic function refuses**, anywhere in the statement. DuckDB's
own catalogue names them: `duckdb_functions().stability` is `VOLATILE` for a
fresh draw (`random`, `gen_random_uuid`, `uuid`, `nextval`) and
`CONSISTENT_WITHIN_QUERY` for one reading of a clock or a session per query
(`now`, `current_date`, and the bare keywords `current_timestamp`,
`localtimestamp`, `current_time`, `localtime`). Either way the value belongs
to the RUN: measured, eight builds of `ORDER BY random()` over one static
table froze **four different sequences**. A `CONSISTENT` function is a
function of its arguments and serves unchanged. You'll see:
`the non-deterministic function random() on a static-tables-only query`.

Two kinds the catalogue cannot answer for refuse under that same message.
A **MACRO** has no stability at all — `duckdb_functions().stability` is NULL
for all 131 macro rows — so its DEFINITION is read instead, one level: a
macro whose body names a `VOLATILE`/`CONSISTENT_WITHIN_QUERY` function or a
clock keyword refuses under its own name. Measured, that is exactly `ago`,
`current_catalog`, `current_database`, `current_query`, `current_schema`,
`current_schemas`, `pg_conf_load_time`, `pg_postmaster_start_time` and
`pg_sleep` of the 122; `pg_postmaster_start_time()` used to freeze this
build's wall clock into every row. `error` is left out of the names a
definition is matched against — it is `VOLATILE` so DuckDB never folds it,
but it never RETURNS a value either, and including it refused `histogram`
and `json_group_object` for their failure branches.
And **four functions DuckDB's own flag calls `CONSISTENT`** are refused by a
list kept in the code, because no flag in DuckDB answers the question this
path asks. `current_localtime`/`current_localtimestamp` are DuckDB's own
inconsistency: its binder maps the bare words `localtime`/`localtimestamp`
onto them, and ICU registers them with no stability at all — measured, the
value moves between two connections milliseconds apart. `version` and
`current_setting` are a function of the wheel and of the build machine: two
machines, two frozen answers for one query.

A **TABLE function refuses unless it is a generator**, which is the opposite
polarity from every list above and is the polarity the class deserves.
`duckdb_functions().stability` is NULL for every one of them (all 90 catalogue
rows), so the catalogue
cannot sort them — and it does not have to: a table function is in the `FROM`
to produce rows from somewhere the query does not name. The served list is
`range`, `generate_series`, `unnest`, `repeat`, `repeat_row`, whose rows are
a function of their arguments; every other name refuses under its own. That
covers the 37 nullary ones v1.5.5 will run — `pragma_version`,
`pragma_platform` and `pragma_user_agent` freeze the wheel, the OS and the
Python that built it, `pragma_database_size` the free disk,
`duckdb_settings` the core count and the timezone, `duckdb_extensions` and
`pragma_collations` what this machine has installed, `duckdb_memory` a live
reading, two dozen catalogue views the schema they were asked in — and every
one that takes arguments and reads a FILE (`read_csv`, `read_parquet`,
`glob`), a string of SQL (`query`, `query_table`), or a sample. The name is
read from the `FROM` position alone, so a scalar `repeat('a', 3)` or
`range(3)` is never mistaken for the table function sharing its name (a
table function used as a scalar is a binder error). You'll see: `the table
function duckdb_settings() on a static-tables-only query`.

The cost is one corpus statement, named at `MATCH_FLOOR`: `test_all_types()`
refuses with the rest, because it carries a `TIMESTAMP WITH TIME ZONE`
column that renders in the build machine's timezone (measured, four
timezones, four answers).

An **order-sensitive aggregate refuses**, by name. DuckDB classifies this
itself, and defaults to order-DEPENDENT: an aggregate is order-free only
where its source calls `SetOrderDependent(NOT_ORDER_DEPENDENT)`, which in
v1.5.5 is `count`, `count_star`, `min`, `max`, `bool_and`, `bool_or`, `mad`,
`median` and the `quantile` family — those still serve. Everything else
refuses: the ones that pick a row (`first`, `last`, `any_value`, `arbitrary`,
`arg_max`, `arg_min`, `mode`), the ones that build a sequence (`list`,
`array_agg`, `string_agg`, `group_concat`), and the ones that accumulate in
floating point, where association is not a law (`avg`, `stddev`,
`product`, `corr`, and `sum` over a `DOUBLE`). Measured over 200k rows
arriving in a hash order the settings move: `first` four answers, `list` six,
`avg` six, `sum` six, while
`min`/`max`/`count`/`median` gave one. An aggregate's OWN `ORDER BY` is not
read as a fix — DuckDB's optimizer uses one, but only where its keys separate
the rows within each group, and measuring that per group is a probe this
reading does not build. You'll see: `order-sensitive aggregate list on a
static-tables-only query`.

`sum` is the one name read by OVERLOAD rather than by name, because the flag
is not the same at all of them: DuckDB opts the `BOOLEAN`, integer, `HUGEINT`
and `DECIMAL` sums out and leaves the `DOUBLE` one order-dependent. Which one
bound is not in the parse, so DuckDB's BINDER is asked — the statement is
re-projected onto its bare `sum` outputs and `DESCRIBE`d, and the return type
names the overload: `HUGEINT` or `DECIMAL` serves, `DOUBLE` refuses. That
includes `sum` over a `UHUGEINT`, which has no exact overload to bind and so
accumulates in `DOUBLE` like any float. A sum whose overload the reading
cannot get at keeps refusing whole: nested in a larger expression, in a
`HAVING`, in a set operation or a subquery, or spelled as a window function.

`min` and `max` are the other two whose flag is not the whole name's, and
they are answered by the STATEMENT rather than by the call. DuckDB's
`BindMinMax` swaps them for `arg_min`/`arg_max` when the argument carries a
COLLATION and returns before the `SetOrderDependent` call (minmax.cpp:333-372
against :381), so `min(x COLLATE NOCASE)` is order-dependent by DuckDB's own
flag under a name that is not. Measured over a 400k-row static where every
value has a `NOCASE` twin 200k rows away, `min(g COLLATE NOCASE)` answered
two ways across settings and plain `min(g)` answered one. So a collation
ANYWHERE in the statement takes `min` and `max` off the served list — coarse
on purpose, since it over-refuses an integer `min` beside a collated `WHERE`,
and the alternative is tracking which expression an argument came from. A
collation can only enter through the query TEXT: statics are materialized
from an Arrow schema, which carries none, and DuckDB exposes no column
collation in `DESCRIBE` or `duckdb_columns()` anyway. The two-argument
`min(x, n)` misses the same call and is left serving: measured over ties that
are equal and still distinguishable (`0.0` against `-0.0`, `INTERVAL 1 MONTH`
against `30 DAYS`, 200k of each) it answered one way, and a flag with no
measurement behind it does not earn a refusal.

The cost of reading the REST by NAME, deliberate and fail-closed: **64 of
DuckDB's 88 aggregate names refuse**. The rest of them
refuse for the one reason DuckDB does not flag them, however order-free the
arithmetic looks: `bit_and`/`bit_or`/`bit_xor`, `histogram`, the counters
`count_if`/`countif`/`regr_count`/`approx_count_distinct`, `entropy`, and the
compensated accumulators `fsum`/`kahan_sum`/`sumkahan`/`favg`, which exist to
BE order-stable. `sum_no_overflow` IS opted out at both overloads and is
absent from the served list anyway, because no query can name it — binding
one is `sum_no_overflow is for internal use only!`.

A **row-based window frame refuses**. `ROWS BETWEEN ... PRECEDING/FOLLOWING/
CURRENT ROW` counts NEIGHBOURS, so which rows are in the frame is the arrival
order of the current row's peers. Measured with `max` — one of the eleven
names the aggregate rule lets through, so the window it sits in is one that
can serve at all; `sum` refuses in every window spelling and could never have
shown the difference: over a 200k-row static table fed through a tying
`GROUP BY`, whose own arrival order moved twelve ways in fifteen runs across
those five settings, a running `max` over `ROWS BETWEEN UNBOUNDED PRECEDING
AND CURRENT ROW` answered **twelve ways** and over `ROWS BETWEEN 2 PRECEDING
AND 2 FOLLOWING` **eleven**, while the same window spelled `RANGE` or `GROUPS`
answered **one**. `RANGE` and `GROUPS` frames move by peer group and are
functions of the key, so they serve even over tied keys — and so does
`ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING`, which is the whole
partition however it is spelled and answered one way too. The `rank` family
(`rank`, `dense_rank`, `percent_rank`, `cume_dist`) is a function of the key
and serves over ties, one answer on the same data.
You'll see: `a row-based window frame (ROWS PRECEDING/FOLLOWING/CURRENT ROW)
on a static-tables-only query`.

An **`ORDER BY` below the top serves**, and always has: row order on this
path is not part of the contract at all (see the nondeterminism chapter of
the oracle spec — the differential compares static-only results as an
unordered multiset), so an inner sort orders nothing anybody was promised.
Measured, it moved the sequence and left the row SET identical under all
five settings. The consumers that could have turned an inner order into a
frozen VALUE — an order-sensitive aggregate above it, a row-framed window —
now refuse under their own names, which is what leaves this carve-out
nothing to be wrong about.

A **sort key spelled as a name** goes where DuckDB's binder sends it, which
matters when two output columns answer to the same name. A `SELECT` fills its
alias map in select-list order and lets a later entry overwrite an earlier
one, so `ORDER BY c` over `SELECT g AS c, v AS c` sorts by **v** — measured,
its `DESC` comes back `2, 1, 1`. Only aliases are in that map. On a miss
DuckDB makes a SECOND pass, over the select list itself, and it counts far
less of that list than the output NAMES do: an entry is a candidate only
where it is a column reference carrying no alias of its own, and the term
binds only where exactly one entry matches — everything else is left to the
query's own input. The entries it skips are where the two readings would come
apart: `SELECT st.a, st AS z ... ORDER BY a` names its first output `a`
although it is a struct field and not a column reference, so DuckDB sorts by
the INPUT column `a` while the output column called `a` holds something else.
A star is read as the unaliased column references it expands to, under the
names the result gives them. A QUALIFIED entry is left to the input here too,
although DuckDB counts one: `st.a` and `t.a` are the same shape in the parse
and only the binder tells them apart, and a candidate IS an unaliased column
reference, so measuring the key beside the rows measures the same values. It
costs a refusal only where the projection cannot carry the key — under a
`DISTINCT`, or where the bare name is ambiguous across the `FROM`. A SET
OPERATION
is the other binder — it gathers its children's output names and keeps the
FIRST of a repeat — and this reading keeps the two apart by whether the node
has a select list of its own.

An entry keeps its own place in the select list until an **expanding** entry
stands in front of it, and a star is not the only one that expands: a
top-level `unnest(<struct>)` expands to one output column per field, and
DuckDB's parse calls it a FUNCTION rather than a star. Both are counted for
placement — `SELECT unnest(st), a AS k ... ORDER BY k` reads `k` at the
output position the fields pushed it to — while only a star also SOURCES
names, because what an unnest expands to is not a column reference and the
binder's second pass counts only those. The cost is one shape: an alias
sitting behind TWO expanding entries has no output position this reading can
compute, and refuses as
`a sort key whose output position this reading cannot place`.

A **star sort key** is read the way DuckDB reads it. `ORDER BY ALL` and a
bare `ORDER BY *` both take DuckDB's ORDER-BY-ALL path (no `EXCLUDE`, no
`REPLACE`, no `COLUMNS(...)` expression, and the only sort term), so they sort
by every OUTPUT column and a tie under them is a repeated output row. Any
other star — `COLUMNS('v')` in all four spellings, `* EXCLUDE (c)`,
`* REPLACE (...)` — DuckDB expands over the FROM's INPUT columns instead;
those keys are not in the frozen result, so the query refuses rather than
measure the wrong ones. You'll see: `a star sort key this reading cannot
expand on a static-tables-only query`.

A clause that removes no row is **not** a row limit: `LIMIT ALL` and
`OFFSET 0` serve. That is read off the parse, not guessed: `LIMIT ALL` is a
NULL-typed constant, `OFFSET 0` a zero, and the side of the modifier the
query did not spell is the JSON literal `null`. A limit that is any other
node — `LIMIT 1+1`, `LIMIT CAST(2 AS BIGINT)`, `LIMIT (SELECT 2)` — is one
this reading cannot evaluate, and it counts as the real limit it is.

A **join whose pairing is not a condition refuses.** DuckDB's `JoinRefType`
has exactly six values, and three of them pair rows by the two sides' VALUES
— `REGULAR` by its own `ON`, `NATURAL` by the shared column names, `CROSS` by
nothing at all — whichever `join_type` (INNER/LEFT/RIGHT/FULL/SEMI/ANTI) is
written in front. Those serve. The reading holds that list rather than the
list of offenders, so a seventh value a later DuckDB adds refuses until
somebody measures it.

`POSITIONAL JOIN` pairs row *i* of one relation with row *i* of the other
and asks nothing of their values — selection by position wearing a join's
clothes. Measured over a 300k-row static table fed through a tying
`GROUP BY`, one answered **five ways** across those five settings, and so did
a scalar `count(*)` over it — a VALUE, not a sequence, so "row order is not
the contract" does not cover it. `ASOF JOIN` draws one of the right rows
tied on its inequality: 3000 left rows joined to 150000 right rows with 50
tied matches each answered **15 ways** over seven settings, again as a scalar
`sum`, so no `ORDER BY` reaches it and a tie probe over the OUTPUT never sees
it. `DEPENDENT` no query can spell — `LATERAL`, which would be its syntax,
serializes as `CROSS` or `REGULAR` and the planner makes the dependent ref
itself — so refusing it costs nothing. All three are DuckDB-only shapes that
reach this path precisely because the row path's parser cannot read them.
You'll see: `POSITIONAL JOIN on a static-tables-only query`, `ASOF JOIN on a
static-tables-only query`.

**`rowid` refuses.** A static table is materialized by `CREATE TABLE ... AS
SELECT`, so `rowid` is whatever physical position that produced — the same
selection-by-position that refuses when it is spelled `row_number()`, and a
row limit besides when it is used in a `WHERE`. Bare and qualified (`s.rowid`)
both refuse. The cost, the same one the bare clock keywords pay: a static
column actually NAMED `rowid` refuses too, and DuckDB would have bound the
column. You'll see: `the rowid pseudo-column on a static-tables-only query`.

A static-only refusal is only ever pinned on a query that IS one. A query
that reads a dynamic table cannot fold at all, and the error that reaches the
caller is the row path's own — `unsupported: LIMIT/OFFSET`,
`unsupported: FROM-first SELECT` — naming the same clause without claiming a
path the query never took.

All of the above is read off **DuckDB's own parse** of the statement
(`json_serialize_sql`) and its own catalogue (`duckdb_functions()`), asked of
the connection the statement has already run on so that a function DuckDB
autoloaded an extension for at bind time is in the catalogue that answers.
The parse is DuckDB's because this carve-out exists for the DuckDB dialect
another parser cannot read.

Both ways a statement can go unread refuse it **whole**, because a statement
nobody could read is a statement in which nothing was ruled out — no draw, no
order-sensitive aggregate, no frame, no limit. A statement DuckDB runs but
will not serialize (`PIVOT` is one) refuses under its serialization:
`a statement DuckDB would not expose for inspection`, or `ORDER BY in a
statement DuckDB would not expose for inspection` when the tokens named a
clause. A string holding more than one statement refuses under its own count
— only one statement of it is read — as `a statement string holding more than
one statement`. The cost is that a deterministic `PIVOT s ON g USING max(v)`
refuses along with the `USING first(v)` that froze a scan-order pick.

## 3. Type-system boundaries

The engine computes in exactly four types: `i64`, `f64`, UTF-8 string,
bool. Measured consequences:

- **f32 base tables reject** (`engine is f64-only`).
- **A static column is served at its declared arrow type or refused by
  name** — never widened into a neighbouring lane. Corrected 2026-08-15:
  the catalogue used to widen `float32` and the unsigned widths, and both
  diverged silently. `float32` in value AND type (`s.v * 3.0` over `0.1`
  is `0.30000001192092896`/FLOAT on DuckDB; f64 arithmetic here gave
  `0.30000000447034836`/DOUBLE), unsigned in type (`uint64` stays UINT64
  there, became int64 here). The row path had always refused both. The two
  exceptions that stay served are measured equivalent, not convenient:
  `large_string`/`utf8` (DuckDB normalises them to VARCHAR) and the
  decimal tiers (below).
- **DECIMAL static columns serve EXACTLY** (since TASK-91): the payload is
  the scaled integer in an i128 lane from ingest through the join, emitted
  as `decimal128(p,s)`. `2^53+1` comes back as itself, and so does
  `2^63+1` — an ordinary fit-time `sum(BIGINT)` produces that, which is
  why the lane is i128 and not i64. `decimal32`/`decimal64` inputs
  normalise to `decimal128(p,s)` output because DuckDB exports every tier
  as 128-bit arrow. What still refuses, by name: EXPRESSIONS over a
  decimal — arithmetic, `CAST` to anything but `DOUBLE`, and
  `COALESCE`/`CASE`/`greatest` unifying it with a non-identical type (m-8
  lattice phase 5; each of these used to serve a silently wrong double).
  Comparisons, joins, `CAST(d AS DOUBLE)` and `SELECT *` all serve.
  `decimal256` statics refuse (DuckDB itself refuses them at arrow
  register, at any precision), and decimal ROW columns stay opaque.
- **Structs of scalars SERVE** (since TASK-56): struct row columns are
  flattened to scalar lanes at build time — field access (`a.i`, deep
  `t.t.t.t` paths) and struct-star (`a.*` incl. EXCLUDE/REPLACE) are
  bit-identical to DuckDB. What still rejects, by name: the struct as a
  WHOLE value (`SELECT a` — non-scalar output), bracket field access
  (`a['i']` — DuckDB names such outputs by full expression text, not
  modeled), and struct fields whose own types are non-scalar.
- **Lists reject** (`row column 'x' has a non-scalar type`) — list types
  are out of the row-schema vocabulary and stay opaque: unreferenced
  (star expansion included) they cost nothing to declare; since TASK-56
  an unreferenced timestamp/list field no longer blocks a scalar-only
  query, and `EXCLUDE`/name filters/`REPLACE` can remove one from a
  star. Referenced, they refuse by name. Lists also still gate
  `regexp_extract_all` / `regexp_split_to_array` / the STRUCT form of
  `regexp_extract` (`list-valued — non-scalar in v0`).
- **DECIMAL literals are f64** — a documented divergence: DuckDB types
  `1.5` as `DECIMAL(2,1)` and does decimal arithmetic; we map to f64.
  Values agree on every corpus case; exact-decimal accumulation semantics
  are not reproduced. One visible consequence: `CAST(-2.5 AS BIGINT)` on
  the bare literal is a DECIMAL cast in DuckDB (half away from zero, `-3`)
  and a DOUBLE cast for us (half to even, `-2`). Casting a DOUBLE *column*
  agrees exactly — measure DOUBLE cast behaviour with a DOUBLE column or an
  explicit `::DOUBLE`, never with a literal, or you will pin the wrong
  rounding mode (TASK-70 did).
- Integer widths: the engine TYPES in DuckDB's lattice (TINYINT..BIGINT —
  literals are INTEGER by magnitude, `::SMALLINT` is real, `ascii` returns
  INTEGER, `infer_arrow` emits int8/int16/int32 from the type) but
  COMPUTES in two machine widths, i64 and f64. The width is observable
  exactly where DuckDB's is: the Arrow schema (shipped, TASK-79/m-8
  phase 2, catalogue pinned in `test_integer_widths.py`) and the overflow
  trap threshold — the trap half is m-8 phase 3, so until it lands a
  narrow lane that overflows serves the i64 value on the row path and
  refuses by name at the `infer_arrow` boundary; every input this refuses
  is one DuckDB itself errors on. HUGEINT and the unsigned family are not
  served at all — they refuse by name rather than collapse to i64 (see the
  static-column entry above); serving them is the i128 lane, whose
  cranelift dependency was verified GO on 2026-08-15 (TASK-100).

## 4. Semantics descoped after measurement

Each of these was measured against DuckDB 1.5.5 first (pins in
`packages/confit/docs/specs/`), and rejected because serving it would risk a
wrong answer or require semantics we can't reproduce exactly:

| Construct | Why it's descoped |
|---|---|
| `^` operator | It IS pow in DuckDB, but sqlparser's precedence differs from DuckDB's (`2*x^y` would parse as `(2*x)^y`). Mapping it computes the wrong tree silently. Use `pow()`. |
| prefix `~`, `#`, `NOT GLOB` | Same class: precedence/parse divergences that would silently mis-associate. `xor()` covers bit-xor; `NOT (x GLOB p)` works. |
| Regex reject list: `\B`, `\Q…\E`, `(?<name>…)`, duplicate group names, bounds > 1000, stacked quantifiers (`a*+`), `\u` escapes, negated Perl classes inside `[...]` | The RE2↔rust-regex differential battery (98 entries) proved these are the constructs where the engines disagree or DuckDB itself is broken (`\B` crashes DuckDB at runtime on non-ASCII). Everything else is byte-identical. |
| Fuzzer-found regex rejects (TASK-54): `\1`–`\9` backrefs outside classes, the full stacked-quantifier grammar (`{2}*`, `?*`, `a???` — one lazy `?` is the only legal follower), nested repetition products > 1000, whitespace inside `{m, n}` bounds, class set-op lookalikes (`--`/`&&`/`~~`), non-POSIX `[` inside classes, Perl-class range endpoints (`[a-\d]`), capturing `(x){0}`, anchor-only multi-anchor patterns (DuckDB is SELF-inconsistent on these — its row path disagrees with its own constant fold), `$` anchors in non-final position (`'$hello'` — DuckDB's row path literal-optimizes the leading `$`+literal into a PREFIX match, matching "hello world", while its own constant fold matches normally; found by the standing fuzzer on seed 20260728), and counted repetitions over RE2's PROGRAM-SIZE budget (`(\p{L}){1,500}` is "pattern too large" in DuckDB while rust-regex serves it — rejected via a one-sided weight estimate that always fires before DuckDB's real budget; same seed, pins `pins-waveB/fuzzer-20260728.json`) | The standing differential fuzzer (`packages/confit/tests/test_duckdb_regexp_fuzz.py`, in the normal gate) found these 12 classes in its first 3k-case deep run — each one a silent-wrong-answer risk in rust-regex — then re-swept to ZERO divergences over 40k cases across 8 seeds. Pins: `pins-waveB/fuzzer-task54.json`. |
| `SIMILAR TO ... ESCAPE` | Not implemented in DuckDB itself. |
| `* EXCLUDE (t.key)` on a USING join | DuckDB UNMERGES the coalesced column (it reappears at the right table's position) — measured, not modeled. Unqualified EXCLUDE works. |
| `BETWEEN`/`IN` mixing non-numeric string literals with numbers | DuckDB converts at EXECUTION time (an empty input succeeds!); a bind-time conversion was measured to be over-eager. Numeric literals convert fine. |
| `COLUMNS(...)` inside expressions, lambda/list forms | Only bare `COLUMNS('re')` / `COLUMNS(*)` as select items are served. |
| Pad/repeat counts past the 1 GiB string-builder budget (TASK-88) | A LITERAL count that can exceed the budget refuses at build; a DATA-DRIVEN count (column, `CAST(k AS INTEGER)`) keeps the runtime cap — the engine traps at 1 GiB where DuckDB's own behaviour is spelling-dependent (a multi-GB string or its own builder error). No gigabyte allocations in a serving engine, by decision. |
| `CASE` where every branch is NULL, `COALESCE`/`least`/`greatest` of only NULLs | DuckDB binds the all-NULL FAMILY forms (SQLNULL → INTEGER); the engine still refuses them. A bare `NULL` select item, `nullif(NULL, x)`, and `NULL <op> NULL` all SERVE with DuckDB's types since m-8 phase 2 (bare NULL is int32, its INTEGER). |
| Bare `NULL` as `repeat`'s string (TASK-86, BLOB face) | DuckDB picks the **BLOB** overload — a type this engine doesn't have until the m-8 Blob phase — so adopting VARCHAR answered with a different schema. Spell it `CAST(NULL AS VARCHAR)`, which both engines type identically. Adopters that agree with DuckDB — `upper(NULL)`, `coalesce(NULL, x)`, `nullif(x, NULL)`, `nullif(NULL, x)` (int32, m-8 phase 2), a NULL `repeat` count — keep serving, pinned schema-equal. |

## 5. Deliberate contract choices (behavior differs from raw DuckDB surface)

These are served, but with a consciously chosen surface — know them:

- **Duplicate output column names are renamed**, using DuckDB's own
  boundary-rename algorithm (`id, id, id_1` → `id, id_1, id_1_1`;
  case-insensitive collision check). Raw DuckDB keeps duplicates at the
  top level, but a dict cannot — and this rename is
  bit-identical to what DuckDB itself does at every subquery/CTE/CTAS
  boundary and in `.df()`. Verified against `.df()` in tests.
- **Error TEXTS are approximate where noted.** Runtime traps
  (overflow, shifts, substring range) reproduce DuckDB's message bodies
  verbatim; some bind-time rejections (star-filter zero-match, regex
  compile errors) use our own wording with the same error class. The
  corpus only ever compares successful results, so texts never affect
  parity.
- **Two known oracle divergences** (excluded from the corpus by name in
  `packages/confit/tests/test_corpus_replay.py::_KNOWN_DIVERGENT_SOURCES`): DuckDB
  behaviors that depend on column STATISTICS (e.g. ILIKE's NUL handling
  selects a different kernel depending on *sibling rows*). A row-at-a-time
  engine cannot reproduce statistics-dependent semantics even in
  principle; the engine is NUL-transparent (the ASCII-kernel behavior).
- **A trapping subexpression DuckDB's OPTIMIZER deletes, we still
  evaluate** — the standing cost of running the oracle with the optimizer
  off (see "Which DuckDB" above). This is the one place where a query you
  can run in your own DuckDB session may raise here:

  ```sql
  SELECT (i + 1) > 5 FROM t              -- i INTEGER = 2147483647
  -- your DuckDB (optimizer on): true   -- it rewrites this to i > 4
  -- confit:                     Out of Range Error, overflow in INT32
  ```

  The shape is always the same: a subexpression that would trap, in a
  position where a plan rewrite removes it before it ever runs. The passes
  measured doing this are `expression_rewriter` (constant shifting, folding
  a trapping constant, dead-range elimination) and
  `statistics_propagation` (proving `IS NOT NULL` from a column's null
  statistic, and pruning a filter from a value range). A 4000-seed
  differential campaign puts it at 8 seeds in 28 findings; all eight are
  labelled `DIVERGE_OPT` by the campaign and enumerated in
  `packages/confit/docs/2026-08-17-fuzz-triage.md`.

  The trade is deliberate: matching the optimizer means matching an
  undocumented moving target that is not a function of the query, and in
  the other direction it would mean *serving* where your DuckDB raises.
  This way the divergence is always a loud trap or refusal, never a
  different served value. If it bites you, `PRAGMA disable_optimizer` in
  your DuckDB session reproduces exactly what confit does.
- **`%`-by-zero NaN bit pattern is platform-libm** — pinned as
  engine==oracle bit agreement per platform, not a constant.
- **Schema qualifiers are registry-noise** (TASK-55): the engine's table
  registry is schema-less, so `s1.t1` (and 3-part `s1.t1.col` refs)
  resolve when the table part matches a registered bare name. DuckDB's
  schema-existence errors (`schema "x" does not exist`) are not
  reproduced — a schema-less registry cannot know which schemas would
  exist. Ambiguous matches still error. With struct paths (TASK-56) the
  same rule extends to n-part references: resolution is
  longest-qualifier-first with backtracking (measured), and any first
  part is accepted as a schema when the second matches the table — so
  `w.w.w` on a table `w` with struct column `w` binds the LONGER
  schema-ish parse (a whole-struct rejection) where schema-aware DuckDB
  would fall through to `column.field`. The divergence is always a loud
  build-time rejection, never a different served value.

## 6. How to read a rejection

Every rejection is a `ValueError` at `DuckDBInferFn(...)` construction
whose message starts with a classification:

- `unsupported: ...` — real SQL, deliberately not served (this document).
- `parse error: ...` — the dialect surface ends here.
- `bind error: ...` — the query is wrong against YOUR schema (typo,
  type mismatch), not a limitation.

If a message you hit isn't in this document or the tests, that's a bug in
our bookkeeping — file it.

## 7. How this document stays honest

Four mechanisms, all in the normal test gate:

1. **The corpus replay** (678 statements mined from DuckDB's test suite):
   every statement must match bit-for-bit, reject cleanly, or be a named
   divergence — a wrong answer anywhere fails the gate.
2. **The executable twin** (`packages/confit/tests/test_known_limitations.py`): every
   limitation in this document is asserted; lifting one breaks a test.
3. **The standing differential fuzzer** (`packages/confit/tests/test_duckdb_regexp_fuzz.py`):
   randomized DuckDB-vs-engine sweeps of the regex surface on every run
   (seed/size overridable for deep runs) — new divergences fail with the
   reproducing seed and SQL, and their fix lands as a reject-list entry
   plus a row in this document.
4. **The campaign fuzzer reads DuckDB TWICE** (`packages/confit/fuzz/`),
   once with the optimizer off and once on, so a finding says which kind it
   is instead of needing a human to reason about it: a disagreement with the
   optimizer-off reading is a bug, a disagreement only with the optimizer-on
   one is the §5 cost above, and an agreement with optimizer-on *against*
   the oracle means the engine is reproducing a pass it should not — that
   last category is reported as a bug and is currently empty.
