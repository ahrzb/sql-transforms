# NATURAL and USING join keys over non-scalar shared columns (TASK-133)

Origin: the "Discovered, needs its own ticket" section of
docs/superpowers/specs/2026-08-25-task-127-remainders-design.md. Direction
decided by AmirHossein 2026-08-25: SUPPORT these joins, do not refuse them.

Everything below was measured on 2026-08-25 against DuckDB 1.5.5 with
`PRAGMA disable_optimizer` (the repo oracle, conftest.py:62-71), and against
confit built from this worktree (`uv run maturin develop --release`). Every
DuckDB cell in the matrix was additionally run as `PREPARE p AS <sql>` on its
own connection, as a plain execute, and as a zero-row leg (`... WHERE 1=0`);
all three agreed in every cell, so no claim below depends on which phase an
answer came from. Every one of our refusals is at CONSTRUCTION
(`DuckDBInferFn(...)`), never at `infer_rows`.

The source citations are a read-only audit of the v1.5.5 checkout at
repos/github.com/duckdb/duckdb.

## Goal

The four TASK-133 acceptance criteria, by name:

- The measure-first criterion: join-key semantics for struct and opaque
  shared columns measured against DuckDB, with a recorded matrix, before any
  key-encoding design. That is this document's "The matrix" section.
- The NATURAL-keys-on-all criterion: NATURAL JOIN keys on ALL shared columns.
- The USING-struct-key criterion: `USING (w)` with a struct column serves, and
  the false `column "w" does not exist on right side of join!` is gone.
- The refuse-by-name backstop criterion: a shared non-scalar column the key
  encoding still cannot carry refuses BY NAME at build, and never silently
  drops out of the key set.

## Non-goals

- Serving a struct as a VALUE. `SELECT w`, `SELECT *` over a struct-carrying
  table, and `SELECT s.*` all keep refusing by name (TASK-125's rule). A
  struct becomes a KEY here; it does not become servable.
- The `SELECT *` duplicate-name question. DuckDB emits `['id','w','id','z']`
  for `USING (w)` -- the non-key `id` twice, not deduped. We rename the second
  occurrence (measured on the all-scalar control: DuckDB `['id','v','v','z']`,
  ours `id/v/v_1/z`). That is TASK-130, unchanged here.
- Opaque (non-vocabulary) shared columns as keys: TIMESTAMP, DATE, TIME,
  FLOAT32, UINT64, LIST, DECIMAL-on-the-row-side. This spec REFUSES them by
  name. Serving them needs a key-only lane kind and is its own ticket -- see
  "Open questions", which flags that this reading contradicts the
  NATURAL-keys-on-all criterion's literal text.
- Self-joins. `frontend.rs:764-768` already refuses `self-join USING/NATURAL
  (stage-B follow-up; use ON)`; that stays.
- The scalar type-mismatch divergence found in passing: DuckDB serves a
  NATURAL join of a row BIGINT `v` against a static VARCHAR `v` (it casts,
  `'3' = 3` is TRUE); we refuse `cannot join i64 with str (ON 'v')`. SEV-4,
  pre-existing, unrelated to non-scalar keys.
- `USING (id)` then `SELECT w.mean`, where `w` is a merged struct key: DuckDB
  resolves it to the LEFT occurrence. Ours refuses `ambiguous column 'w'`.
  That flip is listed below because this fix is its precondition, but the
  unqualified-resolution machinery it needs is TASK-127's D2.

## What DuckDB actually does, source-pinned (v1.5.5)

TASK-127 already pinned that `bind_joinref.cpp` has no `LogicalType`
inspection: NATURAL intersects column NAME SETS (185-208), USING takes the
list verbatim (240-247), and STRUCT falls through `default: break` in
`BoundComparisonExpression::TryBindComparison`. This section adds the four
things that audit did not cover.

### The comparison type is COMPARE_EQUAL, identical to an explicit ON

`bind_joinref.cpp:287-293` is the single construction site NATURAL and USING
share (the NATURAL arm only populates `extra_using_columns`, then falls into
the common block at 256):

```cpp
const auto type = (ref.ref_type == JoinRefType::ASOF && i == extra_using_columns.size() - 1)
                      ? ExpressionType::COMPARE_GREATERTHANOREQUALTO
                      : ExpressionType::COMPARE_EQUAL;
extra_conditions.push_back(
    AddCondition(context, left_binder, right_binder, left_binding, right_binding, using_column, type));
```

An explicit `ON l.w = r.w` produces the same node: `expression_type.cpp:333-335`
maps `"="` to `COMPARE_EQUAL`, and `bind_comparison_expression.cpp:190-191`
preserves it verbatim. `CreateJoinCondition` (`plan_joinref.cpp:143-166`)
copies the type through, mutating it only by `FlipComparisonExpression` on a
side swap, which is the identity for `COMPARE_EQUAL`. The two rewrites to
`COMPARE_NOT_DISTINCT_FROM` in the tree are `deliminator.cpp:341-353` (an
OPTIMIZER pass, dead under `disable_optimizer`) and
`flatten_dependent_join.cpp` (correlated subqueries only). So under the
oracle, USING / NATURAL / explicit `ON =` are the same comparison.

Confirmed behaviourally: a scalar BIGINT `d` that is NULL on both sides
misses under NATURAL, `USING (d)` and `ON t.d = s.d`, and HITS only under
`ON t.d IS NOT DISTINCT FROM s.d`.

### `=` on a STRUCT is field-wise IS NOT DISTINCT FROM under a top-level NULL guard

`NestedComparisonExecutor` (`comparison_operators.cpp:152-214`) filters
TOP-LEVEL NULLs into `result_validity` via `ComparesNotNull`, saves that mask,
runs the nested select, and then RESTORES validity for every row that was
valid before the nested walk (185-213). Inner NULL marks therefore never
reach the result. The nested walk itself is `NestedSelector::Select<Equals>`
-> `VectorOperations::NestedEquals` ->
`TemplatedDistinctSelectOperation<DistinctFrom>`
(`is_distinct_from.cpp:1274-1279`), and the both-NULL decision is
`DistinctFrom::Operation` (`comparison_operators.hpp:83-91`) returning
`left_null != right_null`.

The join kernel agrees exactly. `RowMatcher::GetStructMatchFunction`
(`row_matcher.cpp:379-382`):

```cpp
case ExpressionType::COMPARE_EQUAL:
    result.function = StructMatchEquality<NO_MATCH_SEL, Equals>;
    child_predicate = ExpressionType::COMPARE_NOT_DISTINCT_FROM;
    break;
```

Top-level `=`, every child matched with `NOT_DISTINCT_FROM`. That one line is
the whole semantics, and it is what the encoding below mirrors.

DuckDB's own expected-output file `test/sql/types/struct/struct_null_members.test`
states the four cases, and our live probe reproduces them:

| lhs | rhs | `=` | `IS NOT DISTINCT FROM` |
|---|---|---|---|
| `{'x':NULL,'y':NULL}` | `{'x':NULL,'y':NULL}` | TRUE | TRUE |
| `{'x':1,'y':NULL}` | `{'x':1,'y':NULL}` | TRUE | TRUE |
| `{'x':1,'y':0}` | `{'x':1,'y':NULL}` | FALSE | FALSE |
| `NULL` | `{'x':NULL,'y':NULL}` | NULL | FALSE |
| `NULL` | `NULL` | NULL | TRUE |

The recursion carries the NODE's own validity, not just the leaves'. Measured
directly, and this is the cell that decides the encoding:

```
w = {inner: NULL}          vs  w = {inner: {val: NULL}}   ->  FALSE  (miss)
w = {inner: {val: NULL}}   vs  w = {inner: {val: NULL}}   ->  TRUE   (hit)
w = {inner: NULL}          vs  w = {inner: NULL}          ->  TRUE   (hit)
w = {i: {j: NULL}}         vs  w = {i: {j: {v: NULL}}}    ->  FALSE  (miss)
```

Both sides of the first pair flatten to exactly one leaf value, `val = NULL`.
DuckDB tells them apart. A leaf-lane-only encoding cannot.

Field pairing is BY NAME, not by position, and a name-set mismatch is FALSE
rather than an error:

```
{'a':1.0,'b':2.0} = {'b':2.0,'a':1.0}   -> True    (reorder is a lossless cast)
{'a':1.0,'b':2.0} = {'x':1.0,'y':2.0}   -> False
{'a':1.0}         = {'a':1.0,'b':2.0}   -> False
{'x':1.0,'y':2.0}::STRUCT(a DOUBLE, b DOUBLE)
    -> Binder Error: STRUCT to STRUCT cast must have at least one matching member
```

### Float edges match our existing canonicalisation exactly

`EqualsFloat` (`comparison_operators.cpp:17-33`) makes NaN = NaN TRUE and
falls to IEEE `==` for signed zeros, so `-0.0 = 0.0` is TRUE. The hash side
normalises both before hashing -- `FloatingPointEqualityTransform`
(`hash.cpp:32-58`) flushes `-0.0` to `+0.0` and canonicalises every NaN
payload to one quiet NaN -- so `x = y` implies `hash(x) = hash(y)`. Inside a
struct, `StructLoopHash` (`vector_hash.cpp:90-115`) recurses through the same
`Hash(double)` specialisation, and the child predicate `NOT_DISTINCT_FROM`
reaches `NotEquals`, which is defined as `!Equals::Operation`, so it picks up
the float specialisation too. NaN-in-struct matches NaN-in-struct; -0.0-in-
struct matches 0.0-in-struct. Both measured.

This is exactly `exec/mod.rs:309-331` (`duck_fcmp`, `canon_f64_bits`), which
already collapses all NaNs to one key class and `-0.0` to `+0.0`. Our scalar
DOUBLE key needs no change: all five scalar-DOUBLE control cells MATCH today.

Hash-join gating, for the record: nested keys always land on
`PhysicalHashJoin` -- `plan_comparison_join.cpp:37-63` has no type inspection,
and it is `PhysicalNestedLoopJoin::IsSupported` (`physical_nested_loop_join.cpp:124-130`)
that excludes STRUCT/LIST/ARRAY. NULL keys are dropped on both sides for
`COMPARE_EQUAL` because `null_values_are_equal` is false
(`join_hashtable.cpp:62-63`), and that filter reads the key vector's TOP-LEVEL
validity, so a struct with NULL fields survives it. Consistent with the
scalar kernel.

### A LEFT JOIN's merged USING column is bare `l.w`, not COALESCE

`bind_columnref_expression.cpp:62-75` emits `OPERATOR_COALESCE` only when
`using_binding->primary_binding` is unset; `SELECT *` expansion applies the
identical rule at `bind_context.cpp:515-534`. `SetPrimaryBinding`
(`bind_joinref.cpp:83-100`, called at 297) sets it to the left for
INNER/LEFT/SEMI/ANTI, to the right for the RIGHT family, and leaves it unset
only for FULL OUTER. So COALESCE appears for FULL OUTER alone.

Measured, row `(id=5, w={mean:1.0})` against static `(id=5, w={mean:2.0}, z=7)`:

```
LEFT JOIN s USING (w)  SELECT *  -> names ['id','w','id','z'] rows [(5, {'mean':1.0}, None, None)]
LEFT JOIN s USING (w)  SELECT w  -> [({'mean': 1.0},)]        the LEFT value on a miss
NATURAL LEFT JOIN s    SELECT *  -> names ['id','w','z']      rows [(5, {'mean':1.0}, None)]
```

Our engine's scalar merge already reproduces this in every cell (see matrix
row i-scalar), so nothing here needs new machinery -- only the struct head has
to join the rule.

## The matrix

Fixtures in the repo's test style: row `__THIS__` and static `s` share `id
BIGINT` and a column under test; `s` also carries `z BIGINT`. Probes are the
THREE forms side by side, because the fix must reproduce whichever semantics
each form has:

```
N   SELECT z AS o FROM __THIS__ NATURAL JOIN s
U   SELECT z AS o FROM __THIS__ JOIN s USING (id, <col>)
U1  SELECT z AS o FROM __THIS__ JOIN s USING (<col>)          -- <col> alone
O   SELECT z AS o FROM __THIS__ JOIN s ON __THIS__.id = s.id AND __THIS__.<col> = s.<col>
```

One row in, so "rows out" is 0 or 1. `[]` = no row, `[7]` = one row. Verdicts:
MATCH; SEV-2 wrong value; SEV-3 serve-where-DuckDB-refuses; SEV-4
refuse-where-DuckDB-serves (the acceptable cost); MSG both refuse, our reason
is wrong.

Our refusals, abbreviated:

```
R1  bind error: column "<c>" does not exist on right side of join!     (USING arm)
R2  unsupported: static table 's' column '<c>' is a struct -- project its fields instead
R3  unsupported: static table 's' column '<c>' has type <arrow>, which this engine does not serve
R4  unsupported: row column '<c>' has a non-scalar type
R5  unsupported: column '<c>' has a non-scalar type                    (star expansion)
```

### a. struct `w STRUCT(mean DOUBLE)`

| cell (row w / static w) | DuckDB N | ours N | N verdict | DuckDB U/U1 | ours U/U1 | DuckDB O | ours O |
|---|---|---|---|---|---|---|---|
| `{mean:1.0}` / `{mean:1.0}` | `[7]` | `[7]` | MATCH | `[7]`/`[7]` | R1 | `[7]` | R2 |
| `{mean:1.0}` / `{mean:2.0}` | `[]` | `[7]` | **SEV-2** | `[]`/`[]` | R1 | `[]` | R2 |
| `{mean:NULL}` / `{mean:2.0}` | `[]` | `[7]` | **SEV-2** | `[]`/`[]` | R1 | `[]` | R2 |
| `{mean:NULL}` / `{mean:NULL}` | `[7]` | `[7]` | MATCH (by luck) | `[7]`/`[7]` | R1 | `[7]` | R2 |
| `NULL` / `{mean:2.0}` | `[]` | `[7]` | **SEV-2** | `[]`/`[]` | R1 | `[]` | R2 |
| `NULL` / `NULL` | `[]` | `[7]` | **SEV-2** | `[]`/`[]` | R1 | `[]` | R2 |
| `{mean:NULL}` / `NULL` | `[]` | `[7]` | **SEV-2** | `[]`/`[]` | R1 | `[]` | R2 |

Every U and U1 cell is SEV-4; every O cell is SEV-4.

### b. nested struct `w STRUCT(inner STRUCT(val DOUBLE))`

| cell | DuckDB N | ours N | verdict |
|---|---|---|---|
| `{inner:{val:9.0}}` / same | `[7]` | `[7]` | MATCH |
| `{inner:{val:9.0}}` / `{inner:{val:8.0}}` | `[]` | `[7]` | **SEV-2** |
| `{inner:{val:NULL}}` / same | `[7]` | `[7]` | MATCH (by luck) |
| `{inner:NULL}` / `{inner:NULL}` | `[7]` | `[7]` | MATCH (by luck) |
| `{inner:NULL}` / `{inner:{val:9.0}}` | `[]` | `[7]` | **SEV-2** |
| `{inner:NULL}` / `{inner:{val:NULL}}` | `[]` | -- | THE DISCRIMINATOR |
| `{i:{j:NULL}}` / `{i:{j:{v:NULL}}}` (3 deep) | `[]` | -- | THE DISCRIMINATOR |

U/U1/O are R1/R1/R2 in every row, all SEV-4.

The two DISCRIMINATOR rows are why the encoding is what it is: both sides
flatten to the identical leaf tuple, and DuckDB still says miss.

### c. float edges inside the struct field

| cell | DuckDB N | ours N | verdict |
|---|---|---|---|
| `{mean:NaN}` / `{mean:NaN}` | `[7]` | `[7]` | MATCH |
| `{mean:-0.0}` / `{mean:0.0}` | `[7]` | `[7]` | MATCH |
| `{mean:inf}` / `{mean:inf}` | `[7]` | `[7]` | MATCH |
| `{a:NaN,b:1}` / `{a:NULL,b:1}` | `[]` | -- | NaN is not NULL |

U/U1/O all SEV-4 (R1/R1/R2).

### d. bare `DOUBLE` shared column -- the SCALAR control for (c)

| cell | DuckDB | ours | verdict |
|---|---|---|---|
| NaN / NaN | `[7]` in N,U,U1,O | same in all four | MATCH x4 |
| -0.0 / 0.0 | `[7]` in N,U,U1,O | same in all four | MATCH x4 |
| 1.0 / 1.0 | `[7]` in N,U,U1,O | same in all four | MATCH x4 |
| NULL / 1.0 | `[]` in N,U,U1,O | same in all four | MATCH x4 |
| NULL / NULL | `[]` in N,U,U1,O | same in all four | MATCH x4 |

**No pre-existing scalar divergence.** `canon_f64_bits` already reproduces
DuckDB's join equality on every float edge. Cross-checked against the oracle's
own multi-row form: `USING (d)` over `{NaN, -0.0, 0.0, NULL}` on both sides
returns the NaN pair and BOTH zero pairs in each direction, and drops NULL --
one zero key class, one NaN key class, NULL excluded. Identical under
`ON t.d = s.d`.

### e. `TIMESTAMP` shared column

| cell | DuckDB N | ours N | verdict | U/U1 | O |
|---|---|---|---|---|---|
| equal | `[7]` | `[7]` | MATCH | `[7]` / R1, SEV-4 | `[7]` / R3, SEV-4 |
| unequal | `[]` | `[7]` | **SEV-2** | `[]` / R1, SEV-4 | `[]` / R3, SEV-4 |
| NULL one side | `[]` | `[7]` | **SEV-2** | `[]` / R1, SEV-4 | `[]` / R3, SEV-4 |
| NULL both sides | `[]` | `[7]` | **SEV-2** | `[]` / R1, SEV-4 | `[]` / R3, SEV-4 |

### f. the rest of our opaque classification

Our vocabulary is `schema.rs:131-162`. Policy `Row` serves exactly
bool / int8 / int16 / int32 / int64 / double / string. Policy `Static` adds
large_string / utf8 / large_utf8 and decimal128 (which rides the f64 lane).
Everything else is `RowField::Opaque(arrow_spelling)` and gets NO lane on
either side. Representatives measured:

| type | DuckDB N equal / unequal | ours N | verdict | U/U1 | O |
|---|---|---|---|---|---|
| `date32[day]` | `[7]` / `[]` | `[7]` / `[7]` | MATCH / **SEV-2** | R1, SEV-4 | R3, SEV-4 |
| `date32[day]` NULL both | `[]` | `[7]` | **SEV-2** | R1 | R3 |
| `time64[us]` | `[7]` / -- | `[7]` | MATCH | R1, SEV-4 | R3, SEV-4 |
| `float` (float32) | `[7]` / `[]` | `[7]` / `[7]` | MATCH / **SEV-2** | R1, SEV-4 | R3, SEV-4 |
| `uint64` | `[7]` / `[]` | `[7]` / `[7]` | MATCH / **SEV-2** | R1, SEV-4 | R3, SEV-4 |
| `list<int64>` | `[7]` / `[]` | `[7]` / `[7]` | MATCH / **SEV-2** | R1, SEV-4 | R3, SEV-4 |
| `decimal128(10,2)` | `[7]` / `[]` | `[7]` / `[7]` | MATCH / **SEV-2** | R4, SEV-4 | R4, SEV-4 |

decimal128 is the asymmetric one: SERVABLE on the static side (an f64 lane),
OPAQUE on the row side. So the static loop finds the column, `binder.column`
fails on the row side, and the NATURAL arm's `else { continue }` drops it.
Same wrong answer, different half of the code.

### g. LEFT JOIN legs

Probes: `NATURAL LEFT JOIN s`, `LEFT JOIN s USING (w)`, `LEFT JOIN s USING
(id, w)`, `LEFT JOIN s ON ...`. Struct `w STRUCT(mean DOUBLE)`.

| cell | DuckDB NATURAL LEFT | ours | verdict |
|---|---|---|---|
| `{mean:1.0}` / `{mean:1.0}` | `[{o: 7}]` | `[{o: 7}]` | MATCH |
| `{mean:1.0}` / `{mean:2.0}` | `[{o: None}]` | `[{o: 7}]` | **SEV-2** |
| `{mean:NULL}` / `{mean:NULL}` | `[{o: 7}]` | `[{o: 7}]` | MATCH (by luck) |
| `NULL` / `NULL` | `[{o: None}]` | `[{o: 7}]` | **SEV-2** |
| `NULL` / `{mean:2.0}` | `[{o: None}]` | `[{o: 7}]` | **SEV-2** |

The left-miss row is PRESENT with NULL value columns -- our existing left-miss
NULL pin's shape, unchanged. `LEFT JOIN USING (w)` and `LEFT JOIN ON` are R1
and R2 in every row, all SEV-4.

### h. multiple shared columns

Struct `w` plus `id`. This is where the three forms separate.

| cell | DuckDB N | DuckDB U1 `USING (w)` | ours N | verdict |
|---|---|---|---|---|
| id matches, w does NOT | `[]` | `[]` | `[7]` | **SEV-2** |
| w matches, id does NOT | `[]` | `[7]` | `[]` | MATCH on N |
| both match | `[7]` | `[7]` | `[7]` | MATCH |
| neither matches | `[]` | `[]` | `[]` | MATCH |

Row two is the proof that the forms must be distinguished: NATURAL keys on
`id` AND `w`, `USING (w)` on `w` alone, and they answer differently on the
same data.

### i. the USING / NATURAL output-column merge

Row `(id=5, w={mean:1.0})`, static `(id, w, z=7)`. Column NAMES kept, because
`SELECT *` duplicates a non-key shared column and a dict collapses it.

static `w={mean:2.0}` (`id` equal, `w` unequal -> a LEFT miss on `w`):

```
NATURAL           SELECT *   names ['id','w','z']       rows []
NATURAL LEFT      SELECT *   names ['id','w','z']       rows [(5, {'mean':1.0}, None)]
JOIN USING(w)     SELECT *   names ['id','w','id','z']  rows []
LEFT USING(w)     SELECT *   names ['id','w','id','z']  rows [(5, {'mean':1.0}, None, None)]
LEFT USING(w)     SELECT w   names ['w']                rows [({'mean':1.0},)]
LEFT USING(id)    SELECT *   names ['id','w','w','z']   rows [(5, {'mean':1.0}, {'mean':2.0}, 7)]
LEFT USING(id,w)  SELECT *   names ['id','w','z']       rows [(5, {'mean':1.0}, None)]
LEFT USING(w)     SELECT w.mean                         rows [(1.0,)]
NATURAL LEFT      SELECT w.mean                         rows [(1.0,)]
```

Ours refuses every one of these (R5 for the star forms, R1 for the USING
forms) -- all SEV-4, and the last two are the TASK-127 downstream cell.

**i-scalar**, the same merge with no struct anywhere, which our engine CAN
answer. Row `(id=5, v=1)`, static `(id, v=9, z=7)`:

| cell | static id | DuckDB | ours | verdict |
|---|---|---|---|---|
| `NATURAL LEFT SELECT *` | 6 | `['id','v','z']` `(5,1,None)` | `{id:5,v:1,z:None}` | MATCH |
| `LEFT USING(id) SELECT *` | 6 | `['id','v','v','z']` `(5,1,None,None)` | `{id:5,v:1,v_1:None,z:None}` | MATCH on values; names are TASK-130 |
| `LEFT USING(id) SELECT id` | 6 | `(5,)` | `{id:5}` | MATCH |
| `LEFT USING(id,v) SELECT *` | 6 | `(5,1,None)` | `{id:5,v:1,z:None}` | MATCH |
| `LEFT USING(id,v) SELECT v` | 6 | `(1,)` | `{v:1}` | MATCH |
| `LEFT USING(id) SELECT *` | 5 | `(5,1,9,7)` | `{id:5,v:1,v_1:9,z:7}` | MATCH on values |
| `LEFT USING(id) SELECT id` | 5 | `(5,)` | `{id:5}` | MATCH |
| `LEFT USING(id,v) SELECT v` | 5 | `(1,)` | `{v:1}` | MATCH |

Our merge rule is already right: the LEFT spelling wins, the LEFT value
survives a miss, no COALESCE. The struct head just has to join it.

### j. what the encoding still cannot carry (the backstop cells)

| cell | DuckDB | ours today | verdict |
|---|---|---|---|
| `w STRUCT(mean DOUBLE, t TIMESTAMP)`, mean equal, t equal | `[7]` | `[7]` | MATCH by luck |
| same, mean equal, **t UNEQUAL** | `[]` | `[7]` | **SEV-2** -- no lane exists for `t` |
| `w STRUCT("a.b" DOUBLE, c DOUBLE)`, c equal, `a.b` UNEQUAL | `[]` | `[7]` | **SEV-2** -- the dotted-field flatten skip |
| row `w STRUCT(a,b)` / static `w STRUCT(b,a)` (reordered), equal | `[7]` | `[7]` | MATCH |
| row `w STRUCT(a,b)` / static `w STRUCT(x,y)` (renamed), same values | `[]` | `[7]` | **SEV-2** |
| row `v BIGINT` / static `v STRUCT(mean)` | bind: `Unimplemented type for cast (BIGINT -> STRUCT(mean DOUBLE))` | `[7]` | **SEV-3 + SEV-2** |
| row `W STRUCT(mean)` / static `w STRUCT(mean)`, values differ | `[]` | `[7]` | **SEV-2** -- CI name match |
| `USING (w)` / `USING (W)` / `USING ("W")` on the above | `[]` in all three | R1 in all three | SEV-4 x3 |
| struct on the ROW side only, no shared name | `[7]` | `[7]` | MATCH |
| struct on the STATIC side only, no shared name | `[7]` | `[7]` | MATCH |

### Phase

Every DuckDB cell above answered identically under `PREPARE`, plain execute
and the zero-row leg. The only DuckDB refusal in the whole sweep is an unknown
USING name (`Column "nope" does not exist on left side`), which is bind-phase
in all three. Every one of ours is at `DuckDBInferFn(...)`; the SEV-2 cells all
BUILD cleanly and return the wrong rows, with `infer_rows([])` returning `[]`.

## The design

### What the measurements require

Read the discriminator rows as a specification. For a struct join key, DuckDB's
answer is a structural recursion over the type:

```
match(level 0, l, r)   =  l IS NOT NULL AND r IS NOT NULL AND inner(l, r)
inner(struct l, r)     =  for every field f: node(l.f, r.f)
node(l, r)             =  (l IS NULL) == (r IS NULL)
                          AND (both NULL OR (leaf ? l == r : inner(l, r)))
leaf ==                =  DuckDB scalar equality: NaN == NaN, -0.0 == 0.0
```

Level 0 propagates NULL (so a NULL struct misses even against another NULL
struct); every level below carries its own validity as a VALUE. That is
`row_matcher.cpp:379-382` written out: top-level `Equals`, child predicate
`NOT_DISTINCT_FROM`.

### Chosen: leaf lanes as INDF keys, plus one presence key per nullable struct node

A struct key `w` expands, at bind time, into a LIST of ordinary key columns:

1. **One plain (NULL-propagating) presence key for the top-level struct**,
   when `w` is declared nullable. Its value is the struct's own validity, and
   because it is a PLAIN key it annihilates exactly like any scalar key does
   today -- the build side drops NULL-key rows (`duckdb/mod.rs:768-774`,
   `continue 'row`) and the probe side ANDs the validity flag into
   `keys_valid` (`lower.rs:1952-1956`). That reproduces "NULL struct misses,
   on either side, even against another NULL struct" with zero new runtime
   semantics.
2. **One INDF presence key per NULLABLE INNER struct node**, at every depth.
   INDF because an inner node's NULL is a VALUE: NULL-node matches NULL-node,
   NULL-node misses present-node. This is the pair that the DISCRIMINATOR
   cells demand and that leaf lanes alone cannot supply.
3. **One INDF key per LEAF lane**, carrying `(validity, payload)`. INDF
   because a NULL field equals a NULL field.

Leaves pair BY FIELD PATH -- walk both `StructCol` trees by name,
case-insensitively, matching DuckDB's by-name cast. A reordered field set
pairs fine and serves; the paths are already the encoding TASK-132 installed.

Everything this needs at RUNTIME already exists and is already tested:

- `key_indf: Vec<bool>` per key column, already on `JoinSpec`
  (`plan.rs:220-224`) and `StaticSpec` (`specializer/mod.rs:59-62`).
- The `(validity i1, payload)` flattening of an INDF key
  (`lower.rs:258-268`), the probe-side masked encoding
  (`lower.rs:1941-1951`), and the build-side "keep NULL as `(false, typed
  default)`" materialisation (`duckdb/mod.rs:754-767`).
- Composite keys: `StaticData::Map(Vec<(Vec<KeyBits>, Vec<ScalarVal>)>)`
  (`exec/mod.rs:363`). A struct key is just several entries in that vector.
- Per-key SEGMENT PATHS on the static side: `StaticSpec::key_cols` is already
  `Vec<Vec<String>>` and the materialiser already walks nested dicts down a
  path, returning NULL when any node on the way is NULL
  (`duckdb/mod.rs:707-727`). A struct LEAF is already a legal key column
  there; nothing about that half has to change.
- Float canonicalisation: `canon_f64_bits` (`exec/mod.rs:323-331`) already
  matches DuckDB at every nesting depth, per (c) and (d).

So the new machinery is exactly one thing: **a PRESENCE lane** -- a value that
is "is the node at this path non-NULL", for the row side, and the same for the
static side.

**Static side: no lane needed.** Add `key_kind: Vec<KeyKind>` (`Value` |
`Present`) alongside `StaticSpec::key_cols`. For `Present`, the materialiser
does `KeyBits::I1(!get(path).is_none())` instead of `convert(...)`. Statics
materialise once at build, so this is free at serving time.

**Row side: mint the lane LAZILY, in the frontend, only when a struct becomes
a join key.** `frontend()` returns the minted lanes as
`Vec<(Col, Vec<String>)>` (a synthetic `Ty::I1` non-nullable column plus the
node path it reads); `prepare_opaque` appends them to `in_cols` before
`lower`; `Prepared` carries them out; `duckdb/mod.rs` extends `in_cols` and
`in_paths` before building the `Marshaller`. The marshaller's per-lane walk
(`duckdb/mod.rs:1751-1771`) already breaks at the first `None` on the path --
a presence lane pushes `I1(!attr.is_none())` after walking to the NODE instead
of the leaf. No mutable-reference plumbing, no signature churn beyond one
returned vector.

**Why LAZY and not eager.** Minting presence lanes at schema-parse time (in
`build_fields` / `flatten_static`) would be less code, but it widens `in_cols`
for every query over a struct-carrying row model, joined or not. Measured on
this worktree, 50k rows, `SELECT k` touching nothing else, minimum of 15
runs -- one extra unreferenced input lane costs **+22 to +26 ns/row** at the
marshalling boundary, against a ~200 ns/row floor for a trivial query
(a linear sweep over 0/1/2/4/8 unreferenced plain lanes gives ~27 ns/row per
lane; the medians on this box are too noisy to quote, the minima are stable).
Arrow schemas default to `nullable=True`, so eager minting would charge that
to essentially every struct column in the repo. Since inference is
boundary-bound, a boundary lane is the expensive kind, and a feature used by
one query shape should not tax the rest.

### Rejected: key on the physical bits without serving the value

Encode the whole struct as one canonical byte string (or one
`KeyBits::Str`) per key column: serialise the static side's dict at build with
NaN/-0.0 canonicalisation and an explicit NULL-node marker, and serialise the
probe side's struct per row. It is rejected on two independent grounds.
First, the probe side has no struct VALUE at runtime -- TASK-56 flattened
structs to lanes and there is nothing left to serialise; reconstructing one
means a per-row walk that builds a string, which is an allocation on the hot
path in an engine whose whole doctrine is compile-once with no runtime
decisions. Second, it buys nothing where it would be uniquely useful: an
opaque column (TIMESTAMP, DATE, LIST) has no lane on EITHER side, so
serialising it would first require ingesting it -- which is the key-only-lane
ticket, not this encoding. The chosen design, by contrast, adds one i1 lane
per nullable node and reuses the INDF key machinery already shipped and pinned
by `specializer/tests.rs:868-1000`.

### The refuse-by-name backstop

Both arms stop guessing. The NATURAL arm's silent `else { continue }`
(`frontend.rs:873-875`) is the bug's actual mechanism and is deleted: a shared
name that cannot become a key REFUSES, naming the column and the reason. The
USING arm stops claiming a column "does not exist on right side of join!" when
it plainly does.

Both arms must first WIDEN their scan. NATURAL iterates `st.cols` only
(`frontend.rs:867`) and USING scans `st.cols` only (`frontend.rs:837-845`);
neither visits `st.structs` or `st.opaque`, which is why a struct or opaque
head is invisible. The shared-name set becomes `st.cols` (non-leaf lanes) +
`st.structs` + `st.opaque`, matched case-insensitively, mirroring DuckDB's
`case_insensitive_set_t` at `bind_joinref.cpp:185-208`.

Refuse by name, at build, for:

| condition | why |
|---|---|
| a shared OPAQUE column (either side) | no lane exists on either side; naming the arrow type as R3 already does |
| a shared column opaque on ONE side only (decimal128) | same, and the message must say WHICH side |
| a struct key containing an `Opaque` field at any depth | no lane for that field; matrix (j) row 2 |
| a struct key containing a field whose name has a dot | `flatten_static` / `build_fields` skip it, so no lane; matrix (j) row 3 |
| a struct key whose two sides' field NAME SETS differ | DuckDB answers constant FALSE; we have no leaf to pair. Refusing is SEV-4, honest, and the query is broken anyway |
| a shared name that is a struct on one side and a scalar on the other | DuckDB refuses too (`Unimplemented type for cast`); matrix (j) row 6, currently our SEV-3 |
| a struct key under `shape='many'` | `lower.rs:134-140` already refuses INDF keys under `many`; the message must name the struct column, not just "IS NOT DISTINCT FROM" |

Each refusal names the column and says what about it cannot be keyed. None of
them silently drops a column from the key set -- that is the criterion.

### The remaining pieces

- `JoinSpec::key_cols: Vec<u32>` becomes a richer per-key descriptor
  (`Lane(u32)` | `Present(path)`), because a presence key has no `st.cols`
  entry. `promote_key` (`frontend.rs:1402-1420`) and `key_lane`
  (`frontend.rs:3400-3425`) both index `st.cols[col]` and follow it.
- `val_cols` (`frontend.rs:891-893`) currently takes every `st.cols` index not
  in `key_cols`. A struct key's LEAF lanes must leave `val_cols` too, or the
  struct's leaves would be emitted as separate output columns on the static
  side.
- The USING merge exemption in `Binder::column` (`frontend.rs:3812-3821`)
  skips a USING join's `key_cols`, but the `join_opaque` lookup at
  `frontend.rs:3826-3841` scans `sj.table.structs` and `sj.table.opaque`
  UNCONDITIONALLY. A struct head that is now a USING/NATURAL key must be
  exempted there the same way, or `SELECT w.mean` after `NATURAL JOIN s` keeps
  saying `ambiguous column 'w'` instead of resolving to the left occurrence.
- `NATURAL LEFT JOIN` / `LEFT JOIN USING (w)`: nothing new. The left-miss row
  shape is the existing one, and (i-scalar) shows our merge rule already
  matches DuckDB, including "the LEFT value survives a miss, no COALESCE".

## Behavior flips

Each is a bug fix; each gets a live-oracle pin.

1. **SEV-2 -> MATCH.** A NATURAL join keys on a shared STRUCT column. Every
   SEV-2 row in matrix (a), (b), (c), (g) and (h) flips. This is the ticket's
   severity-2 and it includes the `NULL struct` cells that today serve a row
   DuckDB does not.
2. **SEV-4 -> MATCH.** `JOIN s USING (w)` and `LEFT JOIN s USING (w)` with a
   struct key serve, and R1 (`column "w" does not exist on right side of
   join!`) is gone. All of matrix (a)/(b)/(c)/(g)/(h) column U/U1.
3. **SEV-2 -> SEV-4.** A shared OPAQUE column (TIMESTAMP, DATE, TIME, FLOAT32,
   UINT64, LIST, and decimal128 on the row side) refuses by name instead of
   silently dropping from the key set. Matrix (e) and (f). A wrong answer
   becomes an honest refusal -- the acceptable cost on the ladder, but see
   Open questions, because the criterion's text asks for MATCH here.
4. **SEV-2 -> SEV-4.** A struct key with an unlaneable field (opaque type, or
   a dotted field name) refuses by name. Matrix (j) rows 2-3.
5. **SEV-2 -> SEV-4.** A struct key whose two sides' field name sets differ
   refuses by name. Matrix (j) row 5.
6. **SEV-3+SEV-2 -> SEV-4.** A shared name that is a scalar on one side and a
   struct on the other refuses by name; DuckDB refuses too, so this is
   MATCH-as-refusal in substance. Matrix (j) row 6.
7. **SEV-2 -> MATCH.** Case-insensitive shared-name matching over struct and
   opaque heads: row `W STRUCT(mean)` against static `w STRUCT(mean)` becomes
   a key. Matrix (j) rows 7-8.
8. **The two TASK-133 xfail(strict) legs in
   `packages/confit/tests/test_open_divergences.py` change.** The struct leg
   (`test_a_natural_join_keys_on_every_shared_column[id BIGINT, w
   STRUCT(mean DOUBLE)...]`) FLIPS to passing under flip 1 and its `@xfail` is
   deleted. The TIMESTAMP leg does NOT flip under this spec: it becomes a
   named refusal, so its body must be rewritten to assert the refusal names
   `t`, and it MOVES to `known_divergences/` (or stays xfail against the
   follow-up ticket) rather than being deleted. See Open questions.
9. **SEV-4 -> MATCH, downstream, only if TASK-127's D2 has landed.**
   `NATURAL JOIN s` then `SELECT w.mean` resolves to the left `w` (1.0)
   instead of `ambiguous column 'w'`. This fix is its precondition (the
   USING-merge exemption above); the unqualified ladder is D2's.
10. **No flip, stated for the record.** All five scalar-DOUBLE control cells
    in (d), the whole of (i-scalar), the left-miss NULL shape, and the two
    "struct on one side only" cells in (j) already MATCH. The implementation
    must not move them.

## Test plan

Live-oracle pins in `packages/confit/tests/test_arrow_schema_api.py`,
mirroring the existing `_duck132` helper: register the arrow table, `CREATE
TABLE s AS SELECT * FROM sa`, compare `to_pylist()` AND the column-name list.
Every test asserts the ORACLE's answer first (so an oracle move is a loud
failure, not a silent rebaseline), then ours against it. The conftest fixture
already applies `PRAGMA disable_optimizer` to every connection.

The three-way split is structural, not incidental: every value test is
parametrized over `NATURAL`, `USING (id, w)`, `USING (w)`, and
`ON __THIS__.w = s.w`, because matrix (h) row 2 proves the forms answer
differently on the same data.

- `test_a_struct_join_key_matches_duckdb` -- parametrized over all seven
  matrix (a) cells x the four forms. Asserts rows in and rows out.
- `test_a_nested_struct_join_key_matches_duckdb` -- matrix (b), including BOTH
  DISCRIMINATOR cells (`{inner:NULL}` vs `{inner:{val:NULL}}`, and the
  three-deep `{i:{j:NULL}}` vs `{i:{j:{v:NULL}}}`). These two are the reason
  presence keys exist; without them the suite cannot tell the chosen design
  from the rejected one.
- `test_a_struct_join_key_on_float_edges` -- matrix (c): NaN/NaN, -0.0/0.0,
  inf/inf, and NaN-vs-NULL, each in a struct AND as the bare DOUBLE control
  from (d), asserting the two agree.
- `test_a_left_join_on_a_struct_key_keeps_the_left_miss_shape` -- matrix (g),
  all five cells, `NATURAL LEFT` and `LEFT JOIN USING (w)`; asserts the miss
  row is present with NULL value columns.
- `test_natural_and_using_key_on_different_column_sets` -- matrix (h) row 2
  specifically: `NATURAL` misses and `USING (w)` hits on the same fixture.
- `test_the_using_merge_takes_the_left_value_on_a_miss` -- matrix (i) and
  (i-scalar): `SELECT w`, `SELECT *` with the NAME LIST asserted, and
  `SELECT w.mean` after `NATURAL LEFT JOIN`. Asserts NO coalesce.
- `test_an_unkeyable_shared_column_refuses_by_name` -- parametrized over every
  backstop row: TIMESTAMP, date32, time64, float32, uint64, list, decimal128
  (row side), a struct with a TIMESTAMP field, a struct with a dotted field
  name, mismatched field name sets, and scalar-vs-struct. Each asserts (a)
  DuckDB's own answer, (b) that we RAISE at construction, (c) that the message
  contains the column name, and (d) that the message does NOT contain "does
  not exist" -- the false claim this ticket removes.
- `test_a_shared_column_name_matches_case_insensitively` -- matrix (j) rows
  7-8: row `W` vs static `w` under NATURAL, and `USING (w)` / `USING (W)` /
  `USING ("W")`, all four agreeing with the oracle.
- `test_a_struct_key_under_shape_many_refuses_naming_the_column` -- the
  `lower.rs:134-140` path, asserting the message names `w`.
- Rust unit tests in `specializer/tests.rs`, beside the existing
  `indf_*` trio: a map static whose key vector is
  `[present_i1, leaf_valid_i1, leaf_payload]` probes correctly for the four
  node/leaf NULL combinations; and a presence key with a NULL top-level
  struct drops its build row.

**Survive list** (must stay green, untouched): the TASK-116 lane trio, the
left-miss NULL pin, the TASK-121 ambiguity trio, TASK-125 star + EXCLUDE
order, TASK-126 aliases, the TASK-132 lane-path pins, `indf_left_join_null_key
_joins_null_bucket`, `indf_inner_join_null_key_hits`,
`indf_multi_key_conjunction_mixed_with_eq`, and every cell in matrix (d) and
(i-scalar).

**Gate**: full suite in release AND debug (`debug_asserts`), then a 4k fuzzer
campaign with residue attributed against the pre-change baseline. This change
moves the accept/refuse boundary for every join whose shared-column set is not
all-scalar, so the residue needs reading, not counting.

**Boundary regression check**: because the chosen design mints lanes lazily,
`SELECT k` over a struct-carrying row model must marshal the SAME number of
lanes before and after. Assert it directly on `program.in_cols.len()` rather
than by timing.

## Open questions

1. **The NATURAL-keys-on-all criterion says the TIMESTAMP leg flips to
   passing; this spec makes it a named refusal instead.** Serving an opaque
   column as a key needs a new lane KIND -- servable-as-key, not servable-as-
   value -- threaded through `schema.rs`, `cols`/`in_cols`, `star`, the IR
   verifier, the arrow ingest and both marshallers, plus a per-type equality
   proof (timestamp[us] and date32 are exact int64/int32 bit equality;
   float32 injects losslessly into f64; uint64 needs a bit-reinterpreted total
   order for the sorted probe table; timestamp units and timezones need their
   own answer). That is a bigger change than everything in this spec combined.
   The recommendation is to split: this ticket serves struct keys and refuses
   the rest by name (which discharges the severity-2 everywhere, in both
   arms), and a follow-up ticket adds key-only lanes. Confirm, or say the
   criterion is literal and TIMESTAMP must serve here.
2. **A struct key whose two sides' field name sets differ: refuse, or serve as
   a provably-empty join?** DuckDB answers constant FALSE without erroring.
   Refusing by name is SEV-4 and simpler; serving an always-empty join is
   MATCH but interacts with the `shape='map'` one-row proof
   (`specializer/mod.rs`, `one_row_blocker`) and would need that argued.
   Recommendation: refuse.
3. **`shape='many'` plus a struct key.** `lower.rs:134-140` refuses INDF keys
   under `many` outright, so every struct-keyed NATURAL/USING join refuses
   there. Is that acceptable for now, or does `many` need the INDF loop form
   in this ticket?

## Decisions (AmirHossein, 2026-08-25)

- The split: this ticket serves STRUCT keys; opaque scalar keys (TIMESTAMP,
  DATE, FLOAT32, UINT64, LIST, row-side DECIMAL, and the rest of the
  opaque set) REFUSE BY NAME here -- a follow-up ticket adds key-only
  lanes so they serve too. The severity-2 (wrong rows from silently
  dropped keys) dies in both arms in this PR.
- Mismatched struct field name sets: refuse by name (DuckDB serves a
  constant-empty join; ours is a deliberate severity-4 refusal).
- Struct keys where the static side has DUPLICATE key values (the
  fan-out loop, shape='many'): refuse by name -- the fan-out lowering
  only implements plain equality today (the pre-existing NOT-DISTINCT
  gap at lower.rs). Lifting it is its own follow-up ticket. Struct keys
  under the map and filter shapes (unique static keys, LEFT or inner)
  SERVE with full DuckDB semantics.
