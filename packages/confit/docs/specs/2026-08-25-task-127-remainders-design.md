# The unqualified reference, and what is left of collision detection (TASK-127)

Decision ground: packages/confit/docs/rfcs/2026-08-19-static-struct-lane-encoding.md
(accepted, alternative A). Predecessor: TASK-132, merged -- static struct
lanes carry a structured path, the dotted name is display only
(packages/confit/docs/specs/2026-08-25-static-struct-lane-path-design.md).
This spec covers exactly that spec's first two non-goals.

Everything below was measured on 2026-08-25 against DuckDB 1.5.5 with
`PRAGMA disable_optimizer`, and against confit built from this worktree
(`uv run maturin develop --release`). Bind-vs-execute claims are split:
each oracle cell was run as `PREPARE p AS <sql>` on its own connection,
as a plain execute, and as a zero-row leg (`... WHERE 1=0`). Every
refusal in the matrix below reproduced identically in all three, so
every one of them is bind-phase.

## Goal

Two TASK-127 acceptance criteria, by name:

- The unqualified-reference criterion: "an unqualified `w.mean` resolves
  like DuckDB, or refuses with a message that names the real problem."
- The collision-detection criterion: "a flattened leaf colliding with a
  real sibling column name is detected -- at build, by name, not by a
  struct-key lookup failure."

## Non-goals

- A star that would emit a whole struct value. `SELECT s.*` over a table
  with a struct column still refuses by name (TASK-125's rule). Six
  matrix cells refuse for that reason alone and are unchanged here.
- Struct columns as USING / NATURAL join KEYS. DuckDB has no type check
  there and joins structs by equality; our key encoding is scalar. This
  needs its own ticket (see "Discovered, needs its own ticket").
- `SELECT w.*` struct-field expansion (DuckDB expands a struct's fields;
  we refuse "table 'w' in wildcard does not exist in FROM").
- ORDER BY, column-list aliases over a struct column: pre-existing
  unrelated refusals that showed up in the sweep.
- The `fname.contains('.')` skip that drops a struct field whose own
  name contains a dot (duckdb/mod.rs:182, :1305). Still its own ticket.

## DuckDB's unqualified ladder, source-pinned (v1.5.5, read-only audit)

The RFC already pins the QUALIFIED path. This is the unqualified entry
point, which it did not cover.

The dispatcher is `ExpressionBinder::QualifyColumnName(ColumnRefExpression&,
ErrorData&)` in `src/planner/binder/expression/bind_columnref_expression.cpp`
(lines 449-501). Note there are two overloads of that name; the other,
`(const ParsedExpression&, const string&, ErrorData&)` at 57-110, is the
single-name resolver the ladder calls into.

TWO PARTS `a.b`, block at 475-497, in order:

1. table `a`, column `b`   (line 482, via `binder.GetMatchingBinding`)
2. column `a`, field `b`   (lines 490-494, `CreateStructExtract`)
3. implicit `struct_pack`  (line 496, last resort; treats the parts as
   schema.table / catalog.table, NOT as column.field)

THREE PARTS `a.b.c`, `QualifyColumnNameWithManyDotsInternal` (287-431).
It is a fixed ladder, not a general consume-loop; only the tail is a
loop (442-444, remaining parts become nested `struct_extract`). For
exactly three parts, attempt 1 (catalog.schema.table.column) is guarded
by `size() > 3` and is skipped, so the real order is:

1. catalog.table.column   (317-324)
2. schema.table.column    (325-333)
3. table.column.field     (334-342)
4. column.field.field     (343-350)
5. struct_pack            (351-354)

Lines 356-430 do no binding: they only pick which of the four saved
`ErrorData` objects to surface after total failure.

THE BACKTRACK is `BindContext::GetBinding(alias, column_name, out_error)`
(`src/planner/bind_context.cpp:337-365`) returning `nullptr` at 360-363
when the alias matched a binding but that binding has no such column.
That is the only fall-through: attempt N+1 is reached exactly when the
COLUMN half of attempt N missed.

AMBIGUITY THROWS AND ESCAPES THE LADDER. `BindContext::GetMatchingBinding`
(bind_context.cpp:37-58) throws when a second in-scope binding has the
head name:

```
"Ambiguous reference to column name \"%s\" (use: \"%s.%s\" or \"%s.%s\")"
```

There is no `catch` anywhere in bind_columnref_expression.cpp, so the
struct-extract attempt at 493 and the struct_pack fallback at 496 are
never reached. The verdict is on the HEAD NAME ALONE -- it does not
matter whether either candidate is a struct or has the field. This is
the rule TASK-121 already measured and mirrored; it extends unchanged to
the unqualified multi-part case.

USING keys are exempt: `GetUsingBinding` is checked at line 59, ahead of
everything, and USING columns are skipped inside the ambiguity loop
(41-44).

CASE SENSITIVITY, three different mechanisms, all case-insensitive:

- table/alias: `BindingAlias::Matches`, `StringUtil::CIEquals`
  (src/planner/binding_alias.cpp:47-62).
- column: a `case_insensitive_map_t` lookup, not a comparison
  (src/planner/table_binding.hpp:92, table_binding.cpp:69-77). A
  case-variant column collision is unrepresentable by construction.
- struct field: `StringUtil::Lower` on both sides plus a LINEAR SCAN
  with first-match-wins (src/function/scalar/struct/struct_extract.cpp:59,
  65-73). An ambiguous case-insensitive field match does NOT error --
  the lowest ordinal wins silently. Our `walk_fields` uses
  `eq_ignore_ascii_case` and returns the first hit, which agrees.

ERROR SHAPES (all bind-phase; `Binder::Bind(PrepareStatement&)` at
src/planner/binder/statement/bind_prepare.cpp:8-10 runs the full plan
via planner.cpp:120-123):

- head not found, multi-part: `Referenced table "%s" not found!%s`
  (bind_context.cpp:291-292). For a 2-part name the TABLE error is what
  surfaces, because attempt 1 writes the caller's `error` while attempt
  2 writes a discarded `other_error`.
- head not found, single part: `Referenced column "%s" not found in FROM
  clause!%s` (src/common/exception/binder_exception.cpp:14-28).
- table found, column missing: `Table "%s" does not have a column named
  "%s"\n%s` (table_binding.cpp:296-301).
- field missing on a real struct: `Could not find key "%s" in struct\n%s`
  (struct_extract.cpp:75-84). The key is echoed LOWERCASED (line 59).
- field on a non-struct: `Cannot extract field %s from expression "%s"
  because it is not a struct, union, map, or json`
  (bind_operator_expression.cpp:178-184). SQLNULL is allowed through.
- the suggestion tail is `StringUtil::CandidatesMessage`
  (src/common/string_util.cpp:649-661); empty when there are no
  candidates. Prefixes differ per site ("Candidate tables", "Candidate
  bindings", "Candidate Entries").

USING / NATURAL AND NESTED TYPES: `src/planner/binder/tableref/bind_joinref.cpp`
contains NO `LogicalType` inspection at all. NATURAL intersects column
NAME SETS (185-208, `case_insensitive_set_t`); USING takes the list
verbatim (240-247). Both emit an ordinary `ComparisonExpression`, and
STRUCT falls through `default: break` in
`BoundComparisonExpression::TryBindComparison`
(bind_comparison_expression.cpp:86-126). So a struct column IS a legal
join key there.

## The matrix

Fixture A (`base`): static `s(id BIGINT, w STRUCT(mean DOUBLE, sd DOUBLE,
inner STRUCT(val DOUBLE)), z BIGINT)`, row `__THIS__(k BIGINT)`,
`JOIN s ON s.id = k`.
Fixture B (`collide`): static `s(id BIGINT, w STRUCT(mean DOUBLE),
"w.mean" DOUBLE)` -- the RFC's collision table.
Fixture C (`shared`): row and static BOTH carry `id BIGINT` and
`w STRUCT(mean DOUBLE)`; static also has `z BIGINT`.

Verdicts: MATCH = same answer or an equivalent refusal; SEV-n = the
severity ladder (2 wrong value, 3 serve where DuckDB refuses, 4 refuse
where DuckDB serves); MSG = both refuse, our reason is wrong.

### a. unqualified reference, struct on the static side only (fixture A)

| cell | DuckDB | ours | verdict |
|---|---|---|---|
| `SELECT w.mean` | 1.5, out name `mean` | bind: `unknown table 'w'` | SEV-4 + MSG |
| `SELECT W.MEAN` | 1.5, out name `MEAN` | bind: `unknown table 'W'` | SEV-4 + MSG |
| `SELECT w.inner.val` | 9.0, out name `val` | bind: `unknown table 'inner'` | SEV-4 + MSG |
| `SELECT w.nope` | bind: `Could not find key "nope" in struct` | bind: `unknown table 'w'` | MSG |
| `SELECT w.inner.nope` | bind: `Could not find key "nope" in struct` | bind: `unknown table 'inner'` | MSG |
| `SELECT w` | serves the whole struct | `... column 'w' is a struct -- project its fields instead` | SEV-4, non-goal |
| `SELECT w.inner` | serves the nested struct | bind: `unknown table 'w'` | SEV-4, non-goal + MSG |
| `SELECT q.mean` (no such head) | bind: `Referenced table "q" not found!` | bind: `unknown table 'q'` | MATCH |
| `SELECT z.bad` (scalar + field) | bind: `Cannot extract field 'bad' ... not a struct, union, map, or json` | bind: `Cannot extract field 'bad' ... not a struct` | MATCH |
| `SELECT id.mean` | same as above, for `id` | same as above | MATCH |
| `WHERE w.mean > 1.0` | 1 row | bind: `unknown table 'w'` | SEV-4 + MSG |
| `ON s.id = k AND w.mean > 1.0` | 1 row | bind: `unknown table 'w'` | SEV-4 + MSG |
| `SELECT s.w.mean` (qualified) | 1.5 | 1.5 | MATCH |
| `SELECT main.s.w.mean` | 1.5 | 1.5 | MATCH |
| `SELECT main.s.w.inner.val` | 9.0 | 9.0 | MATCH |
| `SELECT s.nope` | bind: `Table "s" does not have a column named "nope"` | bind: `column 'nope' does not exist in 's'` | MATCH |

### b. ambiguous head (row column and static struct share the name)

| cell | DuckDB | ours | verdict |
|---|---|---|---|
| row `w BIGINT` + static struct `w`, `SELECT w.mean` | bind: `Ambiguous reference to column name "w" (use: "__THIS__.w" or "s.w")` | bind: `ambiguous column 'w' (qualify it)` | MATCH |
| same, `SELECT w` | ambiguous | `ambiguous column 'w'` | MATCH |
| row struct `w` + static struct `w`, `SELECT w.mean` | ambiguous | `ambiguous column 'w' (qualify it)` | MATCH |
| same, case-varied `SELECT W.MEAN` | ambiguous on `"W"` | ambiguous | MATCH |
| TWO statics, both with struct `w` | bind: ambiguous, `(use: "s.w" or "s2.w")` | bind: `unknown table 'w'` | MSG |
| static struct `w` + a second static with scalar `w` | bind: ambiguous | bind: `Cannot extract field 'mean' from expression "w" because it is not a struct` | MSG |
| `SELECT __THIS__.w.mean` (row struct) | 7.75 | 7.75 | MATCH |

### c. the join alias is `w` and `w` is also a struct column (fixture A)

| cell | DuckDB | ours | verdict |
|---|---|---|---|
| `JOIN s AS w`, `SELECT w.z` | 7 (table alias wins; the column binds) | 7 | MATCH |
| `JOIN s AS w`, `SELECT w.mean` | 1.5 (alias has no `mean`, BACKTRACK to column `w` . field `mean`) | bind: `column 'mean' does not exist in 'w'` | SEV-4 + MSG |
| `JOIN s AS w`, `SELECT w.w.mean` | 1.5 | bind: `column 'mean' does not exist in 'w'` | SEV-4 + MSG |
| same plus a row column `w`, `SELECT w.mean` | bind: `Ambiguous reference to column name "w" (use: "__THIS__.w" or "w.w")` | bind: `column 'mean' does not exist in 'w'` | MSG |

### d. struct on the ROW side only

| cell | DuckDB | ours | verdict |
|---|---|---|---|
| `SELECT w.mean` | 7.75 | 7.75 | MATCH (already correct) |

### e. the collision table, unqualified (fixture B)

| cell | DuckDB | ours | verdict |
|---|---|---|---|
| `SELECT "w.mean"` (1 quoted part) | 99.0, out name `w.mean` | 99.0 | MATCH |
| `SELECT w.mean` (2 parts) | 1.5, out name `mean` | bind: `unknown table 'w'` | SEV-4 + MSG |
| `SELECT s.w.mean, s."w.mean"` | `['mean','w.mean']` = 1.5, 99.0 | identical | MATCH |
| `WHERE s.w.mean < s."w.mean"` | 1 row | 1 row | MATCH |
| row also has `"w.mean"`, `SELECT "w.mean"` | bind: `Ambiguous reference to column name "w.mean"` | bind: `ambiguous column 'w.mean'` | MATCH |

### f/g. star and EXCLUDE over the collision table (fixture B)

| cell | DuckDB | ours | verdict |
|---|---|---|---|
| `SELECT s.*` | `['id','w','w.mean']`, struct emitted WHOLE, no dedup needed | `column 'w' has a non-scalar type` | SEV-4, non-goal |
| `SELECT *` | `['k','id','w','w.mean']` | same refusal | SEV-4, non-goal |
| `SELECT s.* EXCLUDE (w)` | `['id','w.mean']` = 5, 99.0 | identical | MATCH |
| `SELECT s.* EXCLUDE (w, "w.mean")` | `['id']` | identical | MATCH |
| `SELECT s.* EXCLUDE ("w.mean")` | `['id','w']` | same struct refusal | SEV-4, non-goal |
| `SELECT s.* EXCLUDE ("W.MEAN")` | `['id','w']` (CI match) | same struct refusal | SEV-4, non-goal |
| `SELECT s.* EXCLUDE (w.mean)` unquoted | bind: `Column "w.mean" in EXCLUDE list not found in s` | bind: `column "w.mean" in EXCLUDE list not found in FROM clause` | MATCH (text differs) |

Note: DuckDB refuses `EXCLUDE (w.mean)` unquoted even though a column
literally named `w.mean` exists in `s`; the unquoted 2-part form is not
the same reference as the quoted one. We refuse too.

### h. USING / NATURAL (fixture C unless stated)

| cell | DuckDB | ours | verdict |
|---|---|---|---|
| `JOIN s USING (id)`, `SELECT *` | `['id','w','w','z']` -- DUPLICATE `w`, not deduped | `column 'w' has a non-scalar type` | SEV-4, non-goal |
| `JOIN s USING (w)` -- a STRUCT as the key | serves; `['id','w','id','z']` | bind: `column "w" does not exist on right side of join!` | SEV-4 + MSG, non-goal |
| `NATURAL JOIN s`, `SELECT *` | `['id','w','z']` -- joins on id AND w | `column 'w' has a non-scalar type` | SEV-4, non-goal |
| `NATURAL JOIN s`, `SELECT z`, w VALUES DIFFER | `[]` (w inequality kills the row) | `[{'o': 7}]` | **SEV-2** (see below) |
| `JOIN s USING (id)` then `SELECT w.mean` | bind: ambiguous | bind: `ambiguous column 'w' (qualify it)` | MATCH |
| `NATURAL JOIN s` then `SELECT w.mean` | 1.5 (w is merged, not ambiguous) | bind: `ambiguous column 'w' (qualify it)` | SEV-4, downstream of the SEV-2 |
| fixture B + row `"w.mean"`, `USING ("w.mean")` | joins on the literal column | same, matching rows and misses | MATCH |
| fixture B + row `"w.mean"`, `NATURAL JOIN` | same | same | MATCH |

### i. the ROW-side collision (row `__THIS__(k, w STRUCT(mean), "w.mean")`)

| cell | DuckDB | ours | verdict |
|---|---|---|---|
| `SELECT w.mean` | 1.5 | build: `internal specializer bug: lowered program failed verification: duplicate in column 'w.mean'` | SEV-4 + MSG |
| `SELECT "w.mean"` | 99.0 | same internal-bug refusal | SEV-4 + MSG |
| `SELECT k` -- touches neither | 5 | same internal-bug refusal | SEV-4 + MSG, and it breaks the unreferenced-columns-cost-nothing doctrine |
| `SELECT __THIS__.w.mean, __THIS__."w.mean"` | 1.5, 99.0 | same internal-bug refusal | SEV-4 + MSG |

## Design

Three changes in frontend.rs, one in the lowering verifier. Nothing else
moves.

### D1. `qualified_path` becomes backtrackable

DuckDB's only fall-through between rungs is "the alias matched but that
relation has no such column" (bind_context.cpp:360-363). Our `compound`
has no fall-through at all: R1 and R2 `return self.qualified_path(...)`
unconditionally the moment `parts[0]` (or `parts[1]`) matches a join
name, so cell (c) `JOIN s AS w; SELECT w.mean` dies at
`column 'mean' does not exist in 'w'` instead of retrying `w` as a
column.

`qualified_path` returns `Option<Result<SExpr, PrepareError>>`, the shape
`this_col_with_fields` and `bare_col_with_fields` already use in this
file:

- `None` -- the relation is in scope but `parts[0]` is not one of its
  columns, struct heads and opaque columns included. Backtrack.
- `Some(Ok(e))` -- bound.
- `Some(Err(e))` -- bound to something, and the reference is an error
  (missing key, not-a-struct, whole-struct refusal). Do NOT backtrack;
  DuckDB does not either.

`compound` R1/R2 then `if let Some(r) = ... { return r; }` and otherwise
fall to the next rung. The existing tail (`self.qualified(parts[..])`)
already re-emits the table-column error when every rung misses, which is
what cell (a) `SELECT s.nope` needs to keep.

### D2. `bare_col_with_fields` sees the whole scope

Today it only asks the driving table (`this_col_with_fields`), then
scans static PLAIN columns solely to produce the not-a-struct error.
Static struct heads are invisible to it, which is why the whole of cell
(a) says "unknown table 'w'": `compound` R3 misses, and the tail emits
`self.qualified(parts[0], parts[1])`, whose head is not a relation.

It becomes a head-candidate collection over the same scope `column()`
already walks, in DuckDB's order (ambiguity first, fields second):

1. Collect every binding that HAS the head name:
   - row: `in_cols[..n_plain]`, `opaque`, `structs`;
   - each join: plain value lanes (`!is_leaf_lane`), key lanes only when
     `!sj.using` (USING merges into the left occurrence -- the rule
     `column()` already applies at frontend.rs:3733, and DuckDB's own
     exemption at bind_columnref_expression.cpp:59), `structs`,
     `opaque`.
2. Two or more candidates -> `ambiguous column '{head}' (qualify it)`,
   BEFORE any field is examined. This is the existing `in_join` rule
   generalized from "row hit plus any join hit" to "count across all
   bindings", and it is what fixes the two (b) rows that currently say
   `unknown table 'w'` and `Cannot extract field ... not a struct`.
3. Exactly one candidate -> resolve `fields` against it:
   - row struct -> `walk_struct` (unchanged);
   - static struct -> D3's helper;
   - scalar (row or static) -> the not-a-struct error (unchanged);
   - opaque -> the named non-scalar refusal (unchanged).
4. Zero candidates -> `None`, and `compound`'s tail emits the
   head-not-found error unchanged.

Because `walk_fields` already consumes an arbitrary tail, the 3-part and
4-part unqualified cases in (a) and (i) fall out with no extra code.
`compound`'s rung order already matches DuckDB's ladder for three parts
(R1 = schema.table.column, R2 = table.column.field, R3 =
column.field.field); only the missing backtrack and the missing R3
candidates make it diverge.

### D3. one shared static-struct-path helper

The struct branch inside `qualified_path` (frontend.rs:3482-3531 --
`structs.find(head)`, `walk_fields`, `val_cols.position`,
`static_lane`, and the four `WalkStop` renderings) becomes
`fn static_struct_lane(&self, j: usize, sj: &ScopeJoin, sc: &StructCol,
fields: &[Ident]) -> Result<SExpr, PrepareError>`, called by
`qualified_path` and by D2. No behavior change on the qualified path --
every message keeps its current spelling, so the TASK-116 and TASK-132
pins survive untouched.

### D4. the collision-detection criterion is satisfied on the static
side and misplaced on the row side

Argued from the matrix, not from the ticket text.

On the STATIC side the criterion is now VACUOUS. It was written when the
lane NAME was the encoding, so a leaf named `"w.mean"` and a sibling
column named `"w.mean"` were literally the same key and one of them had
to lose. Under TASK-132 they are different references at every surface
measured: qualified (both serve, `['mean','w.mean']`), unquoted vs
quoted, EXCLUDE by either spelling, ambiguity against a row column of
the same name, USING/NATURAL over the literal column, and the predicate
path. Ten of the eleven collision-fixture cells already MATCH DuckDB;
the eleventh is the unqualified `w.mean`, which fails identically
WITHOUT a collision -- it is D2's problem, not a collision problem.
There is nothing left to detect, because there is nothing left that
collides. The RFC's alternative A replaced "detect and refuse" with "the
collision does not exist", and that is the stronger outcome: DuckDB
serves both spellings, and so do we.

On the ROW side the criterion still has substance, but its remedy is not
a detector either. Row struct leaves also carry a dotted display name
(`build_fields`, duckdb/mod.rs:1295-1336) and the row lane list is the
lowered program's `in_cols`. The IR verifier requires `in_cols` names to
be unique (specializer/ir/verify.rs:120-132), so a row table with a
struct `w{mean}` and a sibling column `"w.mean"` fails verification --
for EVERY query over that table, including `SELECT k`, which touches
neither. The user sees `internal specializer bug`.

That check is now checking a string that stopped being an identifier.
Post-132 nothing resolves a row lane by its display name: `column()`,
`this_col_with_fields` and `qualified()` all scan `in_cols[..n_plain]`
only, and the DATA path uses `plan::lane_paths(&in_cols, &structs)`
(duckdb/mod.rs:1522), which gives a plain column the whole-name path
`[name]` and a leaf its segment path `[w, mean]` -- distinct by
construction. The one remaining by-name consumer is `arrow::ingest`
(duckdb/arrow.rs:157-164), and `infer_arrow` already refuses any row
schema that has struct columns, so it never sees a leaf lane.

So: move the IN-side half of the duplicate check out of the verifier and
into `DuckDBInferFn::new`, over `in_cols[..n_plain]` -- where the name
IS an identifier and `n_plain` is known, next to where `structs` is
built. The OUT-side half stays in the verifier: output names are a real
contract. Result: the row-side collision serves both spellings like the
static side, and two genuinely duplicated plain columns still refuse by
name at build.

### Display rule

Unchanged from TASK-132. One helper renders a path dotted, used only in
error text and `Col.name`. D1-D3 add no new dotted construction: the
ambiguity message in D2 names the HEAD only (`'w'`), matching DuckDB,
which also reports only the head. The unqualified output NAME is the
LAST part (`w.mean` -> `mean`, `w.inner.val` -> `val`), spelled as the
user typed it (`W.MEAN` -> `MEAN`); `default_name` already takes the
last part, so nothing changes there.

## Behavior flips

Each is a bug fix; each gets a live-oracle pin.

1. SEV-4 -> MATCH. Unqualified `w.mean` / `w.inner.val` over a static
   struct serves, in SELECT, WHERE and ON. (D2)
2. MSG -> MATCH. Unqualified `w.nope` says `Could not find key "nope" in
   struct` instead of `unknown table 'w'`. (D2)
3. SEV-4 -> MATCH. `JOIN s AS w; SELECT w.mean` backtracks from the
   alias to the column and serves 1.5; `SELECT w.z` still takes the
   alias. (D1)
4. MSG -> MATCH. Two static tables that both have a struct `w` refuse
   with `ambiguous column 'w'`, not `unknown table 'w'`. (D2)
5. MSG -> MATCH. A static struct `w` beside a static scalar `w` refuses
   as ambiguous, not as `Cannot extract field ... not a struct`. (D2)
6. SEV-4 -> MATCH. A row table with a struct `w{mean}` and a sibling
   column `"w.mean"` builds, and both spellings serve. Today every query
   over it dies with `internal specializer bug`. (D4)
7. No flip, stated for the record: the qualified paths, the row-side
   unqualified path, the ambiguity trio, EXCLUDE by either spelling, and
   the collision table's qualified spellings ALREADY match DuckDB. The
   implementation must not touch them.

## Test plan

Live-oracle pins in packages/confit/tests/test_arrow_schema_api.py,
mirroring the `_duck132` helper (register the arrow table, `CREATE TABLE
s AS SELECT * FROM sa`, compare `to_pylist()` AND `schema`). Reuse
`_ROW116`; add `_S127` (`STRUCT(mean DOUBLE, sd DOUBLE, inner
STRUCT(val DOUBLE))`) and `_STATIC127`.

- `test_an_unqualified_static_struct_path_serves` -- parametrized over
  `w.mean`, `W.MEAN`, `w.inner.val`, `w.sd + 1`, and the same three in a
  WHERE and in an ON residual. Asserts value AND output column name
  (`mean`, `MEAN`, `val`), because the name is half the flip.
- `test_an_unqualified_static_struct_miss_names_the_key` -- `w.nope`,
  `w.inner.nope`, `z.bad`, `q.mean`; asserts the DuckDB reason word
  (`Could not find key`, `not a struct`, `not found`) and asserts
  `unknown table` is NOT in the message.
- `test_an_ambiguous_unqualified_head_refuses_before_the_fields` --
  parametrized over: row scalar `w` + static struct `w`; row struct `w`
  + static struct `w`; two statics with struct `w`; static struct `w` +
  static scalar `w`. Each asserts DuckDB itself raises, and that we say
  `ambiguous`.
- `test_a_join_alias_backtracks_to_a_struct_column` -- `JOIN s AS w`
  with `w.z` (alias wins, 7), `w.mean` (backtrack, 1.5), `w.w.mean`
  (1.5), and the row-column variant (ambiguous on both engines).
- `test_a_row_struct_leaf_and_a_dotted_sibling_both_serve` -- the row
  fixture `(k, w STRUCT(mean), "w.mean")`; asserts `SELECT k` builds,
  and that `w.mean` -> 1.5 and `"w.mean"` -> 99.0 against the live
  oracle. This is the collision-detection criterion's actual deliverable.
- Rust unit tests in specializer/ir/tests.rs: a `StaticTable` with a
  struct tree whose leaf display name equals a plain sibling's name --
  resolution picks the right lane for both spellings; and an `out_cols`
  duplicate still fails verification after the in-side check moves.

Survive list (must stay green, untouched): the TASK-116 lane trio, the
left-miss NULL pin, `test_a_static_struct_path_that_is_not_a_lane_refuses_by_name`,
`test_the_collision_table_serves_both_spellings`,
`test_a_quoted_dotted_name_is_not_a_struct_leaf`,
`test_a_plain_static_column_with_a_dot_in_its_name_serves`,
`test_a_non_ascii_field_name_misses_cleanly`, the TASK-121 ambiguity
trio, TASK-125 star + EXCLUDE order, TASK-126 aliases.

Gate: full suite release AND debug (debug_asserts), then a 4k fuzzer
campaign with residue attributed against the pre-change baseline. The
fuzzer's generator emits unqualified dotted references, so this change
moves its accept/refuse boundary and the residue needs reading, not
just counting.

## Discovered, needs its own ticket (NOT in scope here)

A NATURAL JOIN silently drops from the key set every shared column this
engine cannot serve as a scalar, and then emits rows DuckDB does not.
The NATURAL arm (frontend.rs:864-888) iterates `st.cols`; a struct head
is not in `cols` (it is in `structs`) and an unservable column is not
either (it is in `opaque`), so neither becomes a key.

Measured 2026-08-25, row `(id BIGINT, w STRUCT(mean DOUBLE))` and static
`(id BIGINT, w STRUCT(mean DOUBLE), z BIGINT)`, ids equal and `w` NOT
equal:

```
SELECT z FROM __THIS__ NATURAL JOIN s
  DuckDB : []            (joins on id AND w)
  ours   : [{'o': 7}]    (joins on id alone)
```

The same with a shared `TIMESTAMP` column instead of the struct
reproduces identically, so it is not struct-specific -- it is every
`opaque` column. This is severity 2 (a wrong answer, not a refusal) and
it predates TASK-132: pre-132 the struct head was an `opaque` entry and
was skipped by the same loop. It is out of TASK-127's scope and needs a
ticket of its own; the honest interim behavior is to REFUSE a NATURAL
join whose common-column set contains a column we cannot key on, by
name.
