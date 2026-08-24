# RFC: the static-struct lane encoding

Status: needs AmirHossein's pick. No code written -- TASK-127 stays
parked on this decision rather than half-built.

## Context

TASK-116 made a static table's struct columns servable by flattening
each struct leaf into its own lane, NAMED by the dotted spelling:
`w STRUCT(mean DOUBLE)` becomes a lane literally called `"w.mean"`
inside `StaticTable::cols`. The name IS the encoding -- nothing else
records that the lane came from a struct.

TASK-127 (dotted lane names leak into name resolution) collects what
that costs. Its open items, by name:

- the unqualified-reference item: `SELECT w.mean` (no table qualifier)
  serves 1.5 on DuckDB; we refuse, and with the wrong reason
  ("unknown table 'w'").
- the collision-detection item: a flattened leaf colliding with a REAL
  sibling column named `"w.mean"` should be detected at build, by name
  -- today it surfaces as a confusing struct-key lookup failure.
- the encoding decision (this RFC): is the dotted NAME the right
  encoding at all? The ticket also requires re-measuring the two items
  above before building, since they were relayed from a review.

That re-measurement happened 2026-08-19. Static table
`s(id BIGINT, w STRUCT(mean DOUBLE), "w.mean" DOUBLE)` -- a struct AND a
literal column whose name is the flattened spelling:

```
SELECT s.w.mean    duck: 1.5   (the struct leaf)      ours: refuses, "Could not find key"
SELECT s."w.mean"  duck: 99.0  (the literal column)   ours: refuses, "ambiguous column"
```

DuckDB SERVES BOTH and distinguishes them -- the two spellings are
different references and its catalog keeps them apart. Our dotted
encoding collapses them into colliding lane names, so we cannot even in
principle serve both: the leaf-vs-literal distinction is destroyed when
the catalog is built. The unqualified `SELECT w.mean` case (no
collision) was also confirmed: DuckDB serves 1.5, we refuse with the
wrong reason.

### How DuckDB does it (source-verified, v1.5.5, 2026-08-24)

DuckDB is structurally alternative A. Verified in the checkout:

- The catalog stores `w STRUCT(mean DOUBLE)` as ONE `ColumnDefinition`
  whose type is the struct; the name map has one whole-string entry per
  column and struct children are never registered in it
  (`column_list.cpp`, `table_binding.cpp`).
- A reference is a `vector<string>` of parts from the parser on
  (`transform_columnref.cpp`): quoting is resolved by the lexer, so
  `s."w.mean"` arrives as 2 parts and `s.w.mean` as 3 -- structurally
  disjoint before the binder ever looks.
- Resolution consumes parts OUTERMOST-FIRST
  (`bind_columnref_expression.cpp`): catalog -> schema -> table ->
  column, and whatever parts remain become nested `struct_extract`
  calls with the field name as a constant argument, resolved to an
  ordinal at bind. Two-part precedence: table.column beats
  column.field, and an implicit struct_pack is the last resort (both
  pinned in DuckDB's own test_implicit_struct_pack.test).
- Head-name ambiguity THROWS before fields are examined
  (`bind_context.cpp::GetMatchingBinding`) -- the rule TASK-121
  measured and mirrored.
- A sweep of src/ found NO dotted-name construction used for name
  resolution anywhere. The two places DuckDB mints dotted strings at
  all are output-only: recursive UNNEST's keep-parent-names aliases,
  and a constant-value fallback in table-function argument binding.
  Neither is ever a lookup key.

## Alternatives

### A. Structured path on the lane

The lane carries a real path (`name: Vec<String>`, or
`name + path: Option<Vec<String>>`); the dotted string becomes a
display detail.

Pros:
- It is what DuckDB structurally IS (source-verified above): parts
  vector in, outermost-first consumption, fields as expression
  structure. Mirroring the reference model instead of encoding around
  it is how the TASK-121 head-name rule already works.
- Collisions stop existing structurally: `s."w.mean"` and `s.w.mean`
  resolve to different lanes, and both serve -- matching DuckDB, which
  no name-based scheme can do.
- The unqualified-reference and collision-detection items fall out of it
  naturally instead of needing their own machinery.
- Ends the wrong-reason error messages (three found so far), which are a
  standing debugging tax on users.

Cons:
- A day-plus refactor: touches every consumer of
  `StaticTable::cols[..].name` (star expansion, the TASK-126 aliases,
  EXCLUDE), and the row-side flatten uses the same convention, so it
  needs the same audit.
- The collision it fixes structurally is a pathological input; most of
  the cost pays for correctness on tables nobody has built yet.

### B. Keep dotted names; refuse collisions loudly at reference time

Keep the encoding; when a reference hits the collision, refuse with a
message naming both origins ("'w.mean' is both a struct leaf and a
column; this engine cannot serve the pair -- rename one").

Pros:
- Cheap: the duplicate already trips the existing ambiguity error, only
  the words are wrong.
- Honest about the limitation, and severity-4 (a refusal, never a wrong
  value).
- The unqualified-reference item can still be built separately on top.

Cons:
- Permanently refuses a query DuckDB serves, and the refusal hinges on
  name-mangling trivia a user cannot be expected to know.
- The wrong-reason messages elsewhere remain; each needs its own patch.
- Leaves the encoding debt in place -- any future name-resolution
  feature can trip over it again, which is how TASK-121, TASK-125, and
  TASK-127 all happened.

### C. Refuse the colliding table at build

If a static table contains both `w STRUCT(mean ...)` and a column
literally named `"w.mean"`, refuse the whole table at pack time.

Pros:
- Simplest possible change; the collision can never be referenced
  because it can never exist.

Cons:
- Violates the unreferenced-columns-cost-nothing doctrine: a user who
  never touches `w` still loses the build.
- Everything B leaves unfixed, C leaves unfixed too.

## Recommendation

A, scheduled as its own specced ticket rather than squeezed into
TASK-127 -- the collision is pathological, but the encoding ALSO blocks
the legitimate unqualified reference and keeps producing wrong-reason
messages. B is the acceptable interim if A does not earn a slot; C is
listed for completeness only.

If A is picked: TASK-127 splits into the encoding refactor (new ticket,
spec first), then its remaining items land on top mostly for free.
