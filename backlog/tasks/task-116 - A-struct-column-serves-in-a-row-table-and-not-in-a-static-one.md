---
id: TASK-116
title: >-
  A struct column serves in a row table and not in a static one
status: To Do
assignee: []
created_date: '2026-08-15 16:00'
labels:
  - m-8
dependencies: []
type: feature
ordinal: 101000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
The same column binds or refuses depending only on which table it sits in:

```python
STRUCT = pa.struct([("mean", pa.float64())])

# ROW table: serves — TASK-56 flattens it to the `w.mean` lane
DuckDBInferFn("SELECT w.mean AS o FROM __THIS__",
              row_tables={"__THIS__": pa.schema([("w", STRUCT)])}, static_tables={})

# STATIC table: unserved
DuckDBInferFn("SELECT s.w.mean AS o FROM __THIS__ JOIN s ON k = s.id",
              row_tables=..., static_tables={"s": params_with_struct_w})
```

Reported 2026-08-15 by the session working `marginalize`'s serving gap.
Its caller can sidestep this by emitting flat columns instead of a struct
(and probably should regardless), so nothing is blocked — but any user
handing confit a static table with a struct column hits it.

**The message half is already fixed.** PR #157 made the refusal name the
type (`static table 's' column 'w' has type struct, which this engine does
not serve`) instead of the old `column 'w' does not exist in 's'`, which
sent readers hunting a typo in a correct query. What remains is serving it.

**Why it should be cheap.** A static table is `map(keys) -> value columns`,
each value column a lane of one computed type. A struct's leaves are
exactly that set of lanes, and TASK-56 already walks them for row tables —
`schema::arrow_static_schema` returns `RowField::Struct` today and the
catalogue simply files it under `opaque` instead of flattening it. Execution
likely needs nothing.

**Resolution rules to honour** (measured against DuckDB, these bite):

```
w.mean          -> 2.0    column w, field mean  ... but with a TABLE named w in scope:
w.mean          -> 99.0   the table wins; table-first, column-second
__THIS__.w.mean -> 2.0    qualifying forces the struct read
(w).mean        -> 2.0    parens force "w is an expression"

w.x.y.z.a       -> 1      \  same name set, different order,
w.z.y.x.a       -> 2      /  different value
w.a             -> Binder Error: Could not find key "a" in struct
w.x             -> {'y': {'z': {'a': 1}}}   a struct is a legal value
```

A lane must be addressed by the FULL ORDERED PATH, and a lookup either walks
it exactly or fails. Key by leaf name, by a name set, or by suffix match and
those two five-hop paths collapse into one while `w.a` starts finding
something instead of erroring.

Also undecided: a flat arrow column literally named `"w.x.y.z.a"` sitting
beside a struct `w`.

Pinned xfail-strict as `test_struct_static_column_serves_its_lanes` in
test_known_divergences.py.

Related: TASK-114 is the same shape on the OTHER boundary (`infer_arrow`
refusing a struct ROW schema). Neither blocks the other.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 a struct static column's leaves bind as lanes, matching DuckDB
      value-for-value and type-for-type
- [ ] #2 a lane is addressed by its full ordered path — `w.x.y.z.a` and
      `w.z.y.x.a` stay distinct, and `w.a` still errors
- [ ] #3 a TABLE named `w` beats a COLUMN named `w`; qualifying or
      parenthesising forces the struct read
- [ ] #4 a struct static column that is never referenced still builds
- [ ] #5 nested structs work to the depth the row path allows
- [ ] #6 the xfail-strict pin flips and its reason line is deleted
<!-- AC:END -->
