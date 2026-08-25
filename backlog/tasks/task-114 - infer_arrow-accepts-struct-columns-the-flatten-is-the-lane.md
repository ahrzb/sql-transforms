---
id: TASK-114
title: >-
  infer_arrow accepts struct columns — the arrow flatten IS the lane
status: Done
assignee: []
created_date: '2026-08-15 13:40'
labels:
  - m-8
dependencies: []
type: feature
ordinal: 99000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`infer_rows` serves struct row columns; `infer_arrow` refuses them:

```python
S = pa.schema([pa.field("st", pa.struct([pa.field("x", pa.int64(), nullable=False)]), nullable=False)])
fn = DuckDBInferFn("SELECT st.x + 1 AS o FROM __THIS__", row_tables={"__THIS__": S}, static_tables={})

fn.infer_rows([{"st": {"x": 41}}])   # [{'o': 42}]
fn.infer_arrow(table)                # refuses: "infer_arrow requires an all-scalar row schema"
```

The two entry points disagree about what the schema means, which is the
TASK-71 lesson again — and this one costs the columnar path on exactly the
workloads that most want it (a wide fitted feature bundle is naturally one
struct column, not forty top-level ones).

The refusal was correct when row schemas were pydantic models: nothing in
`pa.Table` obviously corresponded to a nested model. After PR #144 the
mapping is direct — TASK-56 already flattens a struct row column to
`parent.leaf` lanes internally, and arrow hands us exactly those lanes:

```python
t.column("st").type              # struct<x: int64 not null>
t.column("st").flatten()         # [ChunkedArray<int64>]   <- IS the `st.x` lane
```

So this is a boundary change, not an engine change: bind, lowering, and
both backends already handle `st.x`. What is missing is the ingest arm that
walks a `StructArray`'s children instead of refusing at the schema check,
and the symmetric decision on output.

**The one real subtlety:** `flatten()` folds the parent's validity into each
child, so under a nullable parent a child arrives nullable regardless of its
own field flag. A NOT NULL child inside a nullable parent must therefore not
be read straight from the child buffer — the null lane is the OR of the two
levels, which is exactly what the row path computes for a missing/None
parent today. Get this wrong and a null parent silently reads a garbage
child value instead of refusing or nulling.

Nested structs (`a.b.c`) follow the same recursion; keep the depth limit the
row path already applies, whatever it is.

**Output side is a separate question, ask before building it:** DuckDB can
return a STRUCT-typed column. `infer_rows` emits it as a nested dict. If
`infer_arrow` should emit a real `pa.struct()` rather than flattened
columns, that changes `output_schema` for existing queries, which is a
user-facing format change and needs explicit approval. Do the ingest half
first; it is additive and refuses nothing that works today.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 infer_arrow accepts a struct row column and agrees with infer_rows
      row-for-row on the same data — the two entry points stop disagreeing
- [x] #2 a NOT NULL child under a nullable parent is served through the OR
      of both validity levels, with a test that a null parent does NOT leak
      the child's buffer value
- [x] #3 nested structs work to the same depth the row path allows
- [x] #4 a struct column the query never references stays opaque and does
      not block the build, matching the row path's rule
- [x] #5 output-side struct emission is NOT built here — the current
      flattened output is unchanged, and the question is raised separately
- [x] #6 the serving bench gains a struct-column scenario, so the columnar
      win over infer_rows on this shape is measured rather than assumed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
Landed per the spec (packages/confit/docs/specs/2026-08-25-task-114-design.md),
on the post-TASK-132 surface: ingest walks each leaf lane's segment path
through StructArray children and serves through the AND of every
ancestor's validity with the leaf's own -- the load-bearing line is one
closure. The fold hazard got its own red: with the walk in and the fold
deliberately out, a NULL parent served 999999+1 straight from the child
buffer (pinned with a loud mask-built fixture, never a zero). Sliced,
chunked, empty, wide, and nested-to-the-row-path's-depth batches are all
pinned against the live oracle. Output stays flattened; the output-side
struct question remains open per the ticket's own fence.

The fuzzer's rows_only escape hatch is deleted: 636 struct-bearing
cases per 4k campaign now run the whole infer_arrow boundary battery,
adding zero residue. The serving bench gained feature_bundle; measured
clean-columnar (pre-built batch) infer_arrow beats infer_rows ~1.8-1.9x
at n>=1024 and loses below the crossover between n=64 and n=1024.
<!-- SECTION:NOTES:END -->
