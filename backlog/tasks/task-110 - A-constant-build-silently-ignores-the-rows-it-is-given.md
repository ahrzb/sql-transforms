---
id: TASK-110
title: >-
  A constant build silently ignores the rows it is given
status: Done
assignee: []
created_date: '2026-08-14 10:00'
labels:
  - m-8
dependencies: []
type: task
ordinal: 96000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
A static-tables-only query compiles to `Engine::Constant`: DuckDB evaluated
it once at build and the fixed rows are replayed on every call. `run_rows`
returns those rows **whatever it is handed** — the input is dropped without
a word:

```python
fn = DuckDBInferFn("SELECT sum(v) AS o FROM s", row_tables=..., static_tables=statics)
fn.backend                      # "constant"
fn.infer_rows([])               # [{'o': 6}]   correct
fn.infer_rows([{"a": 1.0}] * 3) # [{'o': 6}]   input silently ignored
```

Every other input mistake at this boundary refuses by name (missing field,
wrong type, out of range, non-nullable None). Silent-ignore is the odd one
out, and it hides a real caller bug: serving N request rows through a
function that structurally cannot see them returns 1 fixed row, and the
caller's zip/positional assumption breaks somewhere downstream instead of
here. Same class as TASK-71's lesson (three entry points to one function
must not answer differently).

Found while writing the arrow schema API (PR #144), not caused by it —
`Engine::Constant` behaved this way before the migration too.

Fix: refuse non-empty `rows` on a constant build, naming what happened —
the query reads only static tables, so it emits fixed rows and
`infer_rows([])` is the call. `infer_arrow` already refuses on this engine
(and its message now points at `infer_rows([])`), so this closes the last
inconsistent entry point.

Note `shape="map"` already refuses to BUILD a constant engine (fixed rows
cannot be one-out-per-row-in), so this only affects the default "filter"
and "many" shapes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 a constant build refuses non-empty rows by name, mentioning infer_rows([])
- [x] #2 infer_rows([]) on a constant build still serves the fixed rows unchanged
- [x] #3 the refusal is a ValueError, consistent with every other boundary refusal
- [x] #4 a compiled (non-constant) build is untouched — no cost on the hot path
<!-- AC:END -->

2026-08-15 done. Two callers in this repo were relying on the silent drop
and both are now explicit: `test_static_only_query_is_a_constant_emitter`
asserted "input rows are irrelevant" in prose and in code, and the corpus
replay harness fed every case its driving table's rows. The harness now
checks `fn.backend == "constant"` and makes the empty call the ticket
documents. That both existed is the argument for the ticket, not against
it — each was a caller quietly getting an answer unrelated to its input.
