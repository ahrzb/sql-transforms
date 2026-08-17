---
id: TASK-71
title: >-
  infer_arrow silently bypasses a supplied output_model
status: Done
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - boundary
  - parity
dependencies: []
documentation:
  - packages/confit/tests/known_divergences/test_arrow_boundary.py
type: bug
ordinal: 64000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`run_rows` (behind `infer` and `infer_rows`) pushes every output row through
`output_model.model_validate` - validators, coercion, defaulted fields; the code
comments call this "full pydantic semantics ... master parity". `infer_arrow`
goes straight from `arrow::emit` to a `pa.Table` and never touches
`output_model`.

```text
one fn, output_model=Out (validator caps x at 15, adds tag="constant")
infer      : [{'x': 10, 'tag': 'constant'}, {'x': 15, 'tag': 'constant'}]
infer_rows : [{'x': 10, 'tag': 'constant'}, {'x': 15, 'tag': 'constant'}]
infer_arrow: [{'x': 10},                    {'x': 20}]
```

Three documented entry points to one function; two answers.

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/known_divergences/test_arrow_boundary.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 `infer_arrow` returns the same values as `infer` for any
      `output_model`, or refuses by name when one is supplied
- [x] #2 Validators, field defaults and coercion all covered
- [x] #3 The differential in `test_infer_arrow.py` is extended to run WITH an
      `output_model`, not only without one
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The existing infer_arrow differential only ever builds fns WITHOUT an
`output_model`, which is exactly why this survived. Fixing the test shape
matters as much as fixing the code.
<!-- SECTION:NOTES:END -->

## Resolution (2026-08-08): refuse by name

AC #1 offered "same values, or refuse by name", and refusal is the honest
answer here. `infer_arrow` exists to build no Python objects; running every
output row through `model_validate` to honour the model would make it exactly
as slow as `infer`, which is to say pointless. Silently skipping it was the
third mode the contract says does not exist.

The predicate is exact and already existed: `supplied`, at the constructor.
A SYNTHESIZED output model — the default — has no validators, no defaults and
no coercion, so the columnar path is equivalent for it and stays available.
Only a model the caller handed us can change an answer, and only that refuses.
It is now stored as `output_model_supplied` (both constructor sites, including
the static-only `Engine::Constant` one).

AC #3 mattered as much as the code, and is why this survived: the differential
in `test_infer_arrow.py` only ever built fns WITHOUT an `output_model`, so the
one path that ignored it was never compared. It now runs every serving
scenario with one — supplying the fn's OWN synthesized model, so shape is held
constant and the only variable is that it was supplied.
