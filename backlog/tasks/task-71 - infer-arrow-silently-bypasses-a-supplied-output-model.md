---
id: TASK-71
title: >-
  infer_arrow silently bypasses a supplied output_model
status: To Do
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - boundary
  - parity
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
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
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 `infer_arrow` returns the same values as `infer` for any
      `output_model`, or refuses by name when one is supplied
- [ ] #2 Validators, field defaults and coercion all covered
- [ ] #3 The differential in `test_infer_arrow.py` is extended to run WITH an
      `output_model`, not only without one
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The existing infer_arrow differential only ever builds fns WITHOUT an
`output_model`, which is exactly why this survived. Fixing the test shape
matters as much as fixing the code.
<!-- SECTION:NOTES:END -->
