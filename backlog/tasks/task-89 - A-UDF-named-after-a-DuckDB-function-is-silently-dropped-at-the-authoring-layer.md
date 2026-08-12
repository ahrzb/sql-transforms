---
id: TASK-89
title: >-
  A UDF named after a DuckDB function is silently dropped at the authoring layer
status: Done
assignee: []
created_date: '2026-08-11 17:00'
labels:
  - bug
  - sql-transform
dependencies: []
documentation:
  - packages/sql-transform/sql_transform/_marginalize.py
type: bug
ordinal: 82000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
`_marginalize` resolves a scalar call as a UDF only when the name is NOT in
`_known_functions()` — so a declared UDF named after any DuckDB function is
never resolved, never recorded, and the call plans as the BUILTIN:

```text
marginalize("SELECT myfn(x) AS o FROM __THIS__", ["x"], {"myfn": U()})
  scalar_udfs=('myfn',)      resolved as a UDF

marginalize("SELECT abs(x) AS o FROM __THIS__", ["x"], {"abs": U()})
  scalar_udfs=()             the declared UDF is GONE; abs() serves the builtin
```

The user declared a callable and got someone else's function, with no error.

confit refuses exactly this collision at its own layer (PR #95, "a udf may
not take a builtin's name") — but the refusal never fires, because the
dropped UDF never reaches confit's `udfs=` list. Two layers, one name, and
the lower one's guard is unreachable from the upper one.

Reproduced by hand 2026-08-11 (the output above is the actual probe).
Originally spotted in the 2026-08-09 adversarial sweep and left unticketed;
the confit half was fixed then, this half was not.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 A scalar call whose name is BOTH a DuckDB function and resolvable
      in the registry / caller scope refuses by name at marginalize time
- [ ] #2 The message names the collision and says to rename, matching the
      confit-layer refusal's wording so the two layers read as one rule
- [ ] #3 A UDF with a non-colliding name still resolves (control), and a
      builtin call with NO declared UDF of that name still plans as the
      builtin (control)
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The check belongs where `scalar_udf` is dispatched (`_marginalize.py:1082`):
today it is `if fname not in _known_functions()`, which conflates "is a
builtin" with "is not a UDF". Resolve against the registry FIRST; a name
that resolves in both places is the refusal.
<!-- SECTION:NOTES:END -->
