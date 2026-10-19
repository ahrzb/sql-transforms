# Resolve the RE2-compatible regex boundary

**Done when:** An RE2-compatible route has retained parity evidence and named unsupported syntax; list-valued regex output serves for that measured slice.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

The regex reject list waits on an RE2-compatible engine. List-valued regex forms, lambda/list forms and COLUMNS inside expressions refuse.

## Wanted

Measure the compatible engine route and list-valued output schema before lifting the reject list.

## Involved

- `packages/confit/src/specializer/frontend/functions.rs`: regex/list signatures.
- `packages/confit/src/specializer/frontend/expr.rs`: COLUMNS and lambda refusals.
- `packages/confit/tests/test_list_literals.py`: list forms.

## Out of scope

COLUMNS and lambda/list expressions; accepting regex syntax with different semantics.
