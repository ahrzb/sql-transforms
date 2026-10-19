# Serve COLUMNS and a lambda/list expression slice

**Done when:** Measured COLUMNS-in-expression forms and one bounded lambda/list slice serve with named residual refusals.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

COLUMNS inside expressions and lambda/list forms refuse.

## Wanted

Measure expansion and lambda scope rules. Implement independently checkable forms without changing existing list-literal semantics.

## Involved

- `packages/confit/src/specializer/frontend/expr.rs`: COLUMNS and lambda binding.
- `packages/confit/tests/test_list_literals.py`: list expressions.

## Out of scope

Regex engine replacement or all list functions in one session.
