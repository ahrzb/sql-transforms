# Serve capped DECIMAL comparison families

**Done when:** Capped-width join keys, IN and BETWEEN serve or retain a precisely tested named boundary.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

A DECIMAL expression against a VARCHAR build key, a key pair past 38 common digits, and capped IN/BETWEEN families refuse. `cmp_common` preserves max scale at width 38; `dec_common` can instead round. DuckDB’s cast can trap per row.

## Wanted

Keep scalar comparison coercion distinct from result/ANY coercion when extending these families.

## Involved

- `packages/confit/src/specializer/frontend/decimal.rs`: `cmp_common`, `dec_common`, `promote_key`, `unify_family`.
- `packages/confit/tests/test_decimal_expressions.py`: capped comparisons.

## Out of scope

Changing the already fixed scalar comparison semantics (#476 follow-up).
