# Coerce negated integer literals in builtin calls

**Done when:** lpad accepts the measured INTEGER-range negated literal like DuckDB.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Confit refuses `lpad(x, -2147483648, 'p')` as lpad(VARCHAR, BIGINT, VARCHAR). DuckDB’s typeof also says BIGINT, but its binder casts the negated literal to INTEGER like a plain literal. Measured 2026-10-15, two class-5-part-5 dense cases.

## Wanted

Keep literal-sensitive call coercion separate from general BIGINT narrowing.

## Involved

- `packages/confit/src/specializer/frontend/calls.rs`: literal argument coercion.
- `packages/confit/src/specializer/frontend/functions.rs`: lpad signature.

## Out of scope

Unrelated query classes and changes to the oracle contract.
