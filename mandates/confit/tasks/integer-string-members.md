# Parse integer string members exactly

**Done when:** BIGINT IN and BETWEEN tests retain integers past 2^53 and match conversion trap timing.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Numeric string members currently convert through f64 and round to BIGINT. `b IN ('9007199254740993')` can compare the wrong integer. UBIGINT/HUGEINT members refuse by name. Non-numeric strings mixed with numbers also refuse because bind-time conversion is too early.

## Wanted

Use the integer kernels at the family’s type. Defer conversions that DuckDB performs at execution.

## Involved

- `packages/confit/src/specializer/frontend/expr.rs`: IN/BETWEEN family binding.
- `packages/confit/src/specializer/frontend/typing.rs`: family coercion.

## Out of scope

General VARCHAR-to-DECIMAL parsing and unrelated casts.
