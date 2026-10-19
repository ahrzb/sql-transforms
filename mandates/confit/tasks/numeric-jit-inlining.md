# Profile numeric helper costs

**Done when:** A retained serving profile establishes whether to inline DECIMAL checks or HUGEINT operations; any selected kernel has parity and before/after costs.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

DECIMAL casts, capped arithmetic and rounding call one helper per row. Historical cost is about 110 ns/row versus 55 for DOUBLE (`packages/confit/docs/reports/2026-09-28-decimal-expressions.md`). Each i128 operation also calls a helper.

## Wanted

Profile first. Inline measured hot checks or checked iadd.i128 without changing overflow behaviour.

## Involved

- `packages/confit/src/specializer/exec/cranelift.rs`: numeric helper calls.
- `packages/confit/tests/test_decimal_expressions.py`: decimal boundaries.
- `packages/confit/tests/test_hugeint.py`: i128 overflow.

## Out of scope

Inlining without evidence of a serving hotspot; exact wide aggregate sum/product.
