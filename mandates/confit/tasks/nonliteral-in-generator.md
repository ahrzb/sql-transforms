# Generate nonliteral IN-list items

**Done when:** Seed-gated generation exercises nonliteral IN items without changing existing draws.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

The generator draws only literal IN items. DuckDB evaluates every item, but confit formerly stopped at the first TRUE in WHERE/CASE. The binder now uses NOT over NOT when an item can trap; master had OPT_EMULATED for WHERE b IN (0, a + 1).

## Wanted

Draw nonliteral and trapping items after existing draws with an independent seed gate. Preserve the measured eager item evaluation.

## Involved

- `packages/confit/fuzz/gen.py`: IN-list draws.
- `packages/confit/src/specializer/frontend/expr.rs`: IN trap wrapper.

## Out of scope

Changing existing seeds or re-fixing already delivered eager evaluation.
