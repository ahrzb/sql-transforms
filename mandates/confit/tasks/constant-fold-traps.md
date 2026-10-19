# Preserve NULL-folded constant traps

**Done when:** A failing constant CAST beside a NULL-folded sibling matches DuckDB; deliberate zero-row refusals remain explicit.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Confit refuses `CAST('' AS DOUBLE) / sqrt(NULL)` at build, while DuckDB serves NULL. Trapping constants under WHERE FALSE or empty input deliberately refuse at build. `fold` and the duplicate `eval_i32_literal` walker are possible cleanup targets.

## Wanted

Measure sibling-fold order and fix over-eager failing CAST evaluation. If a fallible fold replaces the duplicate walker, preserve deliberate zero-row boundaries.

## Involved

- `packages/confit/src/specializer/`: `fold`, `eval_i32_literal`.
- `packages/confit/tests/test_known_limitations.py`: constant refusal boundary.

## Out of scope

Changing deliberate zero-row admission without a maintainer answer.
