# Evaluate uncorrelated static terms at build

**Done when:** Nontrapping static WHERE terms that currently over-refuse serve; actual traps refuse by name.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

A trapping static WHERE term with no outer column refuses. DuckDB runs the uncorrelated subquery once per batch, even an empty batch. With an outer column, WHERE evaluates distinct outer values × static rows in written order, except BETWEEN upper bounds move last.

## Wanted

Evaluate static terms at build in measured DuckDB order. Refuse only when the evaluation traps.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: `scan_value`, `Step`, `build_part`.
- `packages/confit/src/duckdb/mod.rs`: build filtering.
- `packages/confit/tests/test_scalar_subquery.py`: empty-batch traps.

## Out of scope

Changing the oracle’s once-per-batch behaviour.
