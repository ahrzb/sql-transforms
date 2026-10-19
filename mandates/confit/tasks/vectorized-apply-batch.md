# Vectorize UDF apply_batch

**Done when:** infer_arrow uses vectorized apply_batch with parity for values, NULLs and errors.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

infer_arrow calls UDFs per row. A vectorized apply_batch remains open.

## Wanted

Batch UDF calls without changing error visibility or row correspondence.

## Involved

- `packages/confit/`: `infer_arrow`, UDF `apply_batch` interface.

## Out of scope

Changing SQLProjection’s row-local serving contract.
