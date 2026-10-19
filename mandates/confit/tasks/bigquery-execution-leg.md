# Execute BigQuery L3 checks

**Done when:** The L3 check uploads a table, executes printed BigQuery SQL and compares rows_of() through the existing seam.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

The Spark execution leg exists; BigQuery L3 does not.

## Wanted

Use load_table_from_dataframe, execute the printed SQL and compare rows_of() without replacing the dialect contract.

## Involved

- `packages/sql-transform/`: Spark L3 execution seam, `rows_of()`, BigQuery client `load_table_from_dataframe`.

## Out of scope

Reverse parsing or acquiring human-only cloud credentials.
