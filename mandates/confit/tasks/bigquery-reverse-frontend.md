# Parse BigQuery SQL

**Done when:** parse(sql, dialect=BIGQUERY) has dialect-specific round-trip and refusal tests.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

The BigQuery reverse frontend is unbuilt.

## Wanted

Implement its bounded tested grammar through the shared parse seam.

## Involved

- `packages/sql-transform/`: `parse(sql, dialect=BIGQUERY)`.

## Out of scope

BigQuery L3 execution transport.
