# Parse Spark SQL

**Done when:** parse(sql, dialect=SPARK) has dialect-specific round-trip and named-refusal tests.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Reverse dialect frontends are unbuilt.

## Wanted

Implement the Spark parse seam with a bounded tested grammar.

## Involved

- `packages/sql-transform/`: `parse(sql, dialect=SPARK|BIGQUERY)`.

## Out of scope

BigQuery reverse parsing and execution; accepting syntax without semantic translation.
