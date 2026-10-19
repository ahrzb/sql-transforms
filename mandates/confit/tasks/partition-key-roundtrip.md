# Round-trip remaining partition keys

**Done when:** BIT, TIMETZ and UNION parameter-table partition keys pass the retained round-trip cases.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Those types cannot round-trip through the params table; _marginal_test.py has xfails.

## Wanted

Preserve key type, value and grouping identity through fit and marginalization.

## Involved

- `packages/sql-transform/sql_transform/_marginal_test.py`: partition-key xfails.
- `packages/sql-transform/`: params-table serialization and marginalization.

## Out of scope

A general temporal expression lane.
