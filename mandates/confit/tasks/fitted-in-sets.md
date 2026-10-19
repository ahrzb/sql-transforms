# Admit IN-subqueries as fitted sets

**Done when:** A fitted-set IN-subquery has fit/serve parity and NULL semantics under the existing contract.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

IN-subqueries as fitted sets remain authoring admission headroom.

## Wanted

Freeze the set at fit and preserve its measured IN semantics.

## Involved

- `packages/sql-transform/spec/README.md`: fitted authoring boundary.
- `packages/sql-transform/`: fitted artifact translation.

## Out of scope

Changing engine IN trap or batch semantics.
