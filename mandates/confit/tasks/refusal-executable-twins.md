# Add executable refusal twins

**Done when:** Known-limitation tests cover every listed missing refusal family.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Missing twins: HAVING, OFFSET/FETCH/TOP, INTERSECT/EXCEPT, subqueries, multiple statements, table functions, QUALIFY and the 2 GiB Arrow batch ceiling.

## Wanted

Add minimal executable witnesses that test the current named boundary, including existing supported OFFSET forms rather than treating all OFFSET as refused.

## Involved

- `packages/confit/tests/test_known_limitations.py`: refusal witnesses.

## Out of scope

Changing admission rules or allocating 2 GiB unnecessarily in a test.
