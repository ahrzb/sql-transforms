# Reduce unnecessary CASE branches

**Done when:** One measured nontrapping CASE family avoids branches with lower build cost and no serving regression.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

select_case uses selects only for small nontrapping CASE or NULL guards. More than eight operations or a cast still branches; VARCHAR casts would cost every row. #422’s case_forest.py built 100 depth-5 trees in 0.9–1.1 s versus 2.5 s before selects.

## Wanted

Measure larger nontrapping cases and numeric casts before extending selection. Keep trapping branches lazy.

## Involved

- `packages/confit/src/specializer/lower.rs`: `select_case`.
- `packages/confit/src/specializer/exec/cranelift.rs`: branch/select code.

## Out of scope

Deep CASE scaling covered by deep-case-builds; speculative VARCHAR conversion.
