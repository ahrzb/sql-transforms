# Compute IN ANY and ALL scans

**Done when:** Non-key scan IN, ANY, SOME and ALL match NULL, guard and comparison semantics.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Grouped IN counts hit/rows/values. IN evaluates x only after a non-NULL item survives; #475 corrected #474’s count(*) guard. Other ANY comparisons evaluate x after any surviving row, including NULL. Keyed trapping x is batch-dependent and refuses (149 IN and 152 quantified refusals in recorded campaigns).

## Wanted

Use `Guard::InItem` for IN and row guards for other comparisons. ALL is NOT over opposite ANY. ANY uses `dec_common`, not scalar `cmp_common`; refuse trapping item casts. Preserve -0.0/NaN tie semantics.

## Involved

- `packages/confit/src/specializer/frontend/subquery.rs`: `in_subquery`, `quantified_subquery`, `quantified_residual`, `guarded_key`, `trap_keys`.
- `packages/confit/src/specializer/plan.rs`: `Agg::ties`.
- `packages/confit/tests/test_scalar_subquery.py`: quantified scans.

## Out of scope

Serving batch-dependent keyed trapping x or new shape=many forms.
