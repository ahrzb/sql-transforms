# Complete bounded compositions

**Done when:** a bounded step serves in every composition position that [bounded steps in compositions](../../../proposals/closed/bounded-steps-in-compositions.md) allows: each later continuous step carries the bound, a comparison map `g` (such as a distance's square) passes through a composition, and after a discontinuous step `native.check` reports the flip rate in place of a bound; `native.check` proves each derived bound on the catalog fixtures.
**State:** option 1 (a last step, or a part) and the affine part of option 2 are served (#495, #514). Each step after a bounded one gives its output fields a carried bound (`_registry.Entry.carry`, `_carry.py`): the affine scalers, the linear projections, `FeatureAgglomeration`'s mean, the selectors, `FunctionTransformer` of None, `np.negative`, `np.abs` and similar, and a part's weight. Any other step after a bounded one refuses by name. **Next:** carries for the continuous steps that are not affine: `Normalizer`, `PowerTransformer`, the kernel maps, and the functions with a bound of their own.

## Now

A non-affine step after a bounded one refuses by name. `to_native` also refuses a composition where computing each unshared value at each read adds more than 500,000 characters to its definition (`_registry.MAX_UNSHARED`), because confit shares no value above a guarded `ln`, `log2`, `log10`, `sqrt`, `sin`, `cos` or `tan` (the shared-logarithm need in the [confit tracker](../../confit/tracker.md#needs)). `check` skips definitions past 4 million characters, which DuckDB cannot read in memory (the definition-bound-once need).

On rows near the largest double, a sum can overflow in one order of addition and not another; `check` stays strict there until [where a sum overflows](../../../questions/open/where-a-sum-overflows.md) is answered.

## Wanted

Carries for the remaining continuous steps, `g` through a composition, and the flip rate after a discontinuous step.

## Involved

- `packages/sql-transform/sql_transform/native/_carry.py`
- `packages/sql-transform/sql_transform/native/compose.py`: `_pipeline_bounds`, `_parts_bounds`
- `packages/sql-transform/sql_transform/native/_registry.py`: `Entry.carry`, `ErrorScale`, `MAX_UNSHARED`
- `packages/sql-transform/sql_transform/native/_check.py`: `check`

## Out of scope

Changing any registered bound; rows where a sum overflows on one side only (held by the question).
