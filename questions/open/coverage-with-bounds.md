# Should opt-in bounded transformers count as native coverage?

**Asked by:** native catalog, 2026-10-17. **Holds:** changing the `coverage_native` key, or adding coverage keys to the reporting contract.
**Against:** `40f6ab17`.

## Context

Should `coverage_native` count a transformer served only with `allow_bound=True` the same as a bit-exact transformer?
`coverage_native` is the KPI key for the number of native sklearn transformers.

**Interim:** Keep the current count. Do not change the tracker keys while this record waits.
The coverage table continues to name opt-in bounds.

## Read first

- [Native catalog brief](../../mandates/native/brief.md)
- [The native loop review](../../research/native-loop-review/README.md)

## Options

### Option 1

Keep one count, with bounds shown in the generated table and the existing bound KPI.

### Option 2

Count only transformers with a served bit-exact configuration in `coverage_native`; add a separate bounded-only count.

### Option 3

Keep the old key for continuity and add exact, bounded-only and mixed counts.

## Recommendation

Option 3: it keeps the maintainer's tracker comparable over time and shows the bounded share. It would change if the tracker can be changed with the keys.

## Evidence

At master `40f6ab17`, the generated
[coverage table](../../packages/sql-transform/spec/native/coverage.md) counts 51 of 68 in-scope sklearn transformers as native.
The catalog contract defines that set: fitted transformers over rows of columns, not text documents or target labels.
Of the 51, 19 serve only when the caller permits a nonzero bound.
Three more have both bit-exact and bounded configurations.
These are FunctionTransformer, AdditiveChi2Sampler and SparseRandomProjection.
Each can serve bit-exactly for some constructor settings or fitted states.
PR #414 implements the explicit opt-in. Later family PRs, #458 through #513, leave the coverage key unchanged.

What would close it: The maintainer chooses the counting rule and any new KPI keys.

## Answer

