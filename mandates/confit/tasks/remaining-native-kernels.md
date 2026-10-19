# Resolve non-SQL scoring kernels

**Done when:** HistGradientBoosting, MLP and kNN/kernel SVM each have a measured non-SQL kernel boundary and a delegated implementation slice.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

Tree scoring work still names HistGradientBoosting*, MLP and kNN/kernel SVM as unbuilt. The native catalog owns family authoring; confit supplies Rust kernels only where SQL cannot express the work.

## Wanted

Coordinate the first required non-SQL kernel with the native catalog under the [parity bounds](../../../proposals/closed/native-transform-parity-bounds.md). Do not duplicate catalog family implementations.

## Involved

- `packages/sql-transform/sql_transform/native/`: family/kernel boundary.
- `packages/confit/`: required Rust kernel interface.

## Out of scope

Implementing all estimators in one session or assuming SQL cannot express them.
