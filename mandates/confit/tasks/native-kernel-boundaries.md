# Measure a vectorized multi-tree walk

**Done when:** A retained multi-tree reproduction preserves tree_span accumulation order and measures vectorized versus current costs.
**State:** Open. **Next:** Measure the smallest refused or divergent case with the optimizer-off oracle.

## Now

A fitted transformer historically costs about 118 µs/row versus 1.4 µs without it (bench_transforms.py, 2026-09-26); most cost is sklearn transform(). The route is SQL-defined transforms, Rust kernels only where SQL cannot express the operation. Unbuilt: HistGradientBoosting*, MLP, vectorized multi-tree walk preserving tree_span accumulation order, kNN/kernel SVM.

## Wanted

Resolve the vectorized multi-tree implementation with the native catalog. Preserve the agreed parity bounds.

## Involved

- `packages/sql-transform/sql_transform/native/`: SQL/native family boundary and `tree_span`.
- `packages/confit/bench_transforms.py`: transformer serving comparison.

## Out of scope

HistGradientBoosting, MLP, kNN and kernel SVM, which need separate slices; duplicating catalog ownership.
