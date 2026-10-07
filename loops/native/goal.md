# The native catalog: goal

`to_native(step)` turns a fitted `PythonTransform` into a confit
`SqlFunction`. The function serves the same calls, with no Python in the
work for each row. Through sklearn, a fitted transformer costs about 118 µs
for each row. The same query without the transformer costs about 1.4 µs
(`benchmarks/bench_transforms.py`, 2026-09-26). The native catalog removes
that cost for each transformer that it covers.

## What done means

The loop is done when every transformer in scope is native in the
[coverage table](../../packages/sql-transform/spec/native/coverage.md).
The [native catalog contract](../../packages/sql-transform/spec/native/catalog-contract.md)
judges each entry: what it must equal, where the twin raises, and the scope.
