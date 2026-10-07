# Changelog

Changes to the public interface of `sql_transform`, newest first.

## 2026-10-06: one compositional model

[PR #431](https://github.com/ahrzb/sql-transforms/pull/431) replaced the two
authoring implementations with one model. It removed these interfaces:

| Removed | Use instead |
|---|---|
| Automatic marginalization in the `SQLProjection` constructor | `SQLProjection.marginalize`, or explicit `__FIT__` and `__THIS__` SQL |
| The `this_schema` constructor argument | Explicit columns and aliases |
| Hidden `_` aliases | `_` input names are ordinary columns |
| Automatic `unnest(transformer(...))` field expansion | Select the named fields, or keep the output struct |
| `Transform.from_estimator` | A row-local estimator captured in `SQLProjection`, or explicit relation callbacks in `SQLTransform` |
| Conversion of numeric-looking strings at estimator fit | Strings stay strings; write a SQL `CAST` |
| Live non-Arrow relations captured by a projection | Captures convert to Arrow once; refit when a source changes |
| `fit` returning the estimator | `Fitted` or `FittedProjection`; `SQLTransform` also keeps its fitted state |
| `serving_sql`, `infer` and `infer_batch` | The fitted `sql`, then `compile().infer_rows` or `compile().infer_arrow` |
| Estimator `backend`, `boundary` and output-schema metadata | Confit's metadata and the fitted projection's `schema` |
| `from_file` | Read the SQL file before you call the constructor |
| Public ordinal SQL and learned-only projection params | The unordered serving `sql` and the complete `params` mapping |
| `marginalize`, `Marginalized`, `MarginalizeError`, `FitStep`, `ParamsSpec` and `UDFSpec` | `SQLProjection` methods, fitted artifacts and `TransformError` subclasses |
| Imports from `sql_transform.model` | Imports from `sql_transform` |
