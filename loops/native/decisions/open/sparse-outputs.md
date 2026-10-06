# Should the Python step densify a sparse output?

**Question.** Two in-scope transformers return a scipy sparse matrix by
default:
- `OneHotEncoder()` (`sparse_output=True`);
- `KBinsDiscretizer()` (`encode="onehot"`).

`PythonTransform` cannot serve either. For each row it reads
`est.transform([row])[0]` and calls `float()` on each value. A row of a
sparse matrix is a 1-by-k sparse matrix, so the first call raises:

```
TypeError: float() argument must be a string or a real number, not 'csr_matrix'
```

Should the step densify a sparse output, as `.toarray()` does?

**What it decides.**
- Whether those two defaults serve at all.
- Whether they serve natively. The catalog already translates the dense
  forms (`OneHotEncoder(sparse_output=False)` since #347,
  `KBinsDiscretizer(encode="onehot-dense")` since #351). Once the step
  densifies, it would register the sparse forms with the same translation.
  The values are equal; only the container differs.

**Options.**
1. *Densify in the step* (`sql_transform._udf`):
   - when `transform` returns a sparse matrix, take `.toarray()` of the
     1-by-k row;
   - the declared width and lanes are unchanged;
   - the catalog then serves both defaults natively.
2. *Refuse at fit:* `PythonTransform` rejects a transformer whose output is
   sparse, naming `sparse_output=False` / `encode="onehot-dense"`. The
   author learns at fit time instead of at the first served row.
3. *Leave as is:* the step fails at its first call.

**Provisional choice.** None. The catalog declines both sparse forms with
`NotNative`. The step itself still fails, which is outside the catalog.

**What would close it.** An owner ruling between 1 and 2. Option 1 is the
loop's proposal: authors write `OneHotEncoder()` far more often than
`OneHotEncoder(sparse_output=False)`.

**Methodology (2026-10-06).** The note is
[research/2026-10-06/sparse.md](../research/2026-10-06/sparse.md), re-run by
an adversarial verifier. The environment is scikit-learn 1.9.0, scipy 1.18.0,
DuckDB 1.5.5; master 113fba7.

1. **The failure, reproduced under both of sklearn 1.9's sparse interfaces.**
   - Under `spmatrix`, the default, the first row raises this record's
     TypeError.
   - Under `sparray`, new in 1.9, it raises `produced k values, declared 1`.
2. **The fit is wrong before the first row is served.**
   - `_projection.py:352` probes the output with `np.asarray(sparse)`, which
     is a 0-d object array.
   - So the step is declared `struct<f0: double>`, and a field read such as
     `.c_red` fails at fit.
   - Patching only `__call__` still fails, with the width error.
3. **Inventory.** The verifier swept all 84 sklearn transformers on dense
   input.
   - Sparse by default, in scope: OneHotEncoder, KBinsDiscretizer,
     RandomTreesEmbedding, KNeighborsTransformer and
     RadiusNeighborsTransformer. The last two have no dense switch.
   - Sparse on request: SplineTransformer(`sparse_output`),
     MissingIndicator(`sparse=True`), FunctionTransformer, and compositions.
4. **Values, compared on raw bits.**
   - **OneHotEncoder:** 648 configurations, 129,600 rows, 0 differ from the
     dense configuration.
   - **KBinsDiscretizer:** 48 configurations, 11,856 rows, 0 differ.
   - These two are equal by construction: sklearn's dense path is
     `.toarray()` of the same CSR.
   - **SplineTransformer** is equal except at degree 0. With `constant`,
     values above the knots differ. With `periodic`, the sparse twin raises
     on every row.
   - **ColumnTransformer and FeatureUnion:** a sparse stack drops −0.0. That
     changed 10 of 189 and 25 of 126 rows, every one ±0.
5. **Peer converters** (sourced).
   - skl2onnx densifies OneHotEncoder, and refuses KBins `onehot`.
   - hummingbird densifies both.
   - sklearn's own ColumnTransformer densifies with `.toarray()`.
6. **Cost.** `.toarray()` takes 1–24 µs per row for widths from 10 to 10^5,
   at most 0.4% of the step.
7. **The fix, demonstrated** (patched outside the repo). Densify the fit
   probe and `__call__`. OneHotEncoder(), KBinsDiscretizer(`onehot`) and
   SplineTransformer(degree 3, sparse) then declare the same struct as their
   dense configurations. They serve bit-equal on DuckDB and confit, under
   both interfaces.
8. **Verifier corrections taken.**
   - A fourth site fails the same way: `model/_foreign.py:128`, reached
     through `Transform.from_estimator`. It raises IndexError.
   - Without the `encode.py` probe fix, the catalog refuses (NotNative) and
     does not mistranslate.
   - Spline needs a second carve-out: degree 0 with `periodic`.

**Recommendation.** Option 1, as one helper: `.toarray()` behind
`scipy.sparse.issparse`.

1. **Sites:**
   - `_udf.py:314`, the serve call;
   - `_projection.py:352`, the fit probe;
   - `model/_foreign.py:128`;
   - `native/encode.py:110`, needed only to make sparse OneHotEncoder native;
   - the width reads in `catalog_test.py` and `compose_test.py`.
2. **Catalog follow-ups:**
   - Drop the sparse guards for OneHotEncoder, KBinsDiscretizer and
     MissingIndicator(`sparse=True`). Their translation is exact by
     construction.
   - Serve sparse SplineTransformer, except at degree 0 with `constant` or
     `periodic`.
   - In a Pipeline, refuse a real-valued sparse step that is not last:
     Spline into Normalizer drifts up to 4 ulps. Allow 0/1 producers.
   - A FeatureUnion or ColumnTransformer with sparse output either refuses,
     or canonicalises −0.0 with `+ 0.0`, which works in both DuckDB and
     confit.
3. **Tests:**
   - Both sparse interfaces.
   - The width-1 case, `drop='if_binary'`.
   - Field reads at fit.
   - Assert that `check` compared every row (`compared == n`). A step that
     raises on every row otherwise compares nothing, and passes.

Why, judged against the goal that inference does not change:

- **Densifying keeps OneHotEncoder and KBins exactly identical.** It is the
  same `.toarray()` sklearn performs for the dense configuration.
- **Option 2 asks the author to fit a different model.** A densified
  Pipeline refits other `scale_` values: 3–65 ulps on these fixtures.
- **Option 2 cannot serve every case.** KNeighborsTransformer and
  RadiusNeighborsTransformer need an author-written wrapper.
- **The container depends on the caller.** sklearn picks it from the caller's
  thread-local config at serve time, so the same fitted model serves or
  fails depending on who calls it. The step should not depend on that.
- **The cost is under 0.4%** of the step.
