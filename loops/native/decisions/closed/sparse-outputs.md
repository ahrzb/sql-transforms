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

**Methodology (2026-10-06).** The evidence is in the research note
[research/2026-10-06/sparse.md](../research/2026-10-06/sparse.md). An
adversarial verifier repeated the work of the note. The environment was
scikit-learn (sklearn) 1.9.0, scipy 1.18.0 and DuckDB 1.5.5, on master at
commit 113fba7.

1. **The note reproduces the failure under both sparse interfaces of sklearn
   1.9.**
   - Under `spmatrix` (the scipy sparse matrix), which is the default, the
     first row raises the TypeError above.
   - Under `sparray` (the scipy sparse array), which is new in 1.9, the first
     row raises the width error `produced k values, declared 1`.
2. **Fit declares the wrong output before the step serves a row.**
   - The fit probe (`_projection.py:352`) transforms one row to find the
     number of output fields (the width). It converts the output with
     `np.asarray(sparse)`, which gives a 0-dimensional object array.
   - So fit declares the step as `struct<f0: double>`, a struct with one
     field. A field read such as `.c_red` then fails at fit.
   - If a patch densifies only the serve call (`PythonTransform.__call__`),
     the step still fails with the width error.
3. **Inventory.** The verifier tested all 84 sklearn transformers on dense
   input.
   - These in-scope transformers give a sparse output by default:
     OneHotEncoder, KBinsDiscretizer, RandomTreesEmbedding,
     KNeighborsTransformer and RadiusNeighborsTransformer. The last two have
     no parameter that makes the output dense.
   - These transformers give a sparse output only on request:
     SplineTransformer(`sparse_output`), MissingIndicator(`sparse=True`),
     FunctionTransformer and compositions.
4. **The note compared the values on raw bits.**
   - **OneHotEncoder:** 0 of 129,600 rows over 648 configurations differ
     from the dense configuration.
   - **KBinsDiscretizer:** 0 of 11,856 rows over 48 configurations differ.
   - These two are bit-exact by construction. The dense path of sklearn is
     `.toarray()` of the same compressed sparse row (CSR) matrix.
   - **SplineTransformer** gives equal values, except at degree 0. With the
     `constant` extrapolation, the values above the knots differ. With the
     `periodic` extrapolation, the twin of the sparse configuration raises
     on every row.
   - **ColumnTransformer and FeatureUnion:** if they stack their parts into
     a sparse output, the stack drops −0.0. This changed 10 of 189 rows for
     ColumnTransformer and 25 of 126 rows for FeatureUnion. In each changed
     row, the difference was −0.0 against +0.0.
5. **Peer converters** (read in their source code).
   - The converter skl2onnx densifies OneHotEncoder. It raises an error for
     KBinsDiscretizer with `encode="onehot"`.
   - The converter hummingbird densifies both.
   - The ColumnTransformer of sklearn densifies with `.toarray()`.
6. **Cost.** For widths from 10 to 10^5 output fields, `.toarray()` takes
   1–24 µs per row. This is at most 0.4% of the time of the step.
7. **The note demonstrates the fix** with a patch outside the repository.
   The patch densifies the output at the fit probe and at the serve call.
   Then OneHotEncoder(), KBinsDiscretizer(`onehot`) and
   SplineTransformer(degree 3, sparse) declare the same struct as their
   dense configurations. Under both interfaces, they serve values that are
   bit-exact with their dense configurations, on DuckDB and on confit.
8. **The record takes these corrections from the verifier.**
   - A fourth site fails the same way. The site is `model/_foreign.py:128`,
     the estimator wrapper that `Transform.from_estimator` uses. It raises
     `IndexError`.
   - The OneHotEncoder entry has a probe (`native/encode.py`) that calls the
     twin for each category. If this probe does not densify, the entry
     raises `NotNative`. It does not give a wrong translation.
   - SplineTransformer needs a second case that the entry does not serve:
     degree 0 with the `periodic` extrapolation.

**Recommendation.** Use option 1, with one helper function. If
`scipy.sparse.issparse` is true for an output, the helper calls `.toarray()`
on it.

1. **Use the helper at these sites:**
   - the serve call (`_udf.py:314`)
   - the fit probe (`_projection.py:352`)
   - the estimator wrapper (`model/_foreign.py:128`)
   - the probe of the OneHotEncoder entry (`native/encode.py:110`). This
     site is necessary only to make a sparse OneHotEncoder native.
   - the code that reads the width in the catalog tests (`catalog_test.py`)
     and the composition tests (`compose_test.py`)
2. **Then make these changes in the catalog:**
   - Remove the conditions that make these entries raise `NotNative` for a
     sparse output: OneHotEncoder, KBinsDiscretizer and
     MissingIndicator(`sparse=True`). Their translation is exact by
     construction.
   - Serve a sparse SplineTransformer, except at degree 0 with the
     `constant` or the `periodic` extrapolation.
   - In a Pipeline, raise `NotNative` for a real-valued sparse step that is
     not the last step. The reason is that SplineTransformer into Normalizer
     drifts by up to 4 ulps (units in the last place). Allow a sparse step
     that gives only the values 0 and 1.
   - If a FeatureUnion or a ColumnTransformer has a sparse output, do one of
     two things in the entry:
     - raise `NotNative`
     - change −0.0 to +0.0 with `+ 0.0`, which works in both DuckDB and
       confit
3. **Add these tests:**
   - Test both sparse interfaces, `spmatrix` and `sparray`.
   - Test the width-1 case, which has one output field, with
     OneHotEncoder(`drop='if_binary'`).
   - Test field reads at fit.
   - Assert that `native.check` compared every row (`compared == n`).
     `native.check` is the test that serves the same query with the twin and
     with the entry, and compares the results. If the twin raises on every
     row and the test does not assert this, `native.check` compares no rows
     and passes.

These are the reasons for the recommendation. Each reason judges the options
against the goal that inference does not change.

- **Densifying keeps OneHotEncoder and KBinsDiscretizer bit-exact.**
  Densifying is the same `.toarray()` that sklearn does for the dense
  configuration.
- **Option 2 asks the author to fit a different model.** If the author
  changes a Pipeline to the dense configuration, the Pipeline fits other
  `scale_` values. On the test data of the note, the outputs differ by 3–65
  ulps.
- **Option 2 cannot serve every case.** KNeighborsTransformer and
  RadiusNeighborsTransformer need a wrapper that the author writes.
- **The container depends on the caller.** At serve time, sklearn picks the
  container (`spmatrix` or `sparray`) from the configuration of the calling
  thread. So the same fitted model serves or fails, depending on who calls
  it. The step should not depend on that.
- **The cost is under 0.4%** of the time of the step.
