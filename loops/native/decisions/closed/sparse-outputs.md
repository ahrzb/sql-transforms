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
scikit-learn (sklearn) version 1.9.0, scipy version 1.18.0 and DuckDB version
1.5.5. The Python library scipy defines the sparse matrix types. The
repository was on master at commit 113fba7.

The file paths below are under `packages/sql-transform/sql_transform`. A
number after a colon is a line number in that file.

1. **The note reproduces the failure under both sparse interfaces of sklearn
   1.9.** A sparse interface is the scipy type that sklearn uses for a sparse
   output.
   - `spmatrix` is the scipy sparse matrix. It is the default. Under
     `spmatrix`, the step raises the TypeError above at the first row.
   - `sparray` is the scipy sparse array. It is new in sklearn 1.9. Under
     `sparray`, the step raises the width error `produced k values, declared 1`
     at the first row.
   - The width error says that the output row has k values, but fit declared
     1 output field.
2. **Fit declares the wrong output before the step serves a row.**
   - The fit probe (`_projection.py:352`) is the code in fit that finds the
     width of the step. The width is the number of output fields. The probe
     transforms one row of the training data and counts the values in the
     output.
   - The probe converts the output with `np.asarray(sparse)`. For a sparse
     output, this gives a 0-dimensional object array, which holds the sparse
     matrix as one value. So the probe finds a width of 1.
   - Fit then declares the step as `struct<f0: double>`, a struct with one
     field.
   - A field read selects one output field of the struct by its name, for
     example `.c_red`. Such a read then fails at fit.
   - The serve call (`PythonTransform.__call__`) is the code that serves each
     row. If a patch densifies only the serve call, the step still fails with
     the width error.
3. **Inventory.** The verifier tested all 84 sklearn transformers on dense
   input.
   - These in-scope transformers give a sparse output by default:
     OneHotEncoder, KBinsDiscretizer, RandomTreesEmbedding,
     KNeighborsTransformer and RadiusNeighborsTransformer.
   - KNeighborsTransformer and RadiusNeighborsTransformer have no parameter
     that makes the output dense.
   - These transformers give a sparse output only on request:
     SplineTransformer with the parameter `sparse_output`, MissingIndicator
     with the parameter `sparse=True`, FunctionTransformer, and compositions.
     A composition is a transformer that combines other transformers, for
     example a Pipeline.
4. **The note compared the values on raw bits.** Each comparison is between
   a sparse configuration and its dense configuration.
   - **OneHotEncoder:** 0 of 129,600 rows differ from the dense
     configuration. The rows cover 648 configurations.
   - **KBinsDiscretizer:** 0 of 11,856 rows differ. The rows cover 48
     configurations.
   - These two are bit-exact by construction. For the dense configuration,
     sklearn calls `.toarray()` on the same compressed sparse row (CSR)
     matrix. CSR is a scipy format that stores a matrix row by row and leaves
     out the zeros. `.toarray()` is the scipy method that converts a sparse
     matrix into a dense array.
   - **SplineTransformer** gives equal values, except at degree 0. The degree
     is the degree of the polynomial pieces of the spline. The knots are the
     points where the pieces join. The `extrapolation` parameter sets the
     output for an input outside the knots. The value `constant` uses the
     value of the spline at the nearest end knot. The value `periodic` repeats
     the spline with a period of the range of the knots.
   - At degree 0 with the `constant` extrapolation, the outputs for an input
     above the knots differ. At degree 0 with the `periodic` extrapolation,
     the twin of the sparse configuration raises an error on every row.
   - **ColumnTransformer and FeatureUnion** stack the outputs of their parts.
     If the stack is sparse, it drops each −0.0. The value then becomes
     +0.0. This changed 10 of 189 rows for ColumnTransformer and 25 of 126
     rows for FeatureUnion.
   - In each changed row, the difference was a −0.0 against a +0.0. A
     bit-exact comparison counts −0.0 and +0.0 as different values.
5. **Peer converters.** A peer converter is another tool that translates a
   fitted sklearn model into a different form. The note read their source
   code.
   - skl2onnx translates a sklearn model into the Open Neural Network Exchange
     (ONNX) format. It densifies OneHotEncoder. It raises an error for
     KBinsDiscretizer with `encode="onehot"`.
   - hummingbird translates a sklearn model into tensor computations. It
     densifies both.
   - The ColumnTransformer of sklearn densifies with `.toarray()`.
6. **Cost.** For widths from 10 to 10^5 output fields, `.toarray()` takes
   1–24 µs per row. This is at most 0.4% of the time of the step.
7. **The note demonstrates the fix** with a patch outside the repository.
   The patch densifies the output at the fit probe and at the serve call.
   With the patch, three transformers declare the same struct as their dense
   configurations: OneHotEncoder(), KBinsDiscretizer(`onehot`) and
   SplineTransformer(degree 3, sparse). Under both interfaces, they serve
   values that are bit-exact with their dense configurations, on DuckDB and
   on confit.
8. **The record takes these corrections from the verifier.**
   - A fourth site fails the same way. The function
     `Transform.from_estimator` wraps a sklearn transformer. Its code at
     `model/_foreign.py:128` calls the fitted transformer.
   - This site raises `IndexError`, the Python error for an index that is
     not valid.
   - The OneHotEncoder entry has a probe (`native/encode.py`). The probe
     calls the fitted sklearn transformer for each category. If this probe
     does not densify, the entry raises `NotNative`. The entry does not give a
     wrong translation.
   - SplineTransformer needs a second case that the entry does not serve.
     This case is degree 0 with the `periodic` extrapolation.

**Recommendation.** Use option 1, with one helper function. If the scipy
function `scipy.sparse.issparse` finds that an output is sparse, the helper
calls `.toarray()` on it.

1. **Use the helper at these sites:**
   - the serve call (`_udf.py:314`)
   - the fit probe (`_projection.py:352`)
   - the code for `Transform.from_estimator` (`model/_foreign.py:128`)
   - the probe of the OneHotEncoder entry (`native/encode.py:110`). This
     site is necessary only to make a sparse OneHotEncoder native.
   - the code that reads the width in the tests of the catalog
     (`catalog_test.py`) and of the compositions (`compose_test.py`)
2. **Then make these changes in the catalog:**
   - Remove the conditions that make these entries raise `NotNative` for a
     sparse output: OneHotEncoder, KBinsDiscretizer and
     MissingIndicator(`sparse=True`). Their translation is exact by
     construction.
   - Serve a sparse SplineTransformer, except at degree 0 with the
     `constant` or the `periodic` extrapolation.
   - In a Pipeline, if a real-valued sparse step is not the last step, raise
     `NotNative`. The reason is that a sparse SplineTransformer step into a
     Normalizer step drifts by up to 4 ulps (units in the last place). If a
     sparse step gives only the values 0 and 1, do not raise `NotNative` for
     it.
   - If a FeatureUnion or a ColumnTransformer has a sparse output, make the
     entry do one of these two things:
     - raise `NotNative`
     - change −0.0 to +0.0 with `+ 0.0`, which works in both DuckDB and
       confit
3. **Add these tests:**
   - Test both sparse interfaces, `spmatrix` and `sparray`.
   - Test the width-1 case with OneHotEncoder(`drop='if_binary'`). This case
     has one output field.
   - Test field reads at fit.
   - Assert that `native.check` compared every row (`compared == n`).
     `native.check` is the test that serves the same query with the twin and
     with the entry. Then it compares the results. If the twin raises on every
     row, `native.check` compares no rows. Without the assertion, the test
     then passes.

These are the reasons for the recommendation. Each reason judges the options
against the goal that inference does not change.

- **Densifying keeps OneHotEncoder and KBinsDiscretizer bit-exact with the
  dense configuration.** Densifying is the same `.toarray()` call that
  sklearn makes for the dense configuration.
- **Option 2 asks the author to fit a different model.** If the author
  changes a Pipeline to the dense configuration, fit computes other `scale_`
  values for the Pipeline. `scale_` holds the factor for each feature that a
  sklearn scaler learns in fit. On the test data of the note, the outputs
  differ by 3–65 ulps.
- **Option 2 cannot serve every case.** KNeighborsTransformer and
  RadiusNeighborsTransformer need a wrapper that the author writes.
- **The container depends on the caller.** At serve time, sklearn picks the
  container (`spmatrix` or `sparray`) from the sklearn configuration of the
  calling thread. So the same fitted model serves or fails, depending on who
  calls it. The step should not depend on that.
- **The cost is under 0.4%** of the time of the step.

**Ruling (owner, 2026-10-06).** The owner approved option 1. The Python step
densifies a sparse output with one helper, at the four sites above. For
OneHotEncoder and KBinsDiscretizer, the densified values are bit-exact
against their dense configurations. So their entries serve by default.
