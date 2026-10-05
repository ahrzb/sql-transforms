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
