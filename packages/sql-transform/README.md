# sql-transform

SQL feature transforms, fitted once and served by [Confit](../confit).

```python
from sql_transform import SQLTransform

t = SQLTransform("SELECT (age - avg(age) OVER ()) AS age_c FROM __THIS__")
t.fit(train_table)
t.infer({"age": 40})
```

`fit()` runs the SQL over training data to freeze each window aggregate into a
static table, rewrites the SQL into plain-column form, and hands the pair to
Confit, which partially evaluates them into a native serving function.

**This package was reset.** The DataFusion-differentiated native engine, the
Python codegen backend, and the batch `transform()` path have been removed; what
remains is the fit-and-serve path built on Confit. Composing one transform into
another (`t"... {other}(col) ..."`) is kept as a surface but refuses by name
until it is rebuilt.
