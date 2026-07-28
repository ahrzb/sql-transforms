"""SQLTransform — SQL feature transforms, fitted once and served by Confit.

Author the transform as SQL, `fit()` it against training data to freeze the
window-aggregate state into static tables, then serve it row-at-a-time:

    t = SQLTransform("SELECT (age - avg(age) OVER ()) AS age_c FROM __THIS__")
    t.fit(train_table)
    t.infer({"age": 40})

`fit()` hands the rewritten SQL and the frozen tables to Confit
(`packages/confit`), which partially evaluates the pair into a native
function. Confit's contract carries through: the fitted SQL either serves
bit-exact with DuckDB, or `fit()` raises and names the construct it will not
serve.
"""

from __future__ import annotations

from sql_transform._transform import SQLTransform

__all__ = ["SQLTransform"]
