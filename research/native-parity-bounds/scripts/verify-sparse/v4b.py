import warnings; warnings.simplefilter("ignore")
import numpy as np, duckdb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
rng = np.random.default_rng(0)
X = np.column_stack([rng.choice([0.0, 1.0, 2.0], 60), rng.integers(0, 12, 60).astype(float)])
parts = lambda: [("neg", FunctionTransformer(np.negative), [0]), ("ohe", OneHotEncoder(), [1])]
cs = ColumnTransformer(parts(), sparse_threshold=0.3).fit(X)
cd = ColumnTransformer(parts(), sparse_threshold=0.0).fit(X)
print("default-threshold CT sparse_output_:", cs.sparse_output_)
A = cs.transform(X).toarray(); B = cd.transform(X)
d = A.view(np.uint64) != B.view(np.uint64)
print("CT differing lanes:", d.sum(), "rows:", d.any(1).sum(), "/", len(X), "A:", set(A[d]), "B signbit:", set(np.signbit(B[d])))
con = duckdb.connect(); con.execute("CREATE TABLE t AS SELECT -(0.0::DOUBLE) AS x")
for q in ("EXPLAIN SELECT x + 0.0 AS y FROM t", "EXPLAIN SELECT (x + 0.0) + 0.0 AS y FROM t"):
    txt = con.execute(q).fetchall()[0][1]
    print([l for l in txt.splitlines() if "x" in l and "Table" not in l and "Projections" not in l][:6])
con.execute("PRAGMA explain_output='all'")
print(con.execute("EXPLAIN SELECT x + 0.0 AS y FROM t").fetchall()[1][1][:800] if len(con.execute("EXPLAIN SELECT x + 0.0 AS y FROM t").fetchall())>1 else "")
