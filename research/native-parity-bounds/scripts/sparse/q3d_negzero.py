"""Q3d: scipy drops -0.0 when a dense block enters a sparse stack (so the
ColumnTransformer/FeatureUnion sparse twin maps -0.0 -> +0.0), and whether
`x + 0.0` reproduces that exactly in DuckDB (not constant-folded away)."""
import math
import duckdb
import numpy as np
import scipy.sparse as sp

d = np.array([[-0.0, 0.0, 1.0, np.nan]])
c = sp.csr_matrix(d)
print("csr_matrix(dense) nnz:", c.nnz, "data:", c.data.tolist(), "-> toarray signbits:", np.signbit(c.toarray()).tolist())
h = sp.hstack([sp.csr_matrix(np.array([[1.0]])), d]).tocsr()
print("sparse.hstack([sparse, dense]) toarray:", h.toarray().tolist(), "signbit:", np.signbit(h.toarray()).tolist())
con = duckdb.connect()
for v in (-0.0, 0.0, 2.5, float("nan"), -5e-324):
    (r,) = con.execute("SELECT x + 0.0 FROM (SELECT ?::DOUBLE AS x)", [v]).fetchone()
    print(f"DuckDB {v!r} + 0.0 = {r!r} signbit={math.copysign(1, r) < 0 if r == r else 'nan'}")
(r,) = con.execute("SELECT x + 0.0 FROM (VALUES (-0.0::DOUBLE)) t(x)").fetchone()
print("DuckDB VALUES(-0.0)+0.0 ->", repr(r))
print(con.execute("EXPLAIN SELECT x + 0.0 FROM (VALUES (-0.0::DOUBLE)) t(x)").fetchall()[0][1][:400])
