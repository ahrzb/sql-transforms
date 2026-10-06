import warnings; warnings.simplefilter("ignore")
import numpy as np, scipy.sparse as sp, duckdb
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer, StandardScaler, MaxAbsScaler
# toarray of stored -0.0 for each format
for fmt in ("csr", "csc", "coo", "lil", "dok", "bsr", "dia"):
    for cls in ("matrix", "array"):
        m = getattr(sp, f"{fmt}_{cls}")
        a = sp.coo_array((np.array([-0.0, 2.0]), (np.array([0, 0]), np.array([0, 2]))), shape=(1, 3))
        x = m(a)
        try:
            d = x.toarray()
            print(fmt, cls, "stored", getattr(x, "nnz", "?"), "toarray signbit[0]", np.signbit(d[0, 0]))
        except Exception as e:
            print(fmt, cls, type(e).__name__)
v = sp.coo_array((np.array([-0.0]), (np.array([1]),)), shape=(3,))
print("1-D coo toarray signbit", np.signbit(v.toarray()[1]), "nnz", v.nnz)
# does sparse.hstack/csr from dense store -0.0?
print("csr_array(dense [-0.0, 1.0]).nnz:", sp.csr_array(np.array([[-0.0, 1.0]])).nnz)
# ColumnTransformer with a dense part emitting -0.0
X = np.array([[-0.0, 0.0, 1.0], [2.0, 1.0, 0.0], [-0.0, 1.0, 1.0], [3.0, 0.0, 0.0]] * 5, dtype=float)
parts = lambda: [("neg", FunctionTransformer(np.negative), [0]), ("ohe", OneHotEncoder(), [1, 2])]
cs = ColumnTransformer(parts(), sparse_threshold=1.0).fit(X)
cd = ColumnTransformer(parts(), sparse_threshold=0.0).fit(X)
print("CT sparse_output_:", cs.sparse_output_, cd.sparse_output_)
A = cs.transform(X).toarray(); B = cd.transform(X)
diff = (A.view(np.uint64) != B.view(np.uint64))
print("CT differing lanes", diff.sum(), "values A", A[diff][:4], "B", B[diff][:4], "B signbit", np.signbit(B[diff][:4]))
# default threshold 0.3 with OHE-only: sparse?
c03 = ColumnTransformer([("ohe", OneHotEncoder(), [1, 2])]).fit(X)
print("CT default threshold, OHE only, sparse_output_:", c03.sparse_output_)
fu_s = FeatureUnion([("neg", FunctionTransformer(np.negative)), ("ohe", OneHotEncoder())]).fit(X)
fu_d = FeatureUnion([("neg", FunctionTransformer(np.negative)), ("ohe", OneHotEncoder(sparse_output=False))]).fit(X)
A = fu_s.transform(X); print("FU sparse?", sp.issparse(A)); A = A.toarray(); B = fu_d.transform(X)
diff = (A.view(np.uint64) != B.view(np.uint64)); print("FU differing lanes", diff.sum(), "all zeros:", np.all(A[diff] == 0) and np.all(B[diff] == 0))
con = duckdb.connect()
print(con.execute("SELECT signbit(x), signbit(x + 0.0), signbit(0.0 + x), x + 0.0 FROM (VALUES (-0.0::DOUBLE), ('-0.0'::DOUBLE), (-(0.0::DOUBLE))) t(x)").fetchall())
con.execute("CREATE TABLE t AS SELECT -(0.0::DOUBLE) AS x UNION ALL SELECT 1.5")
print(con.execute("EXPLAIN SELECT x + 0.0 AS y FROM t").fetchall()[0][1][-400:])
print(con.execute("SELECT signbit(x + 0.0), signbit(x) FROM t").fetchall())
print(con.execute("SELECT signbit(CASE WHEN x = 0 THEN -0.0::DOUBLE ELSE x END + 0.0) FROM t").fetchall())
