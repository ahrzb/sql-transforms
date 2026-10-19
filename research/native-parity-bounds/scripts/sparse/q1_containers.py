"""Q1: what transform([row]) returns for the sparse defaults, and what the
step's own code paths (fit probe np.asarray, [0], _flatten_row, float) do."""
import warnings
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer, SplineTransformer
from sklearn.impute import MissingIndicator

rng = np.random.default_rng(0)
X = rng.normal(size=(50, 2))
Xc = np.array([["a", "x"], ["b", "y"], ["c", "x"], ["a", "z"]], dtype=object)


def describe(label, est, Xfit, row):
    est.fit(Xfit)
    out = est.transform([row])
    print(f"== {label}")
    print("  transform([row]) type:", type(out).__module__ + "." + type(out).__name__,
          "format", getattr(out, "format", None), "shape", out.shape, "dtype", out.dtype,
          "nnz", out.nnz, "issparse", sp.issparse(out), "isspmatrix", sp.isspmatrix(out))
    # fit probe as _projection._fit_step does
    probe = np.asarray(out)
    print("  np.asarray(out): ndim", probe.ndim, "shape", probe.shape, "dtype", probe.dtype,
          "-> step width", probe.shape[1] if probe.ndim > 1 else 1)
    r0 = out[0]
    print("  out[0] type:", type(r0).__name__, "shape", getattr(r0, "shape", None),
          "format", getattr(r0, "format", None))
    try:
        lst = list(r0)
        print("  list(out[0]):", [type(v).__name__ for v in lst][:5], "len", len(lst))
        try:
            vals = tuple(float(v) for v in lst)
            print("  float() over it OK:", vals[:8])
        except Exception as e:
            print("  float() raises:", type(e).__name__, e)
    except Exception as e:
        print("  list(out[0]) raises:", type(e).__name__, e)
    dense = est.set_params(**({"sparse_output": False} if hasattr(est, "sparse_output") else {})) if False else None
    for name, f in [("toarray()", lambda: out.toarray()),
                    ("todense()", lambda: out.todense()),
                    ("np.asarray(todense())", lambda: np.asarray(out.todense())),
                    ("out[0].toarray()", lambda: r0.toarray())]:
        try:
            v = f()
            print(f"  {name}: {type(v).__name__} shape {v.shape} dtype {v.dtype}")
        except Exception as e:
            print(f"  {name} raises {type(e).__name__}: {e}")
    return out


describe("OneHotEncoder() numeric", OneHotEncoder(), X.round(0), X.round(0)[0].tolist())
describe("OneHotEncoder() strings", OneHotEncoder(), Xc, ["a", "y"])
describe("KBinsDiscretizer(encode='onehot')", KBinsDiscretizer(encode="onehot", quantile_method="averaged_inverted_cdf"), X, X[0].tolist())
describe("SplineTransformer(sparse_output=True)", SplineTransformer(sparse_output=True), X, X[0].tolist())
describe("MissingIndicator(sparse=True)", MissingIndicator(sparse=True, features="all"), np.where(X > 1, np.nan, X), [np.nan, 0.0])

# width-1 sparse: OneHotEncoder(drop='if_binary') on one binary feature
print("\n== width-1 sparse: OneHotEncoder(drop='if_binary') on a binary feature")
e = OneHotEncoder(drop="if_binary").fit(np.array([[0.0], [1.0], [1.0]]))
o = e.transform([[1.0]])
print("  type", type(o).__name__, "shape", o.shape)
probe = np.asarray(o)
print("  fit probe ndim", probe.ndim, "-> width", probe.shape[1] if probe.ndim > 1 else 1)
r0 = o[0]
for lab, f in [("list(row)", lambda: list(r0)), ("float(row)", lambda: float(r0)),
               ("float(list(row)[0])", lambda: float(list(r0)[0]))]:
    try:
        print(f"  {lab} ->", f())
    except Exception as ex:
        print(f"  {lab} raises {type(ex).__name__}: {ex}")

# sparse ARRAY (csr_array) behaviour, in case sklearn moves to it
print("\n== scipy csr_array (the sparse-array API)")
a = sp.csr_array(np.array([[0.0, 1.0, 0.0]]))
print("  csr_array[0]:", type(a[0]).__name__, "shape", a[0].shape, "ndim", a[0].ndim)
try:
    print("  list(csr_array[0]):", list(a[0]))
except Exception as ex:
    print("  list(csr_array[0]) raises", type(ex).__name__, ex)
try:
    print("  [float(v) for v in csr_array[0]]:", [float(v) for v in a[0]])
except Exception as ex:
    print("  float over csr_array[0] raises", type(ex).__name__, ex)
print("  np.asarray(csr_array):", np.asarray(a).ndim, np.asarray(a).dtype)
print("  csr_array.toarray():", a.toarray(), a.toarray().dtype)
print("  csr_array.todense():", type(a.todense()).__name__)
print("  csr_matrix.todense():", type(sp.csr_matrix(a).todense()).__name__)
# sklearn config: does sklearn 1.9 emit sparse arrays anywhere?
import sklearn
print("  sklearn.get_config keys with 'sparse':", {k: v for k, v in sklearn.get_config().items() if 'sparse' in k or 'array' in k})
