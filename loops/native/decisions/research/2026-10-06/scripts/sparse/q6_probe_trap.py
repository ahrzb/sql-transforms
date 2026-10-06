"""Q6: the catalog's OneHotEncoder probe (native/encode.py `_probe`) on a
sparse-output twin: np.asarray(sparse, dtype=float64) raises ValueError,
which `_probe` reads as 'the twin raises here'."""
import sys
sys.path.insert(0, sys.argv[1])
import _shim  # noqa
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sql_transform.native.encode import _probe

X = np.array([["a"], ["b"], ["c"]], dtype=object)
for so in (False, True):
    est = OneHotEncoder(sparse_output=so, handle_unknown="ignore").fit(X)
    print(f"sparse_output={so}:", [(_probe(est, ["a"], 0, v)) for v in ("a", "b", "zz")])
