import sys; sys.path.insert(0, sys.argv[1]); import shim  # noqa
import warnings; warnings.simplefilter("ignore")
import numpy as np, pyarrow as pa
from sklearn.preprocessing import KBinsDiscretizer
from sql_transform.model._foreign import Transform
t = Transform.from_estimator(KBinsDiscretizer(n_bins=3, strategy="uniform"), takes=("a",), returns=("b0", "b1", "b2"))
rel = pa.table({"a": [0.0, 1.0, 2.0, 3.0, 4.0]})
inst = t.fit(rel)
try:
    print(t.transform(inst, rel))
except Exception as e:
    print("model.Transform.from_estimator(KBins onehot) transform raises", type(e).__name__, e)
t2 = Transform.from_estimator(KBinsDiscretizer(n_bins=3, strategy="uniform", encode="onehot-dense"), takes=("a",), returns=("b0", "b1", "b2"))
print(t2.transform(t2.fit(rel), rel).to_pylist()[:2])
