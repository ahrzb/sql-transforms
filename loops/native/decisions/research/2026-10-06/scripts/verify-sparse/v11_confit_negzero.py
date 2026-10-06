import sys; sys.path.insert(0, sys.argv[1]); import shim  # noqa
import math, pyarrow as pa
from sql_transform import SQLProjection
T = pa.table({"x": [-0.0, 1.0, 0.0]})
for sql in ("SELECT x + 0.0 AS y, x AS z, 0.0 + x AS w, x + CAST(0.0 AS DOUBLE) AS v FROM __THIS__",):
    p = SQLProjection(sql).fit(T)
    d = p.transform(T).to_pylist()
    i = p.infer_batch(T.to_pylist())
    print("duckdb:", [{k: (v, math.copysign(1, v)) for k, v in r.items()} for r in d[:1]])
    print("confit:", [{k: (v, math.copysign(1, v)) for k, v in r.items()} for r in i[:1]])
