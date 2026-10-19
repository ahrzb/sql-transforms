import numpy as np, threadpoolctl, json
from numpy.lib.introspect import opt_func_info
info = threadpoolctl.threadpool_info()
print([ (i['internal_api'], i.get('architecture'), i.get('num_threads')) for i in info])
for f in ["log","log1p","expm1","cos","sin","exp"]:
    d = opt_func_info(func_name=f"^{f}$", signature="float64")
    print(f, json.dumps(d))
from numpy._core._multiarray_umath import __cpu_features__ as cf
print({k:v for k,v in cf.items() if v})
