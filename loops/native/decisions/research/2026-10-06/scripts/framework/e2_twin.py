"""The twin's numpy formulas (sklearn AdditiveChi2Sampler._transform_dense,
scipy _yeojohnson_transform), in whatever numpy kernels the env selects."""
import sys, numpy as np, scipy.stats as st
from numpy.lib.introspect import opt_func_info
d = np.load("e2_draws.npz"); out = {}
x = d["chi2_x"]
for s, js in [(0.8, [1]), (0.5, [1, 2]), (0.4, [1, 2, 3])]:
    log_step = s * np.log(x); step = 2 * x * s
    for j in js:
        f = np.sqrt(step / np.cosh(np.pi * j * s))
        out[f"cos_{s}_{j}"] = f * np.cos(j * log_step)
        out[f"sin_{s}_{j}"] = f * np.sin(j * log_step)
out["log"] = np.log(x)
y = d["yj_x"]
out["yj"] = np.concatenate([st.yeojohnson(y, float(l)) for l in d["lam"]])
out["log1p"] = np.log1p(np.abs(y)); out["expm1"] = np.expm1(np.linspace(-40, 700, 200001))
np.savez(sys.argv[1], **out)
print({f: opt_func_info(func_name=f"^{f}$", signature="float64")[f]["dd"]["current"] for f in ["log", "cos", "log1p", "expm1"]})
