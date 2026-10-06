"""With NPY_DISABLE_CPU_FEATURES=X86_V4 (set by the caller), the twin
(sklearn) against the native lanes (confit) on the k4 draw families."""
import os, sys, importlib.util, json
import numpy as np
spec = importlib.util.spec_from_file_location("k4", os.path.join(os.path.dirname(os.path.abspath(__file__)), "k4_K.py"))
k4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(k4)
rng = np.random.default_rng(3)
out = {"env": os.environ.get("NPY_DISABLE_CPU_FEATURES")}
for steps, s in ((2, None), (3, None), (4, 0.4)):
    ss = k4.DEFAULT_S[steps] if s is None else s
    xs = [rng.uniform(0, 1e3, 200_000), np.ldexp(rng.uniform(1, 2, 200_000), rng.integers(-996, 997, 200_000)), 1 + rng.normal(0, 1e-3, 200_000)]
    for j in range(1, steps):
        for lane in ("cos", "sin"):
            xs.append(k4.adversarial(ss, j, 300_000, lane))
    x = np.concatenate(xs); x = x[x > 0]
    t = k4.twin(x, steps, s); nat = k4.native(x, steps, ss)
    diff = sum(int((t[k] != nat[k]).sum()) for k in t)
    mism = int((np.log(x) != k4.glibc_log(x)).sum())
    out[f"steps={steps} s={ss}"] = {"draws": int(len(x)), "lanes": int(len(x) * len(t)), "lanes differ": diff, "log mismatches": mism}
print(json.dumps(out))
