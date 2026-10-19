"""Q2c: pinning numpy's kernel. Run in a fresh process per env setting:
NPY_DISABLE_CPU_FEATURES must be set before numpy is imported.
Prints whether np.log equals glibc's (math.log) bit-for-bit on draws that
include known mismatches, and what numpy reports as found/disabled."""
import math, os, sys, json
import numpy as np
from numpy._core._multiarray_umath import __cpu_features__ as F
rng = np.random.default_rng(5)
x = np.concatenate([rng.uniform(0, 1e3, 300_000), 1 + rng.normal(0, 1e-3, 300_000),
                    rng.uniform(512, 1000, 400_000)])
x = x[x > 0]
g = np.array([math.log(v) for v in x.tolist()])
n = np.log(x)
on = sorted(k for k, v in F.items() if v and (k.startswith("X86") or k.startswith("AVX512")))
print(json.dumps({"env": os.environ.get("NPY_DISABLE_CPU_FEATURES"), "n": int(len(x)),
                  "np.log != glibc": int((n != g).sum()),
                  "x86/avx512 features on": on}))
