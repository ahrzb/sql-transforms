"""glibc's log/exp/log1p/expm1 on fixed draws; run with and without
GLIBC_TUNABLES=glibc.cpu.hwcaps=-FMA,-AVX2_Usable,... to see whether the FMA
ifunc variants answer differently. usage: python glibc_fma.py OUT.npz"""
import ctypes, sys
import numpy as np
libm = ctypes.CDLL("libm.so.6")
rng = np.random.default_rng(3)
X = {"log": np.concatenate([2.0 ** rng.uniform(-1000, 1000, 200000), rng.uniform(0.5, 2, 200000)]),
     "exp": rng.uniform(-745, 709, 400000),
     "log1p": np.concatenate([2.0 ** rng.uniform(-60, 1000, 200000), rng.uniform(0, 1e3, 200000)]),
     "expm1": np.concatenate([rng.uniform(-40, 709, 200000), rng.uniform(-3, 3, 200000)])}
out = {}
for f, x in X.items():
    g = getattr(libm, f); g.restype = ctypes.c_double; g.argtypes = [ctypes.c_double]
    out[f] = np.array([g(float(v)) for v in x])
np.savez(sys.argv[1], **out)
