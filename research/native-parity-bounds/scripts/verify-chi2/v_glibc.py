"""Dump glibc log/cos/sin (via ctypes libm) on fixed draws to compare across
GLIBC_TUNABLES settings."""
import ctypes
import sys

import numpy as np

libm = ctypes.CDLL("libm.so.6")
rng = np.random.default_rng(2024)
x_log = np.concatenate([rng.uniform(0.9, 1.1, 1_000_000), rng.uniform(0, 1e3, 1_000_000)])
x_trig = np.concatenate([rng.uniform(-10, 10, 500_000), rng.uniform(-1e5, 1e5, 500_000)])
out = {}
for name, xs in (("log", x_log), ("cos", x_trig), ("sin", x_trig)):
    f = getattr(libm, name)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double]
    out[name] = np.array([f(v) for v in xs.tolist()])
np.savez(sys.argv[1], **out)
print("glibc", ctypes.CDLL(None).gnu_get_libc_version and ctypes.c_char_p(ctypes.CDLL("libc.so.6").gnu_get_libc_version()).value if False else "")
