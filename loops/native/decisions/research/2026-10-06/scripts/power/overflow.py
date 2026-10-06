"""Where each expm1 first answers inf: numpy (this dispatch), glibc, and the
Kahan spelling on glibc exp/log; and how many doubles lie between."""
import ctypes, math, numpy as np
libm = ctypes.CDLL("libm.so.6")
for f in ("expm1", "exp", "log"):
    getattr(libm, f).restype = ctypes.c_double; getattr(libm, f).argtypes = [ctypes.c_double]
def kahan(w):
    u = libm.exp(w)
    if math.isinf(u): return math.inf
    if u == 1.0: return w
    return (u - 1.0) * (w / libm.log(u))
fs = {"numpy": lambda w: float(np.expm1(np.array([w]))[0]), "glibc": libm.expm1, "kahan": kahan}
np.seterr(all="ignore")
lo = 709.7
for name, f in fs.items():
    a, b = 709.0, 710.0
    while np.nextafter(a, 2e3) != b:
        mid = (a + b) / 2
        if mid in (a, b): mid = np.nextafter(a, 2e3)
        (a, b) = (a, mid) if math.isinf(f(mid)) else (mid, b)
    # check monotone neighbourhood
    print(f"{name:6}: last finite w = {a!r} ({f(a)!r}), first inf w = {b!r}")
print("ln(DBL_MAX) =", math.log(np.finfo(float).max))
