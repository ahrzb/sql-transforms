"""Entry spellings shared by weird.py (copied from kmeasure.py)."""
import ctypes, math, warnings
import numpy as np
EPS = 2.0**-52
libm = ctypes.CDLL("libm.so.6")
for f in ("log", "exp"):
    getattr(libm, f).restype = ctypes.c_double
    getattr(libm, f).argtypes = [ctypes.c_double]
ln, ex = libm.log, libm.exp
warnings.filterwarnings("ignore")
np.seterr(all="ignore")


def log1p_g(z):  # Goldberg: z * (ln(u) / (u - 1)), u = 1 + z
    u = 1.0 + z
    if u == 1.0:
        return z
    return z * (ln(u) / (u - 1.0))


def expm1_k(w):  # Kahan: (u - 1) * (w / ln(u)), u = exp(w); power.py's arms
    u = ex(w)
    if math.isinf(u):
        return math.inf  # an overflow arm the YJ entry would need
    if u == 1.0:
        return w
    if u - 1.0 == -1.0:
        return -1.0
    return (u - 1.0) * (w / ln(u))


def yj(x, lam):
    """(t, w): scipy's _yeojohnson_transform branches, eps = 2**-52."""
    if math.isnan(x):
        return math.nan, 0.0
    if x >= 0:
        if abs(lam) < EPS:
            return log1p_g(x), 0.0
        w = lam * log1p_g(x)
        return expm1_k(w) / lam, w
    if abs(lam - 2) > EPS:
        d = 2.0 - lam
        w = d * log1p_g(-x)
        return -expm1_k(w) / d, w
    return -log1p_g(-x), 0.0


def bc(x, lam):
    """(t, w): xsf boxcox branches (power.py's _box_cox)."""
    if math.isnan(x) or x <= 0:
        return math.nan, 0.0
    lx = ln(x)
    if abs(lam) < 1e-19:
        return lx, 0.0
    w = lam * lx
    if w < 709.78:
        return expm1_k(w) / lam, 0.0  # log shared: S_t = |t|
    big = ex(w - math.log(abs(lam)))
    return (big if lam > 0 else -big) - 1 / lam, 0.0


E = EPS
YJ_LAMBDAS = [0.0, E / 2, E, -E, 2.0, 2 - E, 2 + 2 * E, 2 - 2 * E, 1.0, -2.0,
              3.0, 4.5, -4.0, 1e-8, 0.5]



def ordered_dist(a, b):
    def o(v):
        i = np.float64(v).view(np.int64).item()
        return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
    return abs(o(a) - o(b))
