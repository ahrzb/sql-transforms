"""Independent K measurement. Entry: scipy's YJ predicates, Goldberg log1p and
Kahan expm1 (power.py's arms + an inf arm) on glibc log/exp via math.
usage: vk2.py FX.pkl TW_A.pkl TW_B.pkl"""
import sys, pickle, math, json, struct, warnings
import numpy as np
from scipy import stats
from scipy.special import boxcox
warnings.filterwarnings("ignore"); np.seterr(all="ignore")
EPS = 2.0 ** -52
INF = math.inf

def gexp(w):
    try:
        return math.exp(w)
    except OverflowError:
        return INF

def log1p_e(z):
    u = 1.0 + z
    if u == 1.0:
        return z
    return z * (math.log(u) / (u - 1.0))

def expm1_e(w):
    u = gexp(w)
    if u == INF:
        return INF
    if u == 1.0:
        return w
    if u - 1.0 == -1.0:
        return -1.0
    return (u - 1.0) * (w / math.log(u))

def yj_e(x, lam):
    if x != x:
        return math.nan
    if x >= 0:
        if abs(lam) < EPS:
            return log1p_e(x)
        return expm1_e(lam * log1p_e(x)) / lam
    if abs(lam - 2) > EPS:
        d = 2.0 - lam
        return -expm1_e(d * log1p_e(-x)) / d
    return -log1p_e(-x)

def bc_e(x, lam):
    if x != x or x <= 0:
        return math.nan
    lx = math.log(x)
    if abs(lam) < 1e-19:
        return lx
    w = lam * lx
    if w < 709.78:
        return expm1_e(w) / lam
    big = gexp(w - math.log(abs(lam)))
    return (big if lam > 0 else -big) - 1 / lam

def twin_w(x, lam, yj):
    if not yj:
        return 0.0
    if x != x:
        return 0.0
    if x >= 0:
        return 0.0 if abs(lam) < EPS else lam * float(np.log1p(np.array([x]))[0])
    return 0.0 if abs(lam - 2) <= EPS else (2.0 - lam) * float(np.log1p(np.array([-x]))[0])

def twin_t(x, lam, yj):
    if yj:
        return float(stats.yeojohnson(np.array([x]), lam)[0])
    return float(boxcox(x, lam))

def ordd(v):
    (i,) = struct.unpack("<q", struct.pack("<d", v))
    return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)

