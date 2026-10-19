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

fx = pickle.load(open(sys.argv[1], "rb"))
twA = pickle.load(open(sys.argv[2], "rb"))
twB = pickle.load(open(sys.argv[3], "rb"))
stats_by = {}
for d, oa, ob in zip(fx, twA, twB):
    cfg = d["cfg"]; yj = cfg.startswith("yj")
    st = stats_by.setdefault(cfg, dict(lanes=0, differ=0, K_rec=0.0, K_tight=0.0, K_AB=0.0, K_gl=0.0,
        ulps=0, ulps_AB=0, differ_AB=0, infnan=0, infnan_AB=0, ks=[], worst=None, worst_ulp=None))
    for r, a, b in zip(d["rows"], oa, ob):
        if a is None:
            continue
        est = d["inst"][r["__iid"]]
        vals = [math.nan if r[f] is None else float(r[f]) for f in d["names"]]
        for j, x in enumerate(vals):
            lam = float(est.lambdas_[j])
            t_n = (yj_e if yj else bc_e)(x, lam)
            w = twin_w(x, lam, yj)
            tt = twin_t(x, lam, yj) if x == x else math.nan
            St_rec = (1 + abs(w)) * abs(tt)
            St_t = (1 + max(w, 0.0)) * abs(tt)
            n = t_n
            if est.standardize:
                m = float(est._scaler.mean_[j]); s = float(est._scaler.scale_[j])
                n = (t_n - m) / s
                S_rec = (St_rec + abs(m)) / s; S_t = (St_t + abs(m)) / s
            else:
                S_rec, S_t = St_rec, St_t
            av, bv = float(a[j]), float(b[j])
            st["lanes"] += 1
            for (p, q, tag) in ((av, n, ""), (av, bv, "_AB"), (bv, n, "_gl")):
                same = (p == q) or (p != p and q != q)
                if same:
                    continue
                if tag != "_gl":
                    st["differ" + tag] += 1
                if not (math.isfinite(p) and math.isfinite(q)):
                    st["infnan" + ("_AB" if tag == "_AB" else "")] += 1
                    if tag == "":
                        print("INFNAN", cfg, d["seed"], x, lam, p, q)
                    continue
                dd = abs(p - q)
                k_rec = dd / (EPS * S_rec) if S_rec > 0 else INF
                k_t = dd / (EPS * S_t) if S_t > 0 else INF
                u = abs(ordd(p) - ordd(q))
                if tag == "":
                    st["ks"].append(k_t)
                    if k_rec > st["K_rec"]:
                        st["K_rec"] = k_rec
                    if k_t > st["K_tight"]:
                        st["K_tight"] = k_t
                        st["worst"] = dict(seed=d["seed"], x=x, lam=lam, w=w, t=tt, twin=p, nat=q, ulps=u, k_rec=k_rec)
                    if u > st["ulps"]:
                        st["ulps"] = u
                        st["worst_ulp"] = dict(seed=d["seed"], x=x, lam=lam, w=w, t=tt, twin=p, nat=q, k_rec=k_rec, k_t=k_t)
                elif tag == "_AB":
                    st["K_AB"] = max(st["K_AB"], k_rec)
                    st["ulps_AB"] = max(st["ulps_AB"], u)
                else:
                    st["K_gl"] = max(st["K_gl"], max(k_rec, 0))
for cfg, st in stats_by.items():
    ks = np.array(st.pop("ks")) if st["ks"] else np.zeros(1)
    st["K_tight_p999_of_differing"] = float(np.quantile(ks, 0.999))
    st["frac_differ"] = st["differ"] / st["lanes"]
    st["frac_differ_AB"] = st["differ_AB"] / st["lanes"]
    print(cfg, json.dumps(st, default=float))
