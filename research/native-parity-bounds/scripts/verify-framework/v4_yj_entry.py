"""Yeo-Johnson entry spelled in DuckDB SQL (Goldberg log1p, Kahan expm1 from ln/exp),
against scipy's twin under V4/V2. K = |a-b| / (eps*(1+|w|)*|t|)."""
import math, numpy as np, duckdb, pyarrow as pa
EPS = 2.0 ** -52; e = float(np.finfo(float).eps)
tw = {v: np.load(f"v4_{v}.npz") for v in ["V4", "V2"]}
lam = tw["V4"]["lam"]; y = tw["V4"]["y"]
con = duckdb.connect(); con.register("t", pa.table({"z": y}))
def L(v): return f"CAST('{float(v)!r}' AS DOUBLE)"
def log1p(z): return f"(CASE WHEN 1.0 + {z} = 1.0 THEN {z} ELSE {z} * (ln(1.0 + {z}) / ((1.0 + {z}) - 1.0)) END)"
def expm1(w):
    u = f"exp({w})"
    return (f"(CASE WHEN {w} > 709.78 THEN CAST('inf' AS DOUBLE) WHEN {u} = 1.0 THEN {w} WHEN {u} - 1.0 = -1.0 THEN -1.0 "
            f"ELSE ({u} - 1.0) * ({w} / ln({u})) END)")
def od(a):
    i = np.asarray(a, np.float64).view(np.int64); return np.where(i >= 0, i, -(i & 0x7FFFFFFFFFFFFFFF)).astype(object)
ent = np.empty((len(lam), len(y))); W = np.empty_like(ent)
for i, l in enumerate(lam.tolist()):
    if abs(l) < e: pos_t, pos_w = log1p("z"), log1p("z")
    else: pos_w = f"({L(l)} * {log1p('z')})"; pos_t = f"({expm1(pos_w)} / {L(l)})"
    l2 = 2 - l
    if abs(l - 2) > e: neg_w = f"({L(l2)} * {log1p('-z')})"; neg_t = f"(-{expm1(neg_w)} / {L(l2)})"
    else: neg_t = f"(-{log1p('-z')})"; neg_w = log1p("-z")
    r = con.execute(f"SELECT CASE WHEN z >= 0 THEN {pos_t} ELSE {neg_t} END AS t, CASE WHEN z >= 0 THEN {pos_w} ELSE {neg_w} END AS w FROM t").fetchnumpy()
    ent[i] = r["t"]; W[i] = r["w"]
S = (1 + np.abs(W)) * np.abs(ent)
for v in ["V4", "V2"]:
    T = tw[v]["T"]
    fin = np.isfinite(T) & np.isfinite(ent)
    nfm = int(np.sum(~fin & (T != ent)))
    a, b, s = T[fin], ent[fin], S[fin]
    dif = a != b
    K = np.where(dif, np.abs(a - b) / (EPS * s), 0.0)
    u = np.abs(od(a) - od(b)).astype(float)
    print(f"YJ twin[{v}] vs DuckDB entry: lanes {a.size}, differ {int(dif.sum())}, max ulps {u.max():.0f}, max K {K.max():.3f}, p99.9 K {np.percentile(K, 99.9):.3f}, nonfinite mismatches {nfm}")
a, b = tw["V4"]["T"], tw["V2"]["T"]; fin = np.isfinite(a) & np.isfinite(b)
K = np.where(a[fin] != b[fin], np.abs(a[fin] - b[fin]) / (EPS * S[fin]), 0)
print(f"YJ twin[V4] vs twin[V2]: differ {int((a[fin] != b[fin]).sum())}, max ulps {np.abs(od(a[fin]) - od(b[fin])).max()}, max K {K.max():.3f}")
# where is the worst K for entry vs V2?
T = tw["V2"]["T"]; fin = np.isfinite(T) & np.isfinite(ent)
KK = np.where(fin & (T != ent), np.abs(T - ent) / (EPS * np.where(S > 0, S, 1)), 0)
i, k = np.unravel_index(np.argmax(KK), KK.shape)
print("worst lane: lam", lam[i], "z", y[k], "w", W[i, k], "twin", T[i, k], "entry", ent[i, k])
