"""V6: my own SQL spellings evaluated by DuckDB vs the report's saved N, on a
DIFFERENT random 1,500-row sample of every case (incl. pca_all)."""
import sys, math
import numpy as np, duckdb, pyarrow as pa
R = "/tmp/claude-0/-home-user-sql-transforms/903b9aef-c7e3-5c91-bc7f-b02f0b9c62f5/scratchpad/research/inference"
sys.path.insert(0, R)
from common import FEATS, PLAN, load_case, EPS  # noqa


def L(v):
    v = float(v)
    return "'nan'::DOUBLE" if math.isnan(v) else f"'{v!r}'::DOUBLE"


def dot(xs, w):
    return " + ".join(f"({x} * {L(c)})" for x, c in zip(xs, w))


def log1p(z):
    return f"(CASE WHEN (1.0::DOUBLE + {z}) = 1.0::DOUBLE THEN {z} ELSE {z} * (ln(1.0::DOUBLE + {z}) / ((1.0::DOUBLE + {z}) - 1.0::DOUBLE)) END)"


def expm1(w):
    u = f"exp({w})"
    return f"(CASE WHEN {u} = 1.0::DOUBLE THEN {w} WHEN {u} - 1.0::DOUBLE = -1.0::DOUBLE THEN -1.0::DOUBLE ELSE ({u} - 1.0::DOUBLE) * ({w} / ln({u})) END)"


def exprs(fam, est, p):
    xs = [f"c{i}" for i in range(p)]
    if fam.startswith("pca"):
        mc = (est.mean_.reshape(1, -1) @ est.components_.T)[0]
        sc = np.sqrt(est.explained_variance_); sc[sc < EPS] = EPS
        return [f"(({dot(xs, c)}) - {L(m)}) / {L(s)}" for c, m, s in zip(est.components_, mc, sc)]
    if fam.startswith("kmeans"):
        C = est.cluster_centers_; cc = np.einsum("ij,ij->i", C, C)
        xx = " + ".join(f"({x} * {x})" for x in xs)
        return [f"sqrt(greatest(((-2.0::DOUBLE * ({dot(xs, c)})) + ({xx})) + {L(k)}, 0.0::DOUBLE))" for c, k in zip(C, cc)]
    if fam.startswith("yj"):
        out = []
        for j, lam in enumerate(est.lambdas_.tolist()):
            x = xs[j]
            pos = log1p(x) if abs(lam) < EPS else f"({expm1(f'({L(lam)} * {log1p(x)})')} / {L(lam)})"
            neg = f"(-{expm1(f'({L(2 - lam)} * {log1p(f'(-{x})')})')} / {L(2 - lam)})" if abs(lam - 2) > EPS else f"(-{log1p(f'(-{x})')})"
            out.append(f"((CASE WHEN {x} >= 0.0::DOUBLE THEN {pos} ELSE {neg} END) - {L(est._scaler.mean_[j])}) / {L(est._scaler.scale_[j])}")
        return out
    s = 0.5; ch = float(np.cosh(np.pi * s))
    a = [f"(CASE WHEN {x} = 0.0::DOUBLE THEN 0.0::DOUBLE ELSE sqrt({x} * {L(s)}) END)" for x in xs]
    f = lambda x: f"sqrt(((2.0::DOUBLE * {x}) * {L(s)}) / {L(ch)})"  # noqa
    g = lambda x: f"(1.0::DOUBLE * ({L(s)} * ln({x})))"  # noqa  (j * ls, j = 1, as sklearn: j * log_step)
    b = [f"(CASE WHEN {x} = 0.0::DOUBLE THEN 0.0::DOUBLE ELSE {f(x)} * cos({g(x)}) END)" for x in xs]
    c = [f"(CASE WHEN {x} = 0.0::DOUBLE THEN 0.0::DOUBLE ELSE {f(x)} * sin({g(x)}) END)" for x in xs]
    return a + b + c


tot = bad = 0
rng = np.random.default_rng(31337)
for fam in PLAN:
    for dn in PLAN[fam][0]:
        c = load_case(fam, dn); X = c["Xte"]
        idx = np.sort(rng.choice(len(X), 1500, replace=False))
        N = np.load(FEATS / "native" / f"{fam}__{dn}__N.npy")[idx]
        con = duckdb.connect(); con.register("t", pa.table({f"c{i}": X[idx, i] for i in range(X.shape[1])}))
        E = exprs(fam, c["est"], X.shape[1])
        r = con.execute("SELECT " + ", ".join(f"{e} AS o{i}" for i, e in enumerate(E)) + " FROM t").fetchnumpy()
        got = np.column_stack([np.asarray(r[f"o{i}"], dtype=np.float64) for i in range(len(E))])
        mis = int(((got != N) & ~(np.isnan(got) & np.isnan(N))).sum()); tot += got.size; bad += mis
        print(fam, dn, "mismatch", mis, "/", got.size, flush=True)
print("TOTAL", bad, "/", tot)
