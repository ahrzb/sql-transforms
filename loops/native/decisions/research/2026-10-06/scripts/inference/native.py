"""N: the native-like features (entry order, glibc via math), their per-lane
term scale S, and a bitwise cross-check of N against DuckDB evaluating the
same expression as SQL on a sample of rows."""

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402

from common import (  # noqa: E402
    FEATS, PLAN, load_case, native_chi2, native_kmeans, native_kmeans_direct,
    native_pca, native_yj, EPS,
)


def lit(v: float) -> str:
    v = float(v)
    if np.isnan(v):
        return "'nan'::DOUBLE"
    return f"'{v!r}'::DOUBLE"


def lr_sql(xs, ws):
    return " + ".join(f"({x} * {lit(w)})" for x, w in zip(xs, ws))


def duck_check(fam, c, N, nrows=1500):
    import duckdb

    est, X = c["est"], c["Xte"][:nrows]
    p = X.shape[1]
    xs = [f"f{i}" for i in range(p)]
    con = duckdb.connect()
    import pyarrow as pa

    con.register("t", pa.table({f"f{i}": X[:, i] for i in range(p)}))
    exprs = []
    if fam.startswith("pca"):
        mc = (est.mean_.reshape(1, -1) @ est.components_.T)[0]
        scale = np.sqrt(est.explained_variance_)
        scale[scale < EPS] = EPS
        for k in range(est.components_.shape[0]):
            exprs.append(f"(({lr_sql(xs, est.components_[k])}) - {lit(mc[k])}) / {lit(scale[k])}")
    elif fam.startswith("kmeans"):
        C = est.cluster_centers_
        YY = np.einsum("ij,ij->i", C, C)
        xx = " + ".join(f"({x} * {x})" for x in xs)
        for k in range(len(C)):
            exprs.append(f"sqrt(greatest(((-2.0::DOUBLE * ({lr_sql(xs, C[k])})) + ({xx})) + {lit(YY[k])}, 0::DOUBLE))")
    elif fam.startswith("yj"):
        def log1p(z):
            return f"(CASE WHEN (1::DOUBLE + {z}) = 1::DOUBLE THEN {z} ELSE {z} * (ln(1::DOUBLE + {z}) / ((1::DOUBLE + {z}) - 1::DOUBLE)) END)"

        def expm1(w):
            return (f"(CASE WHEN exp({w}) = 1::DOUBLE THEN {w} WHEN exp({w}) - 1::DOUBLE = -1::DOUBLE THEN -1::DOUBLE"
                    f" ELSE (exp({w}) - 1::DOUBLE) * ({w} / ln(exp({w}))) END)")
        m, s = est._scaler.mean_, est._scaler.scale_
        for j, lam in enumerate(est.lambdas_):
            lam = float(lam)
            x = xs[j]
            pos = (f"{log1p(x)}" if abs(lam) < EPS
                   else f"({expm1(f'({lit(lam)} * {log1p(x)})')} / {lit(lam)})")
            g = 2 - lam
            neg = (f"(-{expm1(f'({lit(g)} * {log1p(f'(-{x})')})')} / {lit(g)})" if abs(lam - 2) > EPS
                   else f"(-{log1p(f'(-{x})')})")
            exprs.append(f"((CASE WHEN {x} >= 0 THEN {pos} ELSE {neg} END) - {lit(m[j])}) / {lit(s[j])}")
    elif fam == "chi2":
        s = 0.5
        ch = float(np.cosh(np.pi * 1 * s))
        lanes0 = [f"(CASE WHEN {x} = 0 THEN 0::DOUBLE ELSE sqrt({x} * {lit(s)}) END)" for x in xs]
        f = lambda x: f"sqrt(((2::DOUBLE * {x}) * {lit(s)}) / {lit(ch)})"  # noqa: E731
        a = lambda x: f"(1::DOUBLE * ({lit(s)} * ln({x})))"  # noqa: E731
        lanes1 = [f"(CASE WHEN {x} = 0 THEN 0::DOUBLE ELSE {f(x)} * cos({a(x)}) END)" for x in xs]
        lanes2 = [f"(CASE WHEN {x} = 0 THEN 0::DOUBLE ELSE {f(x)} * sin({a(x)}) END)" for x in xs]
        exprs = lanes0 + lanes1 + lanes2
    sql = "SELECT " + ", ".join(f"{e} AS o{i}" for i, e in enumerate(exprs)) + " FROM t"
    res = con.execute(sql).fetchnumpy()
    got = np.column_stack([np.asarray(res[f"o{i}"], dtype=np.float64) for i in range(len(exprs))])
    want = N[:nrows]
    same = (got == want) | (np.isnan(got) & np.isnan(want))
    return int((~same).sum()), got.size


def main():
    out = FEATS / "native"
    out.mkdir(parents=True, exist_ok=True)
    fams = sys.argv[1:] or list(PLAN)
    for fam in fams:
        for dn in PLAN[fam][0]:
            t0 = time.time()
            c = load_case(fam, dn)
            est, X = c["est"], c["Xte"]
            extra = {}
            if fam.startswith("pca"):
                N, S = native_pca(est, X)
            elif fam.startswith("kmeans"):
                N, S, D = native_kmeans(est, X)
                extra["direct"] = native_kmeans_direct(est, X)
                extra["D"] = D
            elif fam.startswith("yj"):
                N, S = native_yj(est, X)
            elif fam == "chi2":
                N, S = native_chi2(est, X)
            np.save(out / f"{fam}__{dn}__N.npy", N)
            np.save(out / f"{fam}__{dn}__S.npy", S)
            for k, v in extra.items():
                np.save(out / f"{fam}__{dn}__{k}.npy", v)
            bad, tot = duck_check(fam, c, N)
            print(f"{fam} {dn} N{N.shape} duckdb-vs-N mismatches {bad}/{tot} {time.time() - t0:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main()
