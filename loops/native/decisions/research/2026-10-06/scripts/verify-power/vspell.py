import sys, math, struct, numpy as np, gmpy2, duckdb, pyarrow as pa
from gmpy2 import mpfr
gmpy2.get_context().precision = 200
N = int(sys.argv[1]); rng = np.random.default_rng(99)
def ordd(v):
    (i,) = struct.unpack("<q", struct.pack("<d", v)); return i if i >= 0 else -(i & 0x7FFFFFFFFFFFFFFF)
def gold(z):
    u = 1.0 + z
    return z if u == 1.0 else z * (math.log(u) / (u - 1.0))
def kahan(w):
    u = math.exp(w)
    if u == 1.0: return w
    if u - 1.0 == -1.0: return -1.0
    return (u - 1.0) * (w / math.log(u))
k = N // 4
z = np.concatenate([10.0 ** rng.uniform(-20, -8, k), 10.0 ** rng.uniform(-8, 0, k), rng.uniform(0, 1e3, k), 10.0 ** rng.uniform(0, 300, N - 3 * k)])
w = np.concatenate([10.0 ** rng.uniform(-18, 0.5, k) * rng.choice([-1, 1], k), rng.uniform(-40, 40, k), rng.uniform(-745, 709, k), rng.uniform(-3, 3, N - 3 * k)])
svml_l = np.log1p(z); svml_e = np.expm1(w)
for name, xs, f, ref, sv, gl in (("goldberg", z, gold, gmpy2.log1p, svml_l, math.log1p), ("kahan", w, kahan, gmpy2.expm1, svml_e, math.expm1)):
    errs = np.empty(len(xs)); dg = np.empty(len(xs), dtype=np.int64); ds = np.empty(len(xs), dtype=np.int64); ys = np.empty(len(xs))
    for i, x in enumerate(xs):
        x = float(x); y = f(x); ys[i] = y
        ex = ref(mpfr(x)); cr = float(ex)
        errs[i] = float(abs(mpfr(y) - ex) / math.ulp(cr)) if cr != 0 else (0 if y == 0 else np.inf)
        dg[i] = abs(ordd(y) - ordd(gl(x))); ds[i] = abs(ordd(y) - ordd(float(sv[i])))
    print(name, "max err ulp", errs.max(), "p99.9", np.quantile(errs, 0.999), "not CR", (errs > 0.5).mean(),
          "max dist glibc", dg.max(), "max dist svml", ds.max(), "argmax x", repr(float(xs[errs.argmax()])))
    # duckdb bit identity
    con = duckdb.connect()
    con.register("t", pa.table({"x": pa.array(xs[:200000])}))
    if name == "goldberg":
        q = "select case when 1.0+x = 1.0 then x else x * (ln(1.0+x) / ((1.0+x) - 1.0)) end as y from t"
    else:
        q = "select case when exp(x)=1.0 then x when exp(x)-1.0 = -1.0 then -1.0 else (exp(x)-1.0)*(x/ln(exp(x))) end as y from t"
    yd = con.sql(q).fetchnumpy()["y"]
    print("  duckdb mismatches", int((yd.view(np.int64) != ys[:200000].view(np.int64)).sum()), "of", len(yd))
