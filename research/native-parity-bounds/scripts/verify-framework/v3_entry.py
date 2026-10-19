"""The entry spelled in DuckDB SQL (ln/cos/sin/sqrt are DuckDB's), cosh constant from
glibc (math.cosh) or from numpy-V4; compared with sklearn's twin under V4 and baseline."""
import math, struct, numpy as np, duckdb, pyarrow as pa
EPS = 2.0 ** -52
x = np.load("v3_x.npy"); tw = {v: np.load(f"v3_{v}.npz") for v in ["V4", "V2"]}
con = duckdb.connect(); con.register("t", pa.table({"x": x}))
def lit(v): return f"CAST('{float(v)!r}' AS DOUBLE)"
def od(a):
    i = np.asarray(a, np.float64).view(np.int64); return np.where(i >= 0, i, -(i & 0x7FFFFFFFFFFFFFFF)).astype(object)
# kernel-level comparisons
glog = np.array(con.execute("SELECT ln(x) FROM t").fetchnumpy()["ln(x)"])
for v in tw:
    d = tw[v]["log"] != glog
    print(f"np.log[{v}] vs DuckDB ln: differ {int(d.sum())} of {len(x)}, max ulps {int(np.abs(od(tw[v]['log']) - od(glog)).max())}")
pyl = np.array([math.log(t) for t in x.tolist()]); print("DuckDB ln vs Python math.log differ:", int((pyl != glog).sum()))
ck = dict(zip(tw["V4"]["cosh_keys"], tw["V4"]["cosh_vals"])); ck2 = dict(zip(tw["V2"]["cosh_keys"], tw["V2"]["cosh_vals"]))
cfgs = [(2, None), (3, None), (2, 0.8), (3, 0.5), (4, 0.4), (6, 0.3), (8, 0.7)]
dflt = {1: 0.8, 2: 0.5, 3: 0.4}
summary = []
for steps, iv in cfgs:
    s = iv if iv is not None else dflt[steps]
    for j in range(1, steps):
        cg = math.cosh(math.pi * j * s); c4 = ck[f"{j}_{s}"]; c2 = ck2[f"{j}_{s}"]
        if j == 1 or True:
            pass
        for cname, C in [("glibc", cg), ("npV4", c4)]:
            q = (f"SELECT sqrt((2.0 * x * {lit(s)}) / {lit(C)}) AS f, CAST({j} AS DOUBLE) * ({lit(s)} * ln(x)) AS th FROM t")
            r = con.execute(q).fetchnumpy(); f = r["f"]; th = r["th"]
            q2 = f"SELECT f * cos(th) AS c, f * sin(th) AS s FROM (" + q + ")"
            r2 = con.execute(q2).fetchnumpy()
            S = f * (1 + np.abs(th))
            for k, trig in enumerate(["c", "s"]):
                col = 1 + 2 * (j - 1) + k
                ent = r2[trig]
                for v in ["V4", "V2"]:
                    T = tw[v][f"{steps}_{iv}"][:, col]
                    dif = T != ent
                    K = np.where(dif, np.abs(T - ent) / (EPS * S), 0.0)
                    u = np.abs(od(T) - od(ent)).astype(float)
                    summary.append((steps, iv, s, j, trig, cname, v, int(dif.sum()), u.max(), K.max()))
print("cosh: (s,j) where numpy-V4 cosh != glibc cosh:", [(k, float(ck[k]).hex(), math.cosh(math.pi*int(k.split('_')[0])*float(k.split('_')[1])).hex()) for k in ck if ck[k] != math.cosh(math.pi*int(k.split('_')[0])*float(k.split('_')[1]))])
print("cosh: numpy-V2 vs glibc differ:", [k for k in ck2 if ck2[k] != math.cosh(math.pi*int(k.split('_')[0])*float(k.split('_')[1]))])
print("steps interval s j trig coshsrc twin  differ  max_ulps  max_K")
for row in summary: print(*row[:7], row[7], f"{row[8]:.0f}", f"{row[9]:.3f}")
