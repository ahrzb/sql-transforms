"""Own Dot2 (Ogita-Rump-Oishi) in DuckDB, written as nested subqueries
(one per term), against the LTR dot; n=32, 200k rows, threads=1 and 4."""
import math, time, random
import duckdb, numpy as np
N, ROWS = 32, 200_000
rng = np.random.default_rng(31)
c = rng.standard_normal(N); m = rng.uniform(-1e4, 1e4, N)
X = m + rng.standard_normal((ROWS, N)) * 10 ** rng.uniform(-8, 1, (ROWS, 1))
SP = 134217729.0
def L(v): return f"'{repr(float(v))}'::DOUBLE"
def spl(a): t = SP * a; h = t - (t - a); return h, a - h
ch = [spl(float(v)) for v in c]
con = duckdb.connect(); con.execute("CREATE TABLE t AS SELECT * FROM X_df")  if False else None
import pandas as pd
con.register("df", pd.DataFrame(X, columns=[f"x{i}" for i in range(N)])); con.execute("CREATE TABLE t AS SELECT * FROM df")
xs = ", ".join(f"x{i}" for i in range(N))
ltr = f"x0*{L(c[0])}"
for i in range(1, N): ltr = f"({ltr} + x{i}*{L(c[i])})"
q_ltr = f"SELECT {ltr} AS d FROM t"
def terr(i):
    # TwoProduct error of x_i*c_i (Dekker), x split in SQL, c split as constants
    xh = f"({L(SP)}*x{i} - ({L(SP)}*x{i} - x{i}))"; xl = f"(x{i} - {xh})"
    return f"({xl}*{L(ch[i][1])} - ((((x{i}*{L(c[i])}) - {xh}*{L(ch[i][0])}) - {xl}*{L(ch[i][0])}) - {xh}*{L(ch[i][1])}))"
q = f"SELECT {xs}, x0*{L(c[0])} AS p, {terr(0)} AS s FROM t"
for i in range(1, N):
    q = f"SELECT {xs}, p, s, x{i}*{L(c[i])} AS h, p + x{i}*{L(c[i])} AS pn, {terr(i)} AS r FROM ({q})"
    q = f"SELECT {xs}, pn AS p, s + (((p - (pn - (pn - p))) + (h - (pn - p))) + r) AS s FROM ({q})"
q_dot2 = f"SELECT p + s AS d FROM ({q})"
def run(sql, th):
    con.execute(f"SET threads={th}"); best = 1e9
    for _ in range(5):
        t0 = time.perf_counter(); con.execute(f"SELECT sum(d) FROM ({sql})").fetchall(); best = min(best, time.perf_counter() - t0)
    return best
for th in (1, 4):
    a, b = run(q_ltr, th), run(q_dot2, th)
    print(f"threads {th}: LTR {a*1e3:.1f} ms, Dot2 {b*1e3:.1f} ms, ratio {b/a:.1f}x; sql chars {len(q_ltr)} vs {len(q_dot2)}")
d2 = np.array(con.execute(q_dot2).fetchnumpy()["d"]); dl = np.array(con.execute(q_ltr).fetchnumpy()["d"])
idx = random.Random(1).sample(range(ROWS), 5000)
def exact(x):
    parts = []
    for a, b in zip(x, c): p = a * b; parts += [p, math.fma(a, b, -p)]
    return math.fsum(parts)
def pydot2(x):
    p = x[0] * c[0]; s = math.fma(x[0], c[0], -p)
    for i in range(1, N):
        h = x[i] * c[i]; r = math.fma(x[i], c[i], -h); pn = p + h; z = pn - p; q_ = (p - (pn - z)) + (h - z); p = pn; s = s + (q_ + r)
    return p + s
cr = sum(d2[i] == exact(X[i].tolist()) for i in idx); spec = sum(d2[i] == pydot2(X[i].tolist()) for i in idx); crl = sum(dl[i] == exact(X[i].tolist()) for i in idx)
cond = np.array([np.sum(np.abs(X[i] * c)) / abs(exact(X[i].tolist())) for i in idx])
print(f"Dot2 == spec {spec}/5000; Dot2 correctly rounded {cr}/5000; LTR correctly rounded {crl}/5000; cond median {np.median(cond):.3g} max {cond.max():.3g}")
