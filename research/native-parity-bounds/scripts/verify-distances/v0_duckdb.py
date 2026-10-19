"""Does DuckDB evaluate the native spelling the way the report's Python model does?
(left-to-right, unfused dot; sqrt(greatest(q,0)); ln/exp/cos == glibc via math)."""
import math, duckdb, numpy as np
rng = np.random.default_rng(90210)
con = duckdb.connect()
n, rows = 24, 4000
X = rng.normal(size=(rows, n)) * 10.0 ** rng.integers(-3, 4, size=(rows, n))
C = rng.normal(size=(rows, n)) * 10.0 ** rng.integers(-3, 4, size=(rows, n))
XX = rng.normal(size=rows) * 1e3
cols = [f"x{i} DOUBLE" for i in range(n)] + [f"c{i} DOUBLE" for i in range(n)] + ["xx DOUBLE", "cc DOUBLE"]
con.execute(f"CREATE TABLE t (id INTEGER, {', '.join(cols)})")
data = [[i] + list(map(float, X[i])) + list(map(float, C[i])) + [float(abs(XX[i])), float(abs(XX[i])) * 0.7] for i in range(rows)]
con.executemany(f"INSERT INTO t VALUES ({', '.join(['?'] * (2 * n + 3))})", data)
dot = " + ".join(f"x{i} * c{i}" for i in range(n))
q = f"((-2.0 * ({dot})) + xx) + cc"
res = con.execute(f"SELECT id, {dot} AS d, sqrt(greatest({q}, 0.0)) AS dist, ln(abs(x0)+1.0) AS l, exp(x1/1000.0) AS e, cos(x2*c2) AS co FROM t ORDER BY id").fetchall()
bad = dict(dot=0, dist=0, ln=0, exp=0, cos=0)
for (i, d, dist, l, e, co) in res:
    x, c = list(map(float, X[i])), list(map(float, C[i]))
    acc = x[0] * c[0]
    for k in range(1, n):
        acc = acc + x[k] * c[k]
    xx = float(abs(XX[i])); cc = xx * 0.7
    qq = ((-2.0 * acc) + xx) + cc
    bad["dot"] += d != acc
    bad["dist"] += dist != math.sqrt(max(qq, 0.0))
    bad["ln"] += l != math.log(abs(x[0]) + 1.0)
    bad["exp"] += e != math.exp(x[1] / 1000.0)
    bad["cos"] += co != math.cos(x[2] * c[2])
# fused version for contrast: would DuckDB match an FMA order? (math.fma, py>=3.13)
fma_match = 0
for (i, d, *_ ) in res:
    x, c = list(map(float, X[i])), list(map(float, C[i]))
    acc = x[0] * c[0]
    for k in range(1, n):
        acc = math.fma(x[k], c[k], acc)
    fma_match += d == acc
print("rows", rows, "mismatches vs python-ltr-unfused/glibc:", bad, "| rows where an FMA chain would also match:", fma_match)
print("literal type of -2.0:", con.execute("SELECT typeof(-2.0), typeof(0.0), typeof(-2.0e0)").fetchall())
