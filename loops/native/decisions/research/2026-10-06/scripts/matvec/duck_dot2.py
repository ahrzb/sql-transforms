"""Can DuckDB spell a compensated dot product (Ogita-Rump-Oishi Dot2, TwoProduct by
Dekker/Veltkamp splitting, no FMA), bit-exactly as specified, and at what cost
against the entry's left-to-right dot? n = 32 features, one lane.

Dot2: p = x1*c1, s = err(x1*c1); for i >= 2: h = xi*ci, r = err(xi*ci),
      (p, q) = TwoSum(p, h), s = s + (q + r).  result = p + s.
"""

import math
import time

import duckdb
import numpy as np

N, ROWS = 32, 200_000
SPLIT = 134217729.0  # 2^27 + 1
rng = np.random.default_rng(7)
c = rng.standard_normal(N)
c /= np.linalg.norm(c)
mean = rng.uniform(-100, 100, N) * rng.choice([1, 1e3], N)
X = mean + rng.standard_normal((ROWS, N)) * np.where(rng.random((ROWS, 1)) < 0.5, 1e-6, 1.0)


def split(a):
    t = SPLIT * a
    h = t - (t - a)
    return h, a - h


ch, cl = zip(*(split(float(v)) for v in c))


def f(v):
    # through a string, as confit does (confit/sql.py:276-285): a bare numeral is a
    # DECIMAL first and rounds twice
    return "'" + repr(float(v)) + "'::DOUBLE"


def two_prod_err(xh, xl, x, i):
    return f"({xl}*{f(cl[i])} - ((( {x}*{f(c[i])} - {xh}*{f(ch[i])}) - {xl}*{f(ch[i])}) - {xh}*{f(cl[i])}))"


con = duckdb.connect()
con.execute("SET threads = 1")
import pandas as pd  # noqa: E402

df = pd.DataFrame(X, columns=[f"x{i}" for i in range(N)])
con.register("df", df)
con.execute("CREATE TABLE t AS SELECT * FROM df")

ltr = f"x0*{f(c[0])}"
for i in range(1, N):
    ltr = f"({ltr} + x{i}*{f(c[i])})"
q_ltr = f"SELECT {ltr} AS d FROM t"

# Dot2 as projections: L1 split and products; L2.. prefix sums p_i (one level each);
# last level: TwoSum errors q_i, TwoProduct errors r_i, s accumulated in order.
cols = ", ".join(f"x{i}" for i in range(N))
l1 = ", ".join(
    f"x{i}, (({f(SPLIT)}*x{i}) - (({f(SPLIT)}*x{i}) - x{i})) AS xh{i}, x{i}*{f(c[i])} AS h{i}" for i in range(N)
)
q = f"SELECT {l1} FROM t"
q = f"SELECT *, " + ", ".join(f"x{i} - xh{i} AS xl{i}" for i in range(N)) + f" FROM ({q})"
q = f"SELECT *, h0 AS p0 FROM ({q})"
for i in range(1, N):
    q = f"SELECT *, p{i - 1} + h{i} AS p{i} FROM ({q})"
r = [two_prod_err(f"xh{i}", f"xl{i}", f"x{i}", i) for i in range(N)]
qs = [None] + [
    f"((p{i - 1} - (p{i} - (p{i} - p{i - 1}))) + (h{i} - (p{i} - p{i - 1})))" for i in range(1, N)
]
s = r[0]
for i in range(1, N):
    s = f"({s} + ({qs[i]} + {r[i]}))"
q_dot2 = f"SELECT p{N - 1} + {s} AS d FROM ({q})"


def py_dot2(x):
    def tp(a, i):
        p = a * c[i]
        ah, al = split(a)
        return p, al * cl[i] - (((p - ah * ch[i]) - al * ch[i]) - ah * cl[i])

    p, s_ = tp(x[0], 0)
    for i in range(1, N):
        h, rr = tp(x[i], i)
        pn = p + h
        z = pn - p
        qq = (p - (pn - z)) + (h - z)
        p = pn
        s_ = s_ + (qq + rr)
    return p + s_


def exact(x):
    parts = []
    for a, b in zip(x, c):
        p = a * b
        parts += [p, math.fma(a, b, -p)]
    return math.fsum(parts)


def timeit(qq):
    best = 1e9
    for _ in range(3):
        t0 = time.perf_counter()
        out = con.execute(qq).fetchnumpy()["d"]
        best = min(best, time.perf_counter() - t0)
    return best, out


t_ltr, d_ltr = timeit(q_ltr)
t_d2, d_d2 = timeit(q_dot2)
print(f"rows {ROWS}, n {N}, threads 1")
print(f"LTR  sql {len(q_ltr):7d} chars  {t_ltr * 1e3:8.1f} ms")
print(f"Dot2 sql {len(q_dot2):7d} chars  {t_d2 * 1e3:8.1f} ms   ratio {t_d2 / t_ltr:.1f}x")

sample = range(0, ROWS, 40)  # 5,000 rows checked in Python
xs = [X[i].tolist() for i in sample]
ltr_py = []
for x in xs:
    acc = x[0] * c[0]
    for i in range(1, N):
        acc = acc + x[i] * c[i]
    ltr_py.append(acc)
d2_py = [py_dot2(x) for x in xs]
ex = [exact(x) for x in xs]
idx = list(sample)
print("DuckDB LTR == Python LTR:", sum(d_ltr[i] == v for i, v in zip(idx, ltr_py)), "/", len(idx))
print("DuckDB Dot2 == Python Dot2:", sum(d_d2[i] == v for i, v in zip(idx, d2_py)), "/", len(idx))
print("Dot2 == correctly rounded x.c:", sum(v == e for v, e in zip(d2_py, ex)), "/", len(idx))
print("LTR  == correctly rounded x.c:", sum(v == e for v, e in zip(ltr_py, ex)), "/", len(idx))
S0 = [sum(abs(a * b) for a, b in zip(x, c)) for x in xs]
eps = 2.0**-52
print("max |Dot2 - exact| / (eps*S0):", max(abs(v - e) / (eps * s_) for v, e, s_ in zip(d2_py, ex, S0)))
print("max |LTR  - exact| / (eps*S0):", max(abs(v - e) / (eps * s_) for v, e, s_ in zip(ltr_py, ex, S0)))
print("max |Dot2 - exact| in ulps of exact:", max(abs(v - e) / math.ulp(e) for v, e in zip(d2_py, ex) if e))

