"""E3: (1) DuckDB evaluates the entry's `x0*c0 + x1*c1 + ...` exactly as
Python's left-to-right (no FMA, no reassociation); (2) DuckDB's parallel
SUM(DOUBLE) across thread counts and runs; fsum/kahan_sum."""
import math, pickle, numpy as np, duckdb, pyarrow as pa
cases = pickle.load(open("pca_cases.pkl", "rb"))
con = duckdb.connect()
checked = mism = 0
for seed, est, x in cases[:1500]:
    C = est.components_; k, n = C.shape
    tbl = pa.table({f"x{i}": [x[i]] for i in range(n)})
    con.register("t", tbl)
    exprs, py = [], []
    for j in range(k):
        c = [float(v) for v in C[j]]
        exprs.append(" + ".join(f"x{i} * CAST('{c[i]!r}' AS DOUBLE)" for i in range(n)))
        acc = x[0] * c[0]
        for i in range(1, n): acc = acc + x[i] * c[i]
        py.append(acc)
    got = con.execute("SELECT " + ", ".join(f"({e}) AS l{j}" for j, e in enumerate(exprs)) + " FROM t").fetchone()
    for a, b in zip(got, py):
        checked += 1
        if not (a == b or (math.isnan(a) and math.isnan(b))) or (a == 0 and math.copysign(1, a) != math.copysign(1, b)):
            mism += 1
    con.unregister("t")
print(f"DuckDB left-to-right lanes checked {checked}, differ from Python left-to-right: {mism}")

rng = np.random.default_rng(5)
N = 20_000_000
v = rng.normal(size=N) * 10.0 ** rng.uniform(-8, 8, N)
con.register("big", pa.table({"v": v}))
exact = math.fsum(v.tolist())
for th in [1, 2, 4]:
    con.execute(f"SET threads={th}")
    res = [con.execute("SELECT sum(v) FROM big").fetchone()[0] for _ in range(6)]
    fs = [con.execute("SELECT fsum(v) FROM big").fetchone()[0] for _ in range(3)]
    print(f"threads={th}: sum distinct results {len(set(res))} {sorted(set(float(r).hex() for r in res))[:3]}; "
          f"rel err vs fsum(math) {max(abs(r - exact) for r in res) / abs(exact):.2e}; fsum distinct {len(set(fs))}, fsum==math.fsum {all(f == exact for f in fs)}")
print("math.fsum", exact.hex(), "numpy pairwise sum", float(np.sum(v)).hex())
print(duckdb.__version__)
