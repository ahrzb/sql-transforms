import sys, numpy as np, duckdb, pyarrow as pa
rng = np.random.default_rng(5)
x = np.concatenate([rng.lognormal(0, 3, 300000), np.round(rng.lognormal(3.5, 1.2, 200000), 2) + 0.01])
w = rng.uniform(-30, 30, 500000)
con = duckdb.connect(); con.register("t", pa.table({"x": x, "w": w}))
r = con.execute("SELECT ln(x) AS l, exp(w) AS e, cos(w) AS c, sin(w) AS s FROM t").fetchnumpy()
np.savez(sys.argv[1], **{k: np.asarray(v) for k, v in r.items()})
