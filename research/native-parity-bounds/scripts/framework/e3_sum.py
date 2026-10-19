"""E3b: SUM(DOUBLE) over a native DuckDB table (row groups, parallel scan)."""
import math, numpy as np, duckdb, pyarrow as pa
rng = np.random.default_rng(5); N = 20_000_000
v = rng.normal(size=N) * 10.0 ** rng.uniform(-8, 8, N)
exact = math.fsum(v.tolist())
con = duckdb.connect()
con.register("a", pa.table({"v": v})); con.execute("CREATE TABLE big AS SELECT v FROM a"); con.unregister("a")
for th in [1, 2, 4]:
    con.execute(f"SET threads={th}")
    res = [con.execute("SELECT sum(v) FROM big").fetchone()[0] for _ in range(8)]
    fs = [con.execute("SELECT fsum(v) FROM big").fetchone()[0] for _ in range(4)]
    print(f"threads={th}: sum distinct {len(set(res))}, max rel err vs exact {max(abs(r - exact) for r in res) / abs(exact):.2e};"
          f" fsum distinct {len(set(fs))}, fsum rel err {max(abs(r - exact) for r in fs) / abs(exact):.2e}")
