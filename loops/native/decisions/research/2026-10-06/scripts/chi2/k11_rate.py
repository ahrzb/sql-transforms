"""numpy (SVML) log != glibc log, by |log x|, for draws with random low
mantissa bits (x = exp(t) * (1 + U(-2^-30, 2^-30))), against x = exp(t)
exactly (the record's log-uniform construction). 16M draws."""
import math, numpy as np, duckdb, pyarrow as pa
rng = np.random.default_rng(12)
def glibc_log(x):
    y = duckdb.sql("SELECT ln(x) AS y FROM t", ).arrow() if False else None
    t = pa.table({"x": x}); y = duckdb.sql("SELECT ln(x) AS y FROM t").arrow()
    y = y.read_all() if hasattr(y, "read_all") else y
    return y.column("y").to_numpy()
edges = [0, 1/16, 0.25, 1, 2, 4, 16, 64, 256, 745]
print("| |log x| | draws | rate, random mantissa | rate, x = exp(t) |")
print("|---|---|---|---|")
for lo, hi in zip(edges[:-1], edges[1:]):
    n = 2_000_000
    t = rng.uniform(lo, hi, n) * rng.choice([-1.0, 1.0], n)
    x0 = np.exp(t); x0 = x0[(x0 > 0) & np.isfinite(x0)]
    x1 = x0 * (1.0 + rng.uniform(-(2.0**-30), 2.0**-30, len(x0)))
    r1 = np.mean(np.log(x1) != glibc_log(x1)); r0 = np.mean(np.log(x0) != glibc_log(x0))
    print(f"| [{lo:g}, {hi:g}) | {len(x0):,} | {r1:.2e} | {r0:.2e} |", flush=True)
