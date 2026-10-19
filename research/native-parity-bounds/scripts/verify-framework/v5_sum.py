import math, numpy as np, duckdb, os
print("cpus", os.cpu_count(), duckdb.__version__)
con = duckdb.connect()
con.execute("CREATE TABLE big AS SELECT (random() - 0.5) * pow(10, random() * 16 - 8) AS v FROM range(16000000)")
v = con.execute("SELECT v FROM big").fetchnumpy()["v"]; exact = math.fsum(v.tolist())
for th in [1, 4, 8]:
    con.execute(f"SET threads={th}")
    s = [con.execute("SELECT sum(v) FROM big").fetchone()[0] for _ in range(8)]
    f = [con.execute("SELECT fsum(v) FROM big").fetchone()[0] for _ in range(8)]
    k = [con.execute("SELECT kahan_sum(v) FROM big").fetchone()[0] for _ in range(4)]
    print(f"threads={th}: sum distinct {len(set(s))}, fsum distinct {len(set(f))}, kahan_sum distinct {len(set(k))}, fsum==exact {sum(x == exact for x in f)}/8, max fsum rel err {max(abs(x-exact) for x in f)/abs(exact):.2e}")
