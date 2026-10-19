import math, pickle, numpy as np, duckdb, pyarrow as pa
d = pickle.load(open("v1_default.pkl", "rb")); L = pickle.load(open("v1_lanes.pkl", "rb"))
con = duckdb.connect(); lit = lambda v: f"CAST('{float(v)!r}' AS DOUBLE)"
off = 0; chk = bad = 0; byr = {}
for l in L: byr.setdefault(l["r"], []).append(l)
for r, ((key, x), M) in enumerate(zip(d["recs"], d["M"])):
    if r % 2: continue
    est = d["ests"][key]; C = est.components_; sc = np.sqrt(est.explained_variance_); sc[sc < np.finfo(float).eps] = np.finfo(float).eps
    con.register("t", pa.table({f"x{i}": [x[i]] for i in range(len(x))}))
    ex = [f"((({' + '.join(f'x{i} * {lit(C[j, i])}' for i in range(len(x)))}) - {lit(M[j])}) / {lit(sc[j])})" for j in range(C.shape[0])]
    got = con.execute("SELECT " + ", ".join(ex) + " FROM t").fetchone()
    for g, l in zip(got, byr[r]):
        chk += 1; e = l["ent"]
        if not ((g == e and math.copysign(1, g) == math.copysign(1, e)) or (math.isnan(g) and math.isnan(e))): bad += 1
    con.unregister("t")
print("DuckDB full lane vs Python l2r entry: checked", chk, "differ", bad)
