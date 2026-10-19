import duckdb, random, struct
con = duckdb.connect()
print("duckdb", duckdb.__version__)
print("fma-like functions:", con.execute("select distinct function_name from duckdb_functions() where function_name ilike '%fma%' or function_name ilike '%fused%' or function_name ilike '%muladd%'").fetchall())
rng = random.Random(4242)
vals = []
for _ in range(3000):
    # random doubles across magnitudes, plain-decimal reprs (no exponent) and exponent reprs
    e = rng.uniform(-20, 20)
    vals.append(rng.choice([-1, 1]) * 10 ** e * rng.random())
bad_bare = bad_str = bare_dec = 0; n_noexp = 0; bad_noexp = 0
for v in vals:
    t = repr(v)
    a = con.execute(f"select {t}::DOUBLE").fetchone()[0]
    b = con.execute(f"select '{t}'::DOUBLE").fetchone()[0]
    bad_bare += a != v; bad_str += b != v
    if "e" not in t:
        n_noexp += 1; bad_noexp += a != v
print(f"bare numeral wrong {bad_bare}/{len(vals)} (no-exponent reprs: {bad_noexp}/{n_noexp}); string wrong {bad_str}/{len(vals)}")
print(con.execute("select typeof(0.1234567890123456), typeof(1.5e-7), typeof(123456789012345678901234567890123456789.5)").fetchall())
