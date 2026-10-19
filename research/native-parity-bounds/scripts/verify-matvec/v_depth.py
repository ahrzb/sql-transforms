import duckdb, numpy as np
con = duckdb.connect()
print(con.execute("select current_setting('max_expression_depth')").fetchall())
for n in (500, 999, 1100, 2048):
    e = "x0*'0.5'::DOUBLE"
    for i in range(1, n): e = f"({e} + x{i % 50}*'0.5'::DOUBLE)"
    cols = ", ".join(f"1.0::DOUBLE AS x{i}" for i in range(50))
    try:
        print(n, con.execute(f"SELECT {e} FROM (SELECT {cols})").fetchall())
    except Exception as ex:
        print(n, type(ex).__name__, str(ex)[:120])
