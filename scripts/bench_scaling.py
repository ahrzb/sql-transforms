"""Batch-size scaling bench: the four serving engines across n (TASK-61).

Engines:
  columnar — fn.infer_arrow (arrow in/out; uses the columnar core when the
             program compiles for it)
  row      — fn.infer_rows with output='dict' (the row-at-a-time path)
  duckdb   — DuckDB itself per call: pre-built arrow table registered,
             executed, fetched as arrow (statics pre-materialized native)
  python   — the handcrafted per-row twin returning dicts

Writes benchmarks/scaling_results.json. Run on a RELEASE build only.

    .venv/Scripts/python scripts/bench_scaling.py [--quick]
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb  # noqa: E402

from benchmarks import serving_scenarios as sc  # noqa: E402

NS = [1, 8, 64, 256, 1024, 4096, 16384, 65536, 262144]


def p50(f, budget_ns: float = 2e9, max_iters: int = 400) -> float:
    """Median ns/call under a time budget (min 5 iters)."""
    ts = []
    t_total = 0.0
    while len(ts) < max_iters and (t_total < budget_ns or len(ts) < 5):
        t0 = time.perf_counter_ns()
        f()
        dt = time.perf_counter_ns() - t0
        ts.append(dt)
        t_total += dt
    return statistics.median(ts)


def bench_scenario(name: str, quick: bool) -> dict:
    mod = sc.load(name)
    statics = mod.make_statics(sc.SEED)
    fn = sc.build_spec_fn(mod, statics, output="dict")
    twin = mod.handcrafted(statics)
    model = sc.row_model(mod.ROW_SCHEMA)

    con = duckdb.connect()
    for sname, tbl in statics.items():
        con.register(f"__arrow_{sname}", tbl)
        sql = f'CREATE TABLE "{sname}" AS SELECT * FROM "__arrow_{sname}"'  # noqa: S608
        con.execute(sql)  # repo-trusted scenario names
        con.unregister(f"__arrow_{sname}")

    ns = NS[:6] if quick else NS
    out = {"scenario": name, "backend": fn.backend, "points": []}
    for n in ns:
        rows_d = mod.make_rows(sc.SEED + 1, n)
        rows_m = [model(**r) for r in rows_d]
        tbl = sc.rows_table(mod, rows_d)

        def duck_call(tbl=tbl):
            con.register("__THIS__", tbl)
            r = con.execute(mod.SQL).to_arrow_table()
            con.unregister("__THIS__")
            return r

        budget = 5e8 if n <= 1024 else 2e9
        fn.infer_arrow(tbl)
        fn.infer_rows(rows_m)
        [twin(r) for r in rows_d]
        duck_call()
        point = {
            "n": n,
            "columnar": p50(lambda tbl=tbl: fn.infer_arrow(tbl), budget),
            "row": p50(lambda rows_m=rows_m: fn.infer_rows(rows_m), budget),
            "python": p50(lambda rows_d=rows_d: [twin(r) for r in rows_d], budget),
            "duckdb": p50(duck_call, budget, max_iters=60),
        }
        out["points"].append(point)
        print(
            f"{name:14s} n={n:>7} columnar={point['columnar'] / 1e3:>9.0f}u "
            f"row={point['row'] / 1e3:>9.0f}u python={point['python'] / 1e3:>9.0f}u "
            f"duckdb={point['duckdb'] / 1e3:>9.0f}u",
            flush=True,
        )
    return out


def main():
    quick = "--quick" in sys.argv
    results = [bench_scenario(n, quick) for n in sc.NAMES]
    dst = Path(__file__).parent.parent / "benchmarks" / "scaling_results.json"
    dst.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
