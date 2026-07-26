"""Realistic serving-path bench — wide tables, involved feature engineering.

The scenarios (benchmarks/serving_scenarios/) reproduce the inference paths
of famous tabular-ML problems. Engines compared, p50/p99 ns per call at
n in {1, 8, 64, 1024}:

  spec     — the specializer: cranelift + generated marshaller (ours)
  interp   — interpreter backend, same marshaller (backend control;
             SPECIALIZER_FORCE_INTERP=1)
  generic  — cranelift behind the pre-marshaller boundary (previous
             boundary architecture; SPECIALIZER_GENERIC_BOUNDARY=1)
  native   — the previous DataFusion-semantics InferFn
  codegen  — the previous Python-codegen engine
  duckdb   — DuckDB itself per call (statics pre-materialized as native
             tables; per call: Arrow batch from row dicts -> register ->
             execute -> fetch)
  python   — the handcrafted twin: what an engineer would hand-write for a
             microservice, returning the same typed output models
  python_dict — same, returning plain dicts (the absolute floor; JSON only)

A three-way parity gate (specializer == DuckDB == handcrafted) runs before
any timing; a scenario that disagrees aborts the bench.

Run: uv run python -m benchmarks.bench_serving [--json out.json]
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from benchmarks import serving_scenarios as sc

NS = [1, 8, 64, 1024]

ENGINE_ENV = {
    "spec": {},
    "interp": {"SPECIALIZER_FORCE_INTERP": "1"},
    "generic": {"SPECIALIZER_GENERIC_BOUNDARY": "1"},
}


def scenario_names():
    """All scenarios, or a comma-separated BENCH_SCENARIOS subset."""
    sub = os.environ.get("BENCH_SCENARIOS")
    return sub.split(",") if sub else sc.NAMES


def quantiles(samples):
    s = sorted(samples)
    return s[len(s) // 2], s[min(len(s) - 1, int(len(s) * 0.99))]


def bench_call(f, n):
    iters = max(30, min(2000, 2_000_000 // max(n, 1)))
    f()  # warm
    out = []
    # 3s budget per cell (min 30 samples) — keeps the ms-scale engines
    # (duckdb per call) from dominating wall time.
    deadline = time.perf_counter_ns() + 3_000_000_000
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        f()
        out.append(time.perf_counter_ns() - t0)
        if len(out) >= 30 and time.perf_counter_ns() > deadline:
            break
    p50, p99 = quantiles(out)
    return {"p50_ns": p50, "p99_ns": p99}


def build_callers(mod, engines):
    """-> {engine: callable(rows) or None if the engine can't serve it}."""
    statics = mod.make_statics(sc.SEED)
    model = sc.row_model(mod.ROW_SCHEMA)
    callers = {}

    if any(e in engines for e in ("spec", "interp", "generic")):
        fn = sc.build_spec_fn(mod, statics)
        for e in ("spec", "interp", "generic"):
            if e in engines:
                callers[e] = fn.infer_rows

    for eng, cls_path in (
        ("native", "sql_transform._interpreter.InferFn"),
        ("codegen", "sql_transform._codegen.CodegenFn"),
    ):
        if eng not in engines:
            continue
        mod_path, cls_name = cls_path.rsplit(".", 1)
        cls = getattr(__import__(mod_path, fromlist=[cls_name]), cls_name)
        try:
            f = cls(mod.SQL, row_tables={"__THIS__": model}, static_tables=statics)
            callers[eng] = lambda rows, f=f: f.infer({"__THIS__": rows})
        except Exception as e:  # noqa: BLE001 -- engines differ in coverage
            callers[eng] = str(e)[:140]

    if "duckdb" in engines:
        duck = sc.duckdb_server(mod, statics)
        callers["duckdb"] = duck

    if "python" in engines or "python_dict" in engines:
        hand = mod.handcrafted(statics)
        out_model = sc.build_spec_fn(mod, statics).output_model
        if "python" in engines:
            callers["python"] = lambda rows: [out_model(**hand(r)) for r in rows]
        if "python_dict" in engines:
            callers["python_dict"] = lambda rows: [hand(r) for r in rows]

    return callers


def engine_run(engines):
    results = {}
    for mod in map(sc.load, scenario_names()):
        callers = build_callers(mod, engines)
        per = {}
        for eng, call in callers.items():
            if isinstance(call, str):
                per[eng] = {"error": call}
                continue
            per[eng] = {}
            for n in NS:
                rows = mod.make_rows(sc.SEED + 2, n)
                # Engines take model objects on their classic surface;
                # spec-family and duckdb/python take dicts natively. Feed
                # each what its real caller would: dict rows for
                # spec/duckdb/python, model objects for native/codegen
                # (their only supported input shape).
                if eng in ("native", "codegen"):
                    model = sc.row_model(mod.ROW_SCHEMA)
                    rows = [model(**r) for r in rows]
                per[eng][n] = bench_call(lambda c=call, r=rows: c(r), n)
        results[mod.NAME] = per
    print(json.dumps(results))


def orchestrate():
    print("parity gate:", flush=True)
    for mod in map(sc.load, scenario_names()):
        problems = sc.verify_parity(mod)
        tag = "ok" if not problems else "FAIL"
        print(f"  {mod.NAME:<14} {tag}")
        if problems:
            print("\n".join("    " + p for p in problems))
            sys.exit(1)

    groups = {
        "main": ["spec", "native", "codegen", "duckdb", "python", "python_dict"],
        "interp": ["interp"],
        "generic": ["generic"],
    }
    merged: dict = {}
    for group, engines in groups.items():
        env = os.environ.copy()
        env.update(ENGINE_ENV.get(group if group != "main" else "spec", {}))
        env["BENCH_ENGINES"] = ",".join(engines)
        out = subprocess.run(  # noqa: S603 -- fixed argv
            [sys.executable, "-m", "benchmarks.bench_serving", "--engine-run"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        for scenario, per in json.loads(out.stdout.strip().splitlines()[-1]).items():
            merged.setdefault(scenario, {}).update(per)

    order = ["spec", "interp", "generic", "native", "codegen", "duckdb", "python"]
    hdr = f"{'scenario':<14} {'engine':<9}" + "".join(f"{f'n={n}':>15}" for n in NS)
    print("\n" + hdr)
    print("-" * len(hdr))
    for scenario, per in merged.items():
        for eng in order:
            r = per.get(eng)
            if r is None:
                continue
            if "error" in r:
                print(f"{scenario:<14} {eng:<9}  unsupported: {r['error'][:70]}")
                continue
            cells = "".join(f"{r[str(n)]['p50_ns']:>13,}ns" for n in NS)
            print(f"{scenario:<14} {eng:<9}{cells}")
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=1)
        print(f"wrote {path}")


if __name__ == "__main__":
    if "--engine-run" in sys.argv:
        engine_run(os.environ["BENCH_ENGINES"].split(","))
    else:
        orchestrate()
