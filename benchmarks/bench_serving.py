"""Realistic serving-path bench — wide tables, involved feature engineering.

The scenarios (benchmarks/serving_scenarios/) reproduce the inference paths
of famous tabular-ML problems. Engines compared, p50/p99 ns per call at
n in {1, 8, 64, 1024}:

  spec     — the specializer: cranelift + generated marshaller (ours; dict
             rows out — the only output mode since the arrow schema API)
  interp   — interpreter backend, same marshaller (backend control;
             SPECIALIZER_FORCE_INTERP=1)
  generic  — cranelift behind the pre-marshaller boundary (previous
             boundary architecture; SPECIALIZER_GENERIC_BOUNDARY=1)
  duckdb   — DuckDB itself per call (statics pre-materialized as native
             tables; per call: Arrow batch from row dicts -> register ->
             execute -> fetch)
  python_dict — the handcrafted twin: what an engineer would hand-write
             for a microservice, returning plain dicts (the floor; the
             old typed-model "python" and "spec_dict" rows retired with
             the pydantic surface — dict IS the output now)

A three-way parity gate (specializer == DuckDB == handcrafted) runs before
any timing; a scenario that disagrees aborts the bench.

Run: uv run python -m benchmarks.bench_serving [--json out.json]
IMPORTANT: rebuild the wheel first (uv run --reinstall-package confit
python -c pass) — a stale wheel inflates ONLY the engine rows and once
produced a phantom 7x regression (caught by bisection, 2026-07-26). Debug
builds (the native test guard's `maturin develop` shadowing the release
wheel) are refused automatically before any timing.
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
    callers = {}

    if any(e in engines for e in ("spec", "interp", "generic")):
        fn = sc.build_spec_fn(mod, statics)
        for e in ("spec", "interp", "generic"):
            if e in engines:
                callers[e] = fn.infer_rows

    if "duckdb" in engines:
        duck = sc.duckdb_server(mod, statics)
        callers["duckdb"] = duck

    if "python_dict" in engines:
        hand = mod.handcrafted(statics)
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
                per[eng][n] = bench_call(lambda c=call, r=rows: c(r), n)
        results[mod.NAME] = per
    print(json.dumps(results))


def refuse_debug_build():
    """Abort unless the imported native extension is a release build.

    packages/confit/tests/_native_guard.py rebuilds the cwd-local .pyd via
    plain `maturin develop` (debug) whenever src/*.rs is newer; that shadows
    the venv's release wheel and silently inflates engine rows ~5x (measured
    2026-07-26).
    """
    import confit

    profile = getattr(confit, "BUILD_PROFILE", None)
    if profile != "release":
        sys.exit(
            f"bench_serving: refusing to time a non-release native build "
            f"(BUILD_PROFILE={profile!r}, loaded from {confit.__file__}).\n"
            f"Rebuild with: uv run maturin develop --release  (in packages/confit)"
        )


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
        "main": [
            "spec",
            "duckdb",
            "python_dict",
        ],
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

    order = [
        "spec",
        "interp",
        "generic",
        "duckdb",
        "python_dict",
    ]
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
    refuse_debug_build()
    if "--engine-run" in sys.argv:
        engine_run(os.environ["BENCH_ENGINES"].split(","))
    else:
        orchestrate()
