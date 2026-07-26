"""Specializer ns/call bench — the design doc §10 measurement discipline.

Reports p50/p99 ns per call at n in {1, 8, 64, 1024} for:
  cranelift  — the specializer's JIT backend through the generated row
               marshaller (the default production path)
  interp     — the same programs on the interpreter backend (the control;
               SPECIALIZER_FORCE_INTERP=1 in a subprocess), also marshalled
  generic    — the JIT backend behind the PRE-marshaller generic boundary
               (SPECIALIZER_GENERIC_BOUNDARY=1): per-cell getattr with
               fresh name strings, per-call buffers, model_validate per
               output row. generic vs cranelift on the noop case IS the
               marshaller's win (TASK-45 AC #2).
  native     — the existing DataFusion-semantics InferFn
  codegen    — the existing Python codegen engine

Run: `uv run python scripts/bench_specializer.py [--json out.json]`
The env-knob engines re-exec this script in a subprocess so the knob is
set before the module builds its functions.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import pyarrow as pa
from pydantic import create_model

NS = [1, 8, 64, 1024]
CASES = {
    "noop": "SELECT a FROM __THIS__",
    "arith+where": "SELECT a * 2 + 1 AS x, b / 2 AS h FROM __THIS__ WHERE a % 3 <> 0",
    "strings": "SELECT upper(s) || '-' || a AS t, substr(s, 2, 3) AS m FROM __THIS__",
    "join": "SELECT a, name FROM __THIS__ JOIN dim ON a = dim.id",
}


def rows_of(n):
    return [{"a": i % 60, "b": (i % 7) / 3.0, "s": f"item{i}"} for i in range(n)]


def quantiles(samples_ns):
    s = sorted(samples_ns)
    return s[len(s) // 2], s[min(len(s) - 1, int(len(s) * 0.99))]


def bench_fn(f, n):
    iters = max(30, min(3000, 3_000_000 // max(n, 1)))
    f()  # warm
    out = []
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        f()
        out.append(time.perf_counter_ns() - t0)
    return quantiles(out)


def build_specializer(sql, model, dim):
    from sql_transform._interpreter import DuckDBInferFn

    return DuckDBInferFn(
        sql, row_tables={"__THIS__": model}, static_tables={"dim": dim}
    )


def build_native(sql, model, dim):
    from sql_transform._interpreter import InferFn

    return InferFn(sql, row_tables={"__THIS__": model}, static_tables={"dim": dim})


def build_codegen(sql, model, dim):
    from sql_transform._codegen import CodegenFn

    return CodegenFn(sql, row_tables={"__THIS__": model}, static_tables={"dim": dim})


def main():
    model = create_model("Row", a=(int, ...), b=(float, ...), s=(str, ...))
    dim = pa.table({"id": list(range(60)), "name": [f"n{i}" for i in range(60)]})
    engine = os.environ.get("BENCH_ENGINE", "cranelift")

    builders = {
        "cranelift": build_specializer,
        "interp": build_specializer,
        "generic": build_specializer,
    }
    if engine in ("native", "codegen"):
        builders = {"native": build_native, "codegen": build_codegen}

    results = {}
    for case, sql in CASES.items():
        try:
            fn = builders[engine](sql, model, dim)
        except Exception as e:  # noqa: BLE001 -- engines differ in coverage
            results[case] = {"error": str(e)[:120]}
            continue
        if engine in ("cranelift", "interp", "generic"):
            want = "interpreter" if engine == "interp" else "cranelift"
            assert fn.backend == want, f"{case}: backend is {fn.backend}, wanted {want}"
            wantb = "generic" if engine == "generic" else "marshaller"
            assert fn.boundary == wantb, f"{case}: boundary {fn.boundary} != {wantb}"
        per_n = {}
        for n in NS:
            objs = [model(**r) for r in rows_of(n)]
            p50, p99 = bench_fn(lambda f=fn, o=objs: f.infer({"__THIS__": o}), n)
            per_n[n] = {"p50_ns": p50, "p99_ns": p99}
        results[case] = per_n
    print(json.dumps({engine: results}))


def orchestrate():
    """Run every engine (interp in a subprocess for the env knob), merge."""
    merged = {}
    for engine in ("cranelift", "interp", "generic", "native", "codegen"):
        env = os.environ.copy()
        env["BENCH_ENGINE"] = engine
        if engine == "interp":
            env["SPECIALIZER_FORCE_INTERP"] = "1"
        if engine == "generic":
            env["SPECIALIZER_GENERIC_BOUNDARY"] = "1"
        out = subprocess.run(  # noqa: S603 -- fixed argv
            [sys.executable, __file__, "--engine-run"],
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        merged.update(json.loads(out.stdout.strip().splitlines()[-1]))

    # Human table: p50 ns/call (p99 in parens), one row per case+engine.
    hdr = f"{'case':<12} {'engine':<10}" + "".join(f"{f'n={n}':>16}" for n in NS)
    print(hdr)
    print("-" * len(hdr))
    for case in CASES:
        for engine, per_case in merged.items():
            r = per_case.get(case)
            if r is None:
                continue
            if "error" in r:
                print(f"{case:<12} {engine:<10}  unsupported: {r['error'][:60]}")
                continue
            cells = "".join(
                f"{r[str(n)]['p50_ns'] if str(n) in r else r[n]['p50_ns']:>13,}ns"
                for n in NS
            )
            print(f"{case:<12} {engine:<10}{cells}")
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(merged, f, indent=2)
        print(f"wrote {path}")


if __name__ == "__main__":
    if "--engine-run" in sys.argv:
        main()
    else:
        orchestrate()
