"""Differential fuzzer for confit (spec: 2026-08-11-differential-fuzzer-design).

Seeded, stdlib-only. `gen.gen(seed)` builds a random Case (schema + data +
query AST); `oracle.run_case(case)` answers with a Verdict against DuckDB and
both confit backends; `runner` drives campaigns in crash-isolated workers;
`shrink` minimizes a finding. The repro for anything is its seed.
"""
