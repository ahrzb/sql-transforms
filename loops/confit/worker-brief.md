# Confit worker brief

This file is the second part of every confit worker's opening prompt. It comes after
[`../worker-brief.md`](../worker-brief.md) and before the ticket's section of [`tickets.md`](tickets.md)
([`../workers.md`](../workers.md) §4). It holds only what the shared brief does not say. Send it verbatim.

---

You work on the confit engine in the package `packages/confit`. It is a Rust SQL specializer with Cranelift
and interpreter backends. It is built as a Python extension with maturin. It is checked against DuckDB 1.5.5
(optimizer off) as the oracle.

Before you write code, read `loops/confit/README.md` (sections "One cycle" and "Tickets and review") and
`loops/confit/goal.md`.

## What you must not touch
- Do not change `packages/sql-transform/` or `loops/native/`. The native catalog loop owns them.

## Build and test (repo root unless noted)
- Run the Rust tests with `cd packages/confit && uv run --no-sync cargo test`. It must run under uv so that
  it finds libpython.
- If Rust sources are newer than the .so file, the native guard in the tests rebuilds a DEBUG extension.
  Before you time anything or run campaigns, reinstall the release build with
  `uv sync --locked --group spark --reinstall-package confit`.

## Correctness discipline
- DuckDB (optimizer off) is the oracle. Parity tests use
  `fuzz.parity.assert_parity(sql, rows, statics=..., udfs=..., expect=..., trap=...)`
  from `packages/confit/fuzz/parity.py`. Never open a raw duckdb connection in a test. To probe DuckDB
  directly, use `confit.oracle.Oracle`.
- Measure DuckDB's behavior before you model it. Cite what you measured in comments and tests.
- If confit cannot reproduce a behavior exactly, it must REFUSE BY NAME (`unsupported: ...` or a bind
  error). It must never serve a different answer.
- If you move, share, drop or reorder a computation, every row that trapped before must still trap with the
  same class of error. No new trap may appear.
- Campaign: `cd packages/confit && uv run --no-sync python -m fuzz.runner --seed <S> --n <N> --workers 4`.
  Gated classes (DIVERGE_VALUE, DIVERGE_TRAP, DIVERGE_BUILD, ...) must be absent. DIVERGE_OPT is ungated.
  OPT_EMULATED waits for an owner ruling, so report it and do not fix it. Before you ask for review, run at
  least 10 000 seeds on a release build. Put the numbers in the PR.
- Generator changes (`fuzz/gen.py`) are seed-gated. Each change uses its own
  `random.Random(seed * 7919 + k)`, with an unused k and an unused modulus. Then no other seed's draws
  change.
- In the same PR, update `loops/confit/PLANS.md`. If a limitation changes, also update
  `packages/confit/docs/known-limitations.md`.
- Before you ask for review, merge `origin/master` into your branch (a merge commit). Then re-run the gate.
  Master changes fast. A branch that forked earlier can conflict, fail to compile, or undo a merged fix.

## PR
Say what changed and why. Give the measured before and after numbers and the tests that you added. Give the
gate and campaign results on the merged head.
