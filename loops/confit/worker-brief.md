# Confit worker brief

The second part of every confit worker's opening prompt: after
[`../worker-brief.md`](../worker-brief.md), before the ticket's section of
[`tickets.md`](tickets.md) ([`../workers.md`](../workers.md) §4). It holds
only what the shared brief does not say. Send it verbatim.

---

You work on the confit engine: package `packages/confit`, a Rust SQL
specializer with Cranelift and interpreter backends, built as a Python
extension with maturin and checked against DuckDB 1.5.5 (optimizer off) as
the oracle.

Read `loops/confit/README.md` ("One cycle" and "Tickets and review") and
`loops/confit/goal.md` before writing code.

## What you must not touch
- `packages/sql-transform/` and `loops/native/`: the native catalog loop
  owns them.

## Build and test (repo root unless noted)
- Rust tests: `cd packages/confit && uv run --no-sync cargo test` (it must
  run under uv for libpython).
- The tests' native guard rebuilds a DEBUG extension when Rust sources are
  newer than the .so. Reinstall release
  (`uv sync --locked --group spark --reinstall-package confit`) before timing
  anything or running campaigns.

## Correctness discipline
- DuckDB (optimizer off) is the oracle. Parity tests use
  `fuzz.parity.assert_parity(sql, rows, statics=..., udfs=..., expect=..., trap=...)`
  from `packages/confit/fuzz/parity.py`. Never open a raw duckdb connection
  in a test; use `confit.oracle.Oracle` to probe DuckDB directly.
- Measure DuckDB's behaviour before modelling it, and cite what you measured
  in comments and tests.
- Anything confit cannot reproduce exactly must REFUSE BY NAME
  (`unsupported: ...` or a bind error), never serve a different answer.
- Where you move, share, drop or reorder a computation, every row that
  trapped before must still trap with the same class of error, and no new
  trap may appear.
- Campaign: `cd packages/confit && uv run --no-sync python -m fuzz.runner --seed <S> --n <N> --workers 4`.
  Gated classes (DIVERGE_VALUE, DIVERGE_TRAP, DIVERGE_BUILD, ...) must be
  absent; DIVERGE_OPT is ungated; OPT_EMULATED awaits an owner ruling
  (report it, don't fix it). Run at least 10k seeds on a release build
  before asking for review, and put the numbers in the PR.
- Generator changes (`fuzz/gen.py`) are seed-gated with their own
  `random.Random(seed * 7919 + k)` (an unused k and modulus), so no other
  seed's draws move.
- Update `loops/confit/PLANS.md` (and `packages/confit/docs/known-limitations.md`
  when a limitation changes) in the same PR.
- Before asking for review, merge `origin/master` into your branch (a merge
  commit) and re-run the gate: master moves fast, and a branch that forked
  earlier can conflict, fail to compile, or undo a merged fix.

## PR
Say what changed and why, measured before/after numbers, the tests added,
and the gate and campaign results (on the merged head).
