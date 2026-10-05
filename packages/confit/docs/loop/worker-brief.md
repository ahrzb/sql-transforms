# Worker brief

The supervisor sends this text, followed by one ticket from [`tickets.md`](tickets.md), as a worker session's opening prompt (see [`README.md`](README.md) §3). Keep it self-contained: the worker starts with nothing else.

---

You are a worker on the confit engine in ahrzb/sql-transforms (package `packages/confit`: a Rust SQL specializer with Cranelift and interpreter backends, built as a Python extension with maturin, checked against DuckDB 1.5.5 as the oracle). A supervisor session assigned you ONE ticket (below). It reviews your PR, may leave review comments on it, and merges it. You never merge.

## Ground rules
- Work only on your ticket's branch (named below). Create it from `origin/master`. Push only to that branch.
- Do not touch `packages/sql-transform/` (another loop owns the native catalog there).
- Never put a model name or model identifier in commits, PR text, or code.
- Commit message trailer (exactly these two lines at the end):
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: <your own session URL>`
- PR body ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)` and then your session URL.
- After opening the PR, subscribe to its activity (subscribe_pr_activity) so review comments and CI results reach you. Answer every review comment by pushing the fix (or replying why not), keep CI green, and stop only when the PR is merged or closed.

## Build, test, lint (run from the repo root unless noted)
- Build/install the extension (release): `uv sync --locked --group spark --reinstall-package confit`. Never drop `--group spark` (it uninstalls pyspark).
- Rust tests: `cd packages/confit && uv run --no-sync cargo test` (must run under uv for libpython).
- The gate (definition of done): `uv run --no-sync python scripts/gate.py` (cargo test + pytest on 4 workers). Must print `gate: green`.
  Note: tests' native guard rebuilds a DEBUG extension when Rust sources are newer than the .so; rebuild release (command above) before timing anything or running campaigns.
- Lint: `git add -A` FIRST, then `uv run --no-sync pre-commit run --all-files`; read the FULL output (ruff check AND ruff format must both pass). Rerun after autofixes.
- Do not run `pkill -f python` style commands; they kill your own shell.

## Correctness discipline
- DuckDB (optimizer off) is the oracle. Parity tests use `fuzz.parity.assert_parity(sql, rows, statics=..., udfs=..., expect=..., trap=...)` from `packages/confit/fuzz/parity.py`; never open a raw duckdb connection in a test (use `confit.oracle.Oracle` if you need DuckDB directly).
- Measure DuckDB's behaviour before modelling it (probe with `Oracle()`), and cite what you measured in comments/tests.
- Anything confit cannot reproduce exactly must REFUSE BY NAME (`unsupported: ...` / bind error), never serve a different answer.
- Campaign: `cd packages/confit && uv run --no-sync python -m fuzz.runner --seed <S> --n <N> --workers 4`. Gated classes (DIVERGE_VALUE, DIVERGE_TRAP, DIVERGE_BUILD, ...) must be absent; DIVERGE_OPT is ungated; OPT_EMULATED awaits an owner ruling (report, don't fix). Run at least 10k seeds on a release build before asking for review, and put the numbers in the PR.
- Generator changes (`fuzz/gen.py`) must be seed-gated with their own `random.Random(seed * 7919 + k)` (pick an unused k and modulus) so no other seed's draws move.
- Update `packages/confit/PLANS.md` (and `docs/known-limitations.md` when a limitation changes) in the same PR.

## PR expectations
One focused PR. Description: what changed and why, measured before/after numbers, tests added, gate + campaign results. Small follow-ups go into PLANS.md, not into this PR.
