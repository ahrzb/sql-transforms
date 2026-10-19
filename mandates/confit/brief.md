# Confit engine: brief

## Standing instructions

### Terms

- The oracle is DuckDB 1.5.5 with the optimizer off; `confit.oracle.Oracle` runs it.
- OPT_EMULATED is the verdict where confit matches the optimizer-on reading and the optimizer-off reading differs.
- Gated verdicts (DIVERGE_VALUE, DIVERGE_TRAP, DIVERGE_BUILD and the rest of `fuzz.runner.GATED`) fail a campaign; DIVERGE_OPT does not.

### One task

A task has one concern, one branch and one pull request, and fits one sitting of review.

1. **Sync.** Fetch `master` and branch from it as `claude/confit-<name>`.
   List open confit pull requests; give each a tracker task or close it with a reason.
2. **Pick,** in this order, and claim the task in the [tracker](tracker.md):
   1. correctness: bugs and live divergences from a campaign or the nightly;
   2. the tracker's **Needs** from the native catalog, ranked by measured build or serving cost and the families they affect, never by whether the catalog's caps hide the cost; reproduce each need first;
   3. the query classes, in the order of [next query classes](../../proposals/closed/next-query-classes.md), one slice a task, each slice served and tested;
   4. performance and engineering tasks.
3. **Measure DuckDB first** with `confit.oracle.Oracle`, optimizer off. Model what you measured and cite it in comments and tests.
   If confit cannot reproduce a behaviour exactly, refuse it by name (`unsupported: ...` or a bind error); never serve a different answer.
4. **Implement** with parity tests: `fuzz.parity.assert_parity(sql, rows, statics=..., udfs=..., expect=..., trap=...)` from `packages/confit/fuzz/parity.py`. Never open a raw DuckDB connection in a test.
   - If you move, share, drop or reorder a computation, every row that trapped before still traps with the same class of error, and no new trap appears.
   - A generator change (`fuzz/gen.py`) is seed-gated with its own `random.Random(seed * 7919 + k)`, using an unused k and modulus, so no other seed's draws change.
5. **Check.**
   - The gate: `uv run --no-sync python scripts/gate.py`.
   - Lint: `git add -A`, then `uv run --no-sync pre-commit run --all-files`; read the whole output.
   - A campaign of 10,000 to 20,000 seeds on a release build: `cd packages/confit && uv run --no-sync python -m fuzz.runner --seed S --n N --workers 4`. No gated class may appear. Report OPT_EMULATED; do not fix it.
   - For a change under `packages/confit/src/specializer/` (frontend, plan, share, lower, lets, exec), the serial cost tests: `uv run --no-sync python -m pytest -m cost -p no:xdist`. Paste their output.
6. **Ship.** Merge `origin/master` into the branch, build, and re-run the gate there. Open the pull request with the **Review** section, wait for CI, and squash-merge with the expected head SHA.
7. **Afterwards.** Delete the task's tracker line and description in that pull request. Update `packages/confit/docs/known-limitations.md` when a limitation changes.
   If the merge serves a need, delete it from **Needs** and tell the native catalog's session.
   If the merge changes the catalog's build or serving costs, add a need to the [native tracker](../native/tracker.md) naming the estimates it invalidates.

### Standing duties

- **Nightly triage.** Read the latest report on the issue "Nightly campaign findings". Re-run each gated seed on master. Fix each live class, or pin it as a strict expected failure in `packages/confit/tests/test_open_divergences.py`. Re-run the metamorphic seeds too.
- **Questions.** If a choice concerns an OPT_EMULATED class, an exclusion or the oracle contract, file a question with the evidence; do not settle it in code.

### Review

In the **Review** section, check: DuckDB's behaviour was measured, not assumed; tests are parity tests; each unreproduced behaviour refuses by name; trap semantics are kept; before and after numbers for performance work; the gate and a campaign of at least 10,000 seeds with no gated class, on the merged head; for a generator change, a seed-gated run counted by verdict; the cost tests' output for a specializer change; `known-limitations.md` updated; tests pin the new boundary.
A known downstream cost risk found in review blocks the merge until it is measured.

### Files that tasks share

These files under `packages/confit/src/specializer/` collide most: `frontend/expr.rs`, `frontend/typing.rs`, `frontend/functions.rs`, `frontend/calls.rs`, `plan.rs` (`can_trap`, `trap_skeleton`), `lower.rs` and `exec/cranelift.rs`.

- Two tasks in one of them can run together if they touch different functions; merge the smaller first.
- A new `Lit`, `Ty` or `SKind` variant breaks every exhaustive match on another open branch. Name it in the task; after the merge, check each open branch against master.

### Build

- Run the Rust tests with `cd packages/confit && uv run --no-sync cargo test`, under uv so that it finds libpython.
- Before timing anything or running a campaign, reinstall the release build: `uv sync --locked --group spark --reinstall-package confit`.

### Helper procedure

A runbook helper does steps 3 to 6 of **One task** for the task its assignment names, on that task's branch, then opens the pull request and stops at green CI. It answers each review comment with a push or a reply that gives its reason. It never merges.

## Traps

- Stale base: a branch forked before another merge can conflict, fail to compile, or undo that merge; merge master locally and run the gate on the result before merging. (first wave of tasks, #349, #358, #363)
- Debug extension: when Rust sources are newer than the extension, the tests' native guard rebuilds a debug build, and timings and campaigns then measure it; reinstall the release build first. ([confit brief history](https://github.com/ahrzb/sql-transforms/blob/40f6ab17/loops/confit/worker-brief.md))
- Downstream cost: a change measured only on the shapes it optimizes can slow the native catalog elsewhere; #363 slowed a three-instance QuantileTransformer from 25× to 1.4× its twin, and #456 made field-by-field struct reads quadratic. Run the cost tests. (native reports 2 and 6)
- Cold container: a measurement in a cold container once reported a false regression; repeat in a warm process before reporting one. (#383)
