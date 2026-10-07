# Linux verification resolves the Windows gate blocker

## Original question

May the cutover include a narrow Confit repair, or must the repair remain separate?
The approved plan excludes changes to the Confit engine.
The repository requires a green gate before a push.

## Completed cutover

The local branch is `simplify-sql-transform`.
The first commit replaces the parallel authoring implementations with one compositional model and updates the current specification.
The second commit moves the modules from `sql_transform/model/` to `sql_transform/` and updates imports and paths.
Neither commit changes Confit's Rust engine sources or its tests.
A third commit removes stored plans and reviews comments.
That review changes no executable Python structure or Rust statements.
The owner requested removal of draft and task labels.
This includes labels in the otherwise protected helper for user-defined functions (UDFs).

The system specification now organizes current behavior by topic.
Its entry point is [`docs/specs/README.md`](../../../../docs/specs/README.md).
The authoring chapter is [`contract.md`](../../../../packages/sql-transform/spec/README.md).
These chapters define fit/request SQL, composition, supported interfaces, fitted artifacts, and refusals.
Other linked chapters define Confit serving, the oracle comparison contract, native conversion, and verification controls.

## Windows verification before updating the branch

- The complete authoring-package run before moving modules passed 3831 tests, with 5 skips and 3 expected failures.
- Selected authoring contracts after moving modules passed 331 tests, with 1 expected failure.
- The original-query differential compared fitted execution with original SQL and passed all 1500 seeded cases.
- SHA-256 fingerprints of file content did not change for any of the 44 protected files.
- The gate's Rust engine tests passed 309 tests, with 1 ignored test.
- The Windows Python run reported 7261 passing tests, 15 skips, and 3 expected failures.
- Two parallel test processes crashed, so that run did not pass the complete gate.
- Spark execution agreed on 260 of 678 queries. It refused the remaining 418 cleanly, with no failures.

The complete package command was `uv run --no-sync pytest -q packages/sql-transform/sql_transform`.
The gate entry point is `uv run --no-sync python scripts/gate.py`.
The missing dependency was pytest-xdist, which supplies parallel test execution.
After installing it, the session resumed `uv run --no-sync python -m pytest -q -n 4`.
The `-n 4` option starts four test processes.
Java was configured through `JAVA_HOME` so the Spark execution comparison ran.
Skipped and ignored tests did not execute. Expected failures are cases that pytest already marks as known failures.

The protected files are the UDF/tree helpers, native catalog sources and tests, native docs, the native loop, and its benchmark.
The owner's untracked `.jupyter_ystore.db` database and `scripts/fuzz/` directory remain untouched.

The two worker crashes occurred in unchanged Confit tests:

1. [`test_expression_depth.py`](../../../../packages/confit/tests/test_expression_depth.py), in `test_a_very_deep_expression_refuses_without_walking_it`, reported a Windows stack overflow.
2. [`test_fuzz_smoke.py`](../../../../packages/confit/tests/test_fuzz_smoke.py), in `test_verdicts_cover_the_contract_and_reproduce`, lost its worker.

An isolated follow-up reproduced the depth crash. The fuzz test passed in that follow-up.
The query is `SELECT x + x + ... AS o FROM __THIS__`, with 50,000 terms.
The expected outcome from Confit's public constructor, `DuckDBInferFn`, is the refusal `Max expression depth limit of 1000 exceeded`.
The actual outcome on the default Windows stack is call-stack exhaustion.
Each thread reserves call-stack memory for active function calls.
A diagnostic thread with 64 MiB (mebibytes) of that memory returned the expected refusal.
That diagnostic did not change the gate command or repair its crash.

The installed Confit extension uses an optimized release build, not a debug build.
The installed extension and the current compiled release artifact have the same SHA-256 content fingerprint.
The comparison confirms that the installed extension contains the compiled release build used by the diagnostic.

## Options considered before the ruling

1. **Allow a narrow Confit repair in this PR.** Fix the default-stack failure without weakening the depth test, then run the gate again. This expands the approved scope.
2. **Repair Confit separately before publishing the cutover PR.** This preserves the cutover's engine boundary, but delays its publication. This is the recommended option.
3. **Permit a draft PR with a red gate.** Publish the cutover and document both worker crashes. This needs an explicit exception to the green-before-push rule. Do not merge it.

## Ruling

The owner directed the session to use Windows Subsystem for Linux (WSL) and the repository's Nix development shell.
The owner also suggested updating the branch.
The session rebased the cutover commits onto upstream `master`.
The upstream version is Git commit `1962f29`, which supplies the repository's Nix shell.
Before publication, this ruling requires the complete gate to pass in Linux.
The ruling does not permit executable engine changes.
It does not authorize an engine repair, weakened tests, or a draft PR with a failing gate.

## Linux verification

The session used an independent Linux clone in WSL.
Its working directory was `/tmp/sql-transforms-cutover-wsl-20261006`.
This avoided sharing Windows virtual environments or compiled extensions with Linux.

The repository's Nix shell selected Python 3.14.7, Rust 1.98.1, and Java 21.
The dependency command was `nix develop --command uv sync --locked --group spark`.
The gate command was `nix develop --command uv run --no-sync python scripts/gate.py`.

| Execution | Observed result |
| --- | --- |
| Rust engine tests | 326 passed, 1 ignored |
| Complete Python tests | 7732 passed, 12 skipped, 3 expected failures |
| DuckDB execution comparison | 288/678 agreed, 390 refused cleanly, 0 failed |
| Spark execution comparison | 260/678 agreed, 418 refused cleanly, 0 failed |
| Complete gate | `gate: green` |

Runtime checks also exercised keyed windows, explicit grouped estimator fit, and reconstruction from the four public projection fields.
Batch execution, compiled row execution, and compiled Arrow execution produced the expected values.
Strict native conversion produced native estimator execution without a Python fallback.

## Resolution

The Linux gate satisfies the owner's ruling.
No Confit engine repair or test exception is needed for this PR.
The Windows stack failure remains a platform-specific observation, not a blocker for this publication.
The session may publish the cutover PR, but must not merge it.
