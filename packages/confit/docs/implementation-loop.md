# Goal-driven implementation loop

Use the [goal](goal.md) as the objective, the [oracle](oracle/README.md) and
[serving contract](specs/serving-contract.md) as constraints, and the
[work register](oracle/11-proposed-tickets.md) as the current queue. Historical
reports and deleted `TASK-*` records are evidence, not current instructions.

Run one bounded implementation iteration at a time. Completing an iteration is
not a claim that the open-ended goal is complete. No new orchestrator or task
registry is required.

## Start here

Read the [readiness report](reports/2026-09-21-loop-readiness.md) for the measured
starting state, environment, and remaining gaps. Then use this prompt:

> Follow packages/confit/docs/implementation-loop.md for one implementation
> iteration. Select one ready outcome from the current work register, state its
> observable completion condition, implement and verify it, update that same
> register with evidence, and stop for review. Preserve the goal and evaluation
> contract. Do not recreate backlog/tasks/.

## Environment and baseline

Run from the repository root. The full gate needs Rust, Python 3.14, the locked
workspace dependencies, and Java 21 for Spark. Keep Spark enabled: an explicit
opt-out is a partial run, not a passing full gate. Set `JAVA_HOME` to the JDK;
if a Windows terminal predates installation, reload its user-level `JAVA_HOME`.

```text
uv sync --locked --group spark
uv run --no-sync python scripts/gate.py
```

After changing Rust, Cargo inputs, or native build configuration, explicitly
refresh the extension before validating Python behavior:

```text
uv run --no-sync --directory packages/confit maturin develop --release
```

The existing pytest guard also rebuilds stale Rust sources before importing the
extension. Explicit rebuilding covers dependency/build-input changes outside
that guard's source-mtime check. Preserve `--no-sync` on subsequent runs so uv
does not remove the optional Spark group.

A bounded exploratory campaign uses the existing runner:

```text
uv run --no-sync --directory packages/confit python -m fuzz.runner --seed 0 --n 100 --workers 2 --timeout 30
```

Its successful exit means the campaign completed, not that it found no defects.
Inspect the findings and complete result counts. Keep each run's dated artifacts;
never overwrite a historical baseline. Compare the same cases/population when
claiming progress, and retain the pre-execution SQL and inputs, not only seeds.

Each output stem has four files: `.jsonl` findings, `.results.jsonl` for every
verdict, `.cases.jsonl` for prepared inputs, and `.provenance.json` for parameters,
versions, settings, source changes, and the native build identity. Refusals retain
their reference outcome; unsupported widths remain separate from AGREE coverage.
A case that cannot be encoded is SKIP, not evaluated without recoverable inputs.
Missing provenance or a failure before case announcement is an explicit evidence
gap, not a reproducible measurement.

## One iteration

1. **Select.** Choose one ready, observable outcome. State its applicable rule,
   affected package, reproducer/workload, and completion condition before editing.
   Fix confirmed in-scope wrong answers before widening that affected family.
   The first planned product slice is retirement of the static-only backend.
2. **Baseline.** Record the revision and existing working-tree changes; preserve
   unrelated work. Establish current behavior on the selected case. A dirty run
   must retain its source evidence rather than masquerade as a clean revision.
3. **Implement.** Follow existing code paths and finish the cutover, including
   affected callers. Use subagents only for genuinely independent ownership;
   one integration owner reviews their results. Run shared validation after edits
   settle, not concurrently with agents changing the same build.
4. **Verify.** Exercise the selected behavior, then run the full gate once before
   handing off. Keep regressions for plausible bugs, not test-count bookkeeping.
   Serving changes must preserve row/batch independence and applicable backend,
   row/Arrow, schema, and refusal behavior. Authoring expansion also exercises C1
   with `MARGINALIZE_FUZZ_N=1500` or greater; the smaller default is not that gate.
5. **Measure.** Count verified agreement separately from acceptance, refusals,
   unsupported widths, and unknown outcomes. For performance work, run the
   existing serving benchmarks in release mode with the same workload and an
   in-run baseline; record the environment. C1–C5 constrain D1/D2, not a weighted
   score that trades correctness for speed.
6. **Hand off.** Update the same work-register item with the reproducer/artifact,
   exact commands and results, remaining uncertainty, and next ready action.
   Mark it complete only when its observable requirement is satisfied. Commit
   only scoped changes when authorized; obtain review before merging.

## Stop conditions

Stop and report a concrete blocker when the comparison contract is unresolved,
required infrastructure is unavailable, a new regression remains unexplained,
or a repeated approach produces no new evidence. Preserve failing evidence and
working changes; do not reset unrelated work or invent a pass.

Implementation may not loosen tolerances, add semantic exceptions, hide findings,
change the evaluation population, or lower floors to make a result green. A
reviewed scope change records its reason and affected cases. Raising a floor
with demonstrated support is ordinary progress.

The `DOUBLE sum/avg` domain remains a prerequisite to that family, not a reason
to block unrelated exact operations. Do not implement a speculative global
thread pin, universal coverage registry, or new blocking KPI.
