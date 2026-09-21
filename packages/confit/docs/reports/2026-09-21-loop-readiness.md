# Implementation-loop readiness — 2026-09-21

The repository is ready for one bounded implementation iteration using the
[loop procedure](../implementation-loop.md) and the existing
[work register](../oracle/11-proposed-tickets.md). This is preparation and a dated
baseline, not completion of the goal or a claim of expanded product support.

## Verified environment and gate

Windows 11 AMD64; CPython 3.14.0; DuckDB 1.5.5; PyArrow 25.0.0; NumPy 2.5.1;
scikit-learn 1.9.0. The locked Spark group installed PySpark 4.2.0.
Temurin 21.0.12.1+1 was installed outside the repository and its archive's SHA-256
was verified before extraction. User-level `JAVA_HOME` is set; older Windows
processes need their environment refreshed. No system PATH change was required.

Commands, from the repository root:

```text
uv sync --locked --group spark
uv run --no-sync --directory packages/confit maturin develop --release
uv run --no-sync python scripts/gate.py
```

The gate ran with the installed JDK supplied through `JAVA_HOME` and
`CONFIT_ALLOW_NO_SPARK=0`:

- Rust: **273 passed, 0 failed, 1 ignored**; the ignored case is an explicit benchmark.
- Python: **3,341 passed, 1 skipped, 9 xfailed**, in 393.36 seconds.
- Dialect L2: **288/678 match, 390 clean unsupported, 0 failures**.
- Real Spark L3: **260/678 match, 418 clean unsupported, 0 failures**.
- BigQuery remains credential-dependent and was not exercised.
- Existing warnings remain, including PySpark's pandas >=3 compatibility warning.
  They were not suppressed or treated as failures.

This ran the normal gate, not the deeper `MARGINALIZE_FUZZ_N=1500` authoring-expansion
run, and did not establish a new serving-performance result.

The first gate attempt exposed five stale Rust test failures. Direct measurements
against the fixed reference established `substr('hello', -6, 3) = 'he'`,
`substr('hello', -10, 8) = 'hel'`, and `CAST('0x10' AS BIGINT) = 16`; expectations
were corrected without changing engine behavior. The float-to-integer test now
checks trapping rather than incidental error wording. An internal catalogue-partition
test and its test-only name list were removed: DuckDB lowers `if` to a CASE AST, so
membership in those internal tables was not a valid test of supported behavior.

## Evaluator safeguards

- The reference constructor rejects a different DuckDB version before connecting.
  The guard also passed a `python -O` smoke check. Only the dev environment's pin
  changed; the published `sql-transform` dependency floor did not.
- `OPT_EMULATED` stops before a later self-comparison can replace the primary finding.
- Refusals retain the already-computed reference outcome and remain distinct from
  findings, unsupported widths, and AGREE coverage.
- Workers record prepared inputs before evaluation. Crashes and deadlines retain
  announced cases; malformed protocols or incomplete campaigns fail rather than
  printing a clean result. Unrecordable inputs become SKIP before evaluation.
- Campaign files are not overwritten. All results and source/build/reference
  provenance accompany the findings, including dirty source changes.
- The native rebuild guard runs before importing the extension and rejects a stale
  build when maturin is unavailable, rather than silently validating the old binary.

The focused oracle, fuzzer, runner, and native-guard run passed **40 tests**. The new
version-guard and primary-verdict regressions both failed against the pre-preparation
modules at `b38fa8fcc4f84f9e63ead99e1ce1961fa3807166`. The native import-order and
missing-build-tool smoke reproductions also failed before their fixes and passed after.

## Fresh campaign baseline

```text
uv run --no-sync --directory packages/confit python -m fuzz.runner --seed 0 --n 100 --workers 2 --timeout 30 --out docs/reports/2026-09-21-loop-baseline.jsonl
```

Executed 2026-09-21 09:38:25–09:38:29 UTC. The native extension was a **release**
build. This was a dirty working tree based on the revision above, not a clean-commit
measurement; provenance retains the runtime source patch and compiled binary hash.
The reference recorded **12 threads** and **Europe/Berlin** timezone. Neither was
changed into a new global policy.

| Verdict | Cases |
|---|---:|
| AGREE | 50 |
| REFUSED | 46 |
| UNSHIPPED | 3 |
| AGREE_TRAP | 1 |
| Findings | 0 |

Of the 46 refusals, the reference served **26**, failed at build for **15**, and
trapped at execution for **5**; none had an unknown reference outcome. These are
reporting populations, not automatic correctness or permanent-scope adjudications.
The three unsupported cases were decimal widths, not verified agreement.

Artifacts:

- [Findings](2026-09-21-loop-baseline.jsonl) — intentionally empty.
- [Every verdict](2026-09-21-loop-baseline.results.jsonl).
- [Prepared cases](2026-09-21-loop-baseline.cases.jsonl).
- [Source, build, environment, and reference provenance](2026-09-21-loop-baseline.provenance.json).

Verified 100 unique prepared-case/result pairs covering seeds 0–99, matching SQL,
and matching generator-source and native-binary hashes. A zero-finding 100-case smoke
is not a zero-defect claim or a new acceptance-rate KPI. This population still includes
the legacy constant path at **seed 95**; it is not solely serving-contract coverage.

## Next boundary

The next product outcome is **ticket: retire-static-only-fold**. Remove the legacy
query-level `constant` backend while preserving legitimate construction-time
preparation and constant expressions within request-dependent transforms. Record
seed 95 and any other affected cases as a reviewed scope reclassification, not a
support increase.

The goal and oracle overview remain unchanged, and `backlog/tasks/` remains deleted.
Open nullability, unresolved-observation, refusal-quality, width-reachability, scope,
and behavioral-coverage work stays in the current register. No new orchestrator,
coverage registry, numerical tolerance, or blocking KPI was introduced.
