# Success measures

The [goal](../../../../loops/confit/goal.md) sets the priorities. These are the five correctness
controls (C1–C5) and two optimization measures (D1–D2), including checks owned by
`sql_transform`, which keeps ownership of its checks.

## Standing rule

Never weaken a correctness control to improve coverage or latency.
Changing a bound requires an explicit reviewed decision.
Do not change a bound to make a failing test pass.
Missing implementation is not an exception to a requirement.

## Correctness controls

### Training round-trip (C1)

**kpi: training-round-trip.** Fit followed by execution on the fit data must
be bit-exact equal to the original authored computation. For windows, compare the original
SQL with `__THIS__` bound to that data at `SET threads = 1` on both DuckDB paths.
Keep schemas, signed zero and NaN comparisons exact. Transformer columns use C4.
The [authoring contract](../../../sql-transform/docs/contract.md) defines the fit model.

Gate: `packages/sql-transform/sql_transform/_marginal_projection_test.py::gate`
and `::test_fuzz_differential`. `MARGINALIZE_FUZZ_N` controls the depth
(seed 20260729); the default is 1,500 cases
([C1 depth](../../../../loops/confit/decisions/closed/c1-depth.md)).
Explicit replacements preserve the original computations.
They do not count as acceptance of the original syntax.

### Engine parity (C2)

**kpi: engine-parity.** Confit must serve bit-for-bit identical to the fixed
[optimizer-off DuckDB oracle](../oracle/README.md), with the same declared UDFs
registered, or refuse construction with an error naming the unsupported construct.
The [comparison contract](../oracle/05-the-comparison-contract.md) specifies the
order axis and the one approved value exception, the `cbrt` bound. No ledger
entry permits a further deviation.

Gates: `packages/confit/tests/test_duckdb_*.py`, `test_params_joins.py`,
`test_udfs.py`, and the refusal suites. The internal byte-for-byte
Cranelift/interpreter invariant is also checked by the random-IR differential in
`packages/confit/src/specializer/exec/tests.rs`. Refusal tests cover named cases,
not a proven complete inventory.

### Binding parity (C3)

**kpi: binding-parity.** A Confit-servable fitted projection exposes `.sql`,
`.schema`, `.params` and `.udfs`. DuckDB batch execution and both Confit
interfaces must agree value-for-value from these four public fields.
The [authoring contract](../../../sql-transform/docs/contract.md) defines this artifact.
Projection admission and Confit admission are separate.
A batch-only query does not establish binding parity.

Gate: `packages/sql-transform/sql_transform/_serving_test.py`, `serve_gate()`.

### Transformer parity (C4)

**kpi: transformer-parity.** Transformer columns must agree with an independently
cloned, fitted, and applied per-group sklearn reference, not a reference derived
through the library code under test.

**claim: sklearn-is-the-reference.** Where DuckDB cannot execute the model
operation, use the independent model reference. The bounds are surface-specific:

| Surface | Comparison | Evidence |
|---|---|---|
| Tree scoring | Bit-exact target against `sklearn.predict` | `_trees_test.py::test_matches_sklearn_bit_exactly` |
| StandardScaler columns | `rtol=1e-12`, `atol=0` | `_transformers_test.py` |
| StandardScaler/PCA pipeline columns | `rtol=1e-9`, `atol=0` | `_transformers_test.py` |
| Campaign sklearn second-reference leg | Absolute `1e-9` | `packages/confit/fuzz/oracle.py`, `_extra_legs` |

The first three paths are in `packages/sql-transform/sql_transform/`.
`_transformers_test.py::_reference()` constructs the independent reference.
The tree test uses exact numerical equality, not a bit-view comparison, so it does
not independently verify signed-zero bits. The campaign bound is a separate
self-leg, not a replacement for C4, and has no dedicated bound test. A
SQL-window/NumPy mean assertion does not define transformer parity.

### No third mode (C5)

**kpi: no-third-mode.** A query must compute under the applicable controls
or refuse at the applicable boundary with an error naming the unsupported construct.
The SQLProjection constructor checks syntax and row-local projection form.
Fit checks learned schemas, fields and data-dependent params multiplicity.
DuckDB binding can fail during fit or batch execution.
Confit checks its own admission when the fitted projection compiles.
Silently accepting a different computation is not a permitted outcome.
Data-dependent runtime traps remain governed by the oracle contract.

Gate: `packages/sql-transform/sql_transform/_corpus_test.py` checks each kept,
migrated and refused case. A comparison failure is not an admitted outcome.
Refusal suites cover named cases, not a proven complete inventory of diagnostics.

## Optimization measures

### Coverage ladder (D1)

**kpi: coverage-ladder.** Report authoring cases as kept, migrated or refused.
Kept cases retain their original syntax. Migrated cases use explicit replacement
SQL but retain the original computation as the reference.
Refused cases retain a named refusal. No category permits a comparison failure.
Add a case before widening support and extend C1 to its computation.

`packages/sql-transform/sql_transform/_corpus_test.py` checks cases through
`test_original_accepted_mined_computations`, `test_mined_meaningful_refusals`,
`test_curated_kept_original_syntax`, `test_curated_migrated_chain_computations`,
`test_curated_schema_migrated_computations` and `test_curated_refuses`.
Report original syntax acceptance separately from migrated computation coverage.
This is the authoring package's ladder, not Confit's build acceptance rate.

### Serving latency (D2)

**kpi: serving-latency.** Measure row-at-a-time cost using a release build with
`benchmarks/bench_serving.py` and `benchmarks/bench_transforms.py`. Compare
alternatives within a run and record the environment. Cross-run ratios require
a stable baseline measured in each run; absolute numbers alone are load-sensitive.
The goal's single-digit-microsecond objective has no enforced numeric threshold.
Measurements and baseline changes belong in dated reports; the
[performance report](../reports/performance-report.md) holds the method.

## Acceptance and measurement

Acceptance asks whether construction returned a function; verified coverage asks
whether comparison established agreement. An accepted function with a parity defect
still counts as accepted, but fails correctness. Only `AGREE` counts as campaign
coverage. `UNSHIPPED` is an enumerated unsupported width with no value comparison,
not agreement or a finding. The complete taxonomy is in
[verdicts](../oracle/04-verdicts-agreement-abstention-refusal.md).

A verdict other than `REFUSED` does not prove construction succeeded: a timeout or
harness failure may leave that unknown. An acceptance rate must state its population
and treatment of unknown outcomes and invalid declarations. A generated-grammar rate
measures that generator, not the SQL users write.

### Measurement methods

| Surface | Method and gate |
|---|---|
| Mined DuckDB corpus | `test_corpus_replay.py`: zero FAILs and `MATCH_FLOOR` |
| Dialect L2 | `test_dialect_corpus_gate.py`: parse/print parity and `SUPPORTED_FLOOR` |
| Dialect L3 | `test_dialect_cross_engine_gate.py`: second-engine comparison and `SPARK_MATCH_FLOOR`; needs Spark |
| Generated campaigns | `python -m fuzz.runner`: manual campaign, not a standing CI gate |
| Authoring admission ladder | `sql_transform/_corpus_test.py`: kept/migrated/refused cases compared with their original computations |

Confit test paths above are relative to `packages/confit/tests/`. The mined floor
permits deliberate scope tightening with an explained adjustment. Its optimizer-on
capture lacks sufficient provenance and is not interchangeable with a live
optimizer-off oracle comparison. See [campaign validity](../oracle/10-campaign-validity-and-blind-spots.md)
for enforcement limits. A test path resolving does not show that its gate ran.

## Measurement policy

C1–C5 and D1–D2 are the blocking set; there are no numeric acceptance targets.
Stable corpora are held by reviewable ratchets and generated campaigns are
reported. Each reported population is defined, with unknown outcomes and invalid
declarations shown separately; an acceptance rate is never improved by changing
the generator or denominator.

Stable-corpus support may not decrease without a reviewed reason and the affected
cases. A total can hide one regression offset by one new match, so it does not
replace case-level checks. This is the oracle's
[stable-corpus ratchet](../oracle/10-campaign-validity-and-blind-spots.md). Scope
classification lives in the [serving contract](serving-contract.md#scope-classification).
