# Reference rationale and enforcement

The concise oracle definition lives in [README.md](README.md). This chapter explains why
that reference was chosen and how repository comparisons reach it. It does not define an
alternative configuration. Settled reference policy comes from the
[oracle policy record](../decisions/closed/oracle-policy.md).

## Why the optimizer is disabled

**claim: optimizer-on-reading.** Optimizer-on DuckDB is not always a function of the
query and current rows. `statistics_propagation` can read stored column statistics, so
two tables with identical contents but different insert/delete histories can answer the
same query differently. Confit compiles against a schema and cannot reproduce that table
history.

*Evidence:* `packages/confit/tests/known_divergences/test_trap_elision.py` and
`packages/confit/docs/known-limitations.md` (introduction).

**claim: unoptimized-verifier.** Optimizer-off is also an upstream DuckDB verification
leg. `PRAGMA enable_verification` registers an `UNOPTIMIZED` verifier and compares its
result with the original plan; DuckDB describes this as checking correctness with and
without optimizers.

*Evidence:* DuckDB v1.5.5
`src/function/pragma/pragma_functions.cpp:134`,
`src/main/client_verify.cpp:45,55,187`, and
`src/verification/statement_verifier.cpp:155`.

**claim: disable-optimizer-scope.** `PRAGMA disable_optimizer` disables more than the 33
named `OptimizerType` passes. In DuckDB v1.5.5, twelve additional source sites read
`enable_optimizer` directly:

| area | sites | observed effect |
|---|---|---|
| physical selection | `plan_distinct.cpp:66`; `plan_window.cpp:31,35`; `sorted_aggregate_function.cpp:686,744` via `plan_aggregate.cpp:318` | DISTINCT-ON rewrite, window operator choice, sorted-aggregate simplification |
| window execution | `window_aggregate_function.cpp:32,56`; `window_rank_function.cpp:24`; `window_rownumber_function.cpp:26`; `window_value_function.cpp:207,849` | aggregate strategy and rank/row-number/value fast paths |
| logical-plan construction | `plan_subquery.cpp:255`; `plan_joinref.cpp:411` | delim-join choice and whether a RIGHT join is flipped |

Expression binding, overload selection, type inference, and execution-level laziness do
not read this flag. The binder-directory sites above construct logical plans; they do
not change binding semantics. The join flip can change hash-join output order; the
[ordering contract](03-nondeterminism.md) separates DuckDB multiset parity from
confit's serving-order promise.

Disabling the optimizer also disables DuckDB's constant-folding rewrite. `SELECT 1 + 2`
still returns `3`, but by execution rather than that optimizer rule. This is observable
for `2147483647 + 1 ... LIMIT 0`, where optimizer-on can replace the plan with
`EMPTY_RESULT`.

*Evidence:* DuckDB v1.5.5
`src/include/duckdb/common/enums/optimizer_type.hpp:16-50` and the sites above. The
`confit/oracle.py` module docstring and `known-limitations.md` state this scope; see also
**claim: phase-separated-probes** in [pins](06-pins.md).

## How comparison code reaches the reference

**claim: no-raw-connections.** Comparison code in `tests/` and `fuzz/` obtains DuckDB
through `confit.oracle.Oracle`. A source gate rejects raw `duckdb.connect(` calls in
those trees.

*Enforced-by:* `confit.oracle.Oracle.__init__` and
`packages/confit/tests/test_oracle.py::test_no_raw_connections_in_the_sources`.

**claim: optimizer-flip-in-place.** A comparison that also needs the ordinary
optimizer-on reading calls `Oracle.optimizer_on()` on the same connection. Reusing the
connection holds loaded tables and their statistics fixed, so the bracket measures the
optimizer rather than two table histories.

*Evidence:* `packages/confit/tests/test_oracle.py::test_optimizer_on_flips_the_same_connection`,
`packages/confit/tests/known_divergences/test_trap_elision.py`, and
`fuzz.oracle._duck_run`. [Campaign verdicts](04-verdicts-agreement-abstention-refusal.md)
define the consequence of comparing both readings.

**claim: contract-surface-gap.** Ordinary DuckDB has the optimizer on. A case where
confit agrees with the optimizer-off reference but differs from optimizer-on DuckDB is
reported as `DIVERGE_OPT`, not accepted as agreement with both surfaces.

*Enforced-by:* `fuzz.oracle.run_case`; `DIVERGE_OPT` is in `fuzz.runner.INTERESTING`.
*Evidence:* `packages/confit/tests/test_fuzz_smoke.py::test_verdicts_cover_the_contract_and_reproduce`
and `packages/confit/tests/test_fuzz_report.py::test_findings_and_coverage_are_what_the_contract_says`.

## Reference identity

**claim: oracle-version-constant.** `Oracle.VERSION` records `"1.5.5"`, and
construction raises `RuntimeError` unless `duckdb.__version__` equals it. The root
dev dependency group pins `duckdb==1.5.5`, so the reproducible oracle/test
environment cannot resolve any other version. `packages/sql-transform` keeps
`duckdb>=1.5.5` and `packages/confit/pyproject.toml` declares only `pyarrow>=19.0`:
unrelated consumers stay unconstrained.

*Enforced-by:* `confit.oracle.Oracle.__init__`, the root `pyproject.toml` dev group,
and `uv.lock`.
*Evidence:* `packages/confit/tests/test_oracle.py::test_construction_asserts_the_pinned_version`.

**claim: reference-platform.** The oracle is DuckDB's behavior on Linux. Where DuckDB
answers differently on another platform — a libm-dependent NaN sign, a platform-specific
bug — confit follows the Linux answer and does not replicate the other. Rust pins of
Linux-measured DuckDB behavior carry `#[cfg(target_os = "linux")]`, and Python ones
`skipif(sys.platform != "linux")`; CI (`.github/workflows/ci.yml`) runs on
`ubuntu-latest`.

*Decision:* [oracle policy](../decisions/closed/oracle-policy.md#reference-and-comparison).
*Evidence:* `src/specializer/exec/tests.rs` (`pin_ssubstr_window_arithmetic`,
`pin_ftoi_rounding_and_traps`, `pin_stoi_trims_whitespace_like_duckdb_cast`),
`src/specializer/tests.rs::substr_window_arithmetic_via_sql`, and
`tests/test_duckdb_interpreter.py::test_two_arg_substr_is_a_uint32_max_window_differential`.

**claim: version-policy.** DuckDB 1.5.5 is the reference. The reproducible
oracle/test environment pins that version exactly, and opening the oracle asserts
`duckdb.__version__ == Oracle.VERSION`. Unrelated DuckDB consumers are not
constrained by this rule. Changing the reference version is a separate reviewed change
that moves `Oracle.VERSION` and the dev pin together.

**claim: one-door-bypass.** The static-only engine path bypasses the oracle:
`eval_static_only` in `packages/confit/src/duckdb/mod.rs` calls `duckdb.connect()`
directly and evaluates, with the optimizer on, a query that reads no request table.
The [goal](../goal.md#scope) places such queries outside the model. Pin-capture scripts
are a separate family described by **claim: capture-outside-the-oracle** in
[version changes](09-version-bumps-and-mutability.md).

*Evidence:* `eval_static_only` and its caller in `packages/confit/src/duckdb/mod.rs`;
the [fold decision](../decisions/closed/static-only-queries.md).

## Nearby DuckDB uses with different contracts

**claim: fit-serving-oracle.** `sql_transform`'s fit/serving checks are independent.
Its projection path uses optimizer-on DuckDB with `SET threads = 1`; training round-trip
and transformer parity are defined in
[success measures](../specs/success-measures.md). Fit reproducibility is not part of
the contract.

*Evidence:* `packages/sql-transform/sql_transform/_projection.py` and P11, P16 in
`packages/confit/docs/properties.md`.

**claim: dialect-gate-oracle.** Dialect gates have their own pinned targets. Spark L3
uses ANSI mode, UTC, `local[1]`, and `pins-dialect/spark-ansi.json`; it compares names
and row multisets exactly. BigQuery skips loudly without credentials. The Spark support
floor (`SPARK_MATCH_FLOOR`) and the corpus gate's `MATCH_FLOOR` are ratchets that follow
claim: stable-corpus-ratchet in [campaign validity](10-campaign-validity-and-blind-spots.md):
no unexplained decrease in support. A floor is a regression threshold, not a fresh
measurement or a universal compatibility claim.

*Evidence:* `packages/confit/tests/test_dialect_cross_engine_gate.py` module docstring
and `SPARK_MATCH_FLOOR`; `packages/confit/tests/test_corpus_replay.py` (`MATCH_FLOOR`).

DuckDB also supplies the parser/printer used by `sql_transform`; serialized shapes are
pinned per DuckDB version in `sql_transform/model/_shapes.json`. That role does not make
parser shape capture a differential-oracle comparison.
