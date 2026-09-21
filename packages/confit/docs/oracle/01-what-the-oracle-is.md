# Reference rationale and enforcement

The concise oracle definition lives in [README.md](README.md). This chapter explains why
that reference was chosen, how repository comparisons reach it, and where enforcement is
still incomplete. It does not define an alternative configuration. Settled reference
policy comes from the [oracle policy record](../decisions/oracle-policy.md).

## Why the optimizer is disabled

**claim: optimizer-on-reading.** Optimizer-on DuckDB is not always a function of the
query and current rows. `statistics_propagation` can read stored column statistics, so
two tables with identical contents but different insert/delete histories can answer the
same query differently. Confit compiles against a schema and cannot reproduce that table
history.

*Evidence:* `packages/confit/tests/known_divergences/test_trap_elision.py` and
`packages/confit/docs/known-limitations.md:32-39`.

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
`src/include/duckdb/common/enums/optimizer_type.hpp:16-50` and the sites above, inspected
2026-08-25. Proposed corrections to older summaries are tracked by
**ticket: oracle-docstring-corrections** and **claim: phase-separated-probes**.

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

*Enforced-by:* `fuzz.oracle.run_case`; `fuzz.runner.INTERESTING` is the intended findings
membership.
*Evidence:* emission is covered by
`packages/confit/tests/test_fuzz_smoke.py::test_verdicts_cover_the_contract_and_reproduce`.
No test imports `fuzz.runner`, so findings membership remains **Unverified**; see
**ticket: verdict-tuple-test**.

## Known identity-enforcement gaps

**claim: oracle-version-constant.** **[FACT]** `Oracle.VERSION` records `"1.5.5"`, but
construction does not compare it with `duckdb.__version__`. The root and
`packages/sql-transform` manifests use `duckdb>=1.5.5`,
`packages/confit/pyproject.toml` declares only `pyarrow>=19.0`, and `uv.lock` currently
resolves DuckDB 1.5.5. A lock upgrade can therefore move the executable reference
without an assertion.

*Evidence:* `packages/confit/confit/oracle.py:74-82`; the cited manifests; and
`uv.lock:368-370`. No test reads `Oracle.VERSION`.

**claim: version-policy.** DuckDB 1.5.5 remains the reference. The reproducible
oracle/test environment must pin that version exactly, and opening the oracle must
assert `duckdb.__version__ == Oracle.VERSION`. Unrelated DuckDB consumers are not
constrained by this rule. Leaving 1.5.5 is a separate reviewed reference change;
the generic re-recording tools in [version changes](09-version-bumps-and-mutability.md)
remain proposals, not prerequisites adopted by this rule.

The assertion is the rule, not the current behavior: claim: oracle-version-constant
above records **[FACT]** that construction still performs no comparison, and
**ticket: version-assert** tracks the implementation.

**claim: one-door-bypass.** **[FACT, current implementation only]** The legacy
static-only engine path is the comparison-path bypass: `eval_static_only` calls
`duckdb.connect()` directly and folds with the optimizer on. The target now refuses
queries that read no request table, but that removal is not implemented; the remaining
path is **gap: static-only-fold**. Pin-capture scripts are a separate family described by
**claim: capture-outside-the-oracle**.

*Evidence:* `packages/confit/src/duckdb/mod.rs:1173-1203,1707-1743` and the
[fold-retirement decision](../decisions/trustworthy-fold.md). The decision record is the
sole history of alternatives considered for this fold.

## Nearby DuckDB uses with different contracts

**claim: fit-serving-oracle.** `sql_transform`'s fit/serving checks are independent.
Its projection path uses optimizer-on DuckDB with `SET threads = 1`; training round-trip
and transformer parity are defined in
[success measures](../specs/success-measures.md). Fit reproducibility is not a v0
contract.

*Evidence:* `packages/sql-transform/sql_transform/_projection.py:188-189,410` and P11,
P16 in `packages/confit/docs/properties.md`.

**claim: dialect-gate-oracle.** Dialect gates have their own pinned targets. Spark L3
uses ANSI mode, UTC, `local[1]`, and `pins-dialect/spark-ansi.json`; it compares names
and row multisets. Exact comparison is separate from its reserved float-accumulation
epsilon tier. BigQuery skips loudly without credentials and is recorded as
unversionable. The Spark support floor is a ratchet, and the corpus gate uses the same
mechanism with `MATCH_FLOOR = 547`. Both follow claim: stable-corpus-ratchet in
[campaign validity](10-campaign-validity-and-blind-spots.md): no unexplained decrease in
support. A floor is a regression threshold, not a fresh measurement or a universal
compatibility claim.

*Evidence:* `packages/confit/tests/test_dialect_cross_engine_gate.py:1-31`,
`packages/confit/docs/specs/2026-08-13-dialect-logical-plan-design.md:32-36,244-248`,
and `packages/confit/tests/test_corpus_replay.py:18-24,39-47,205-209`.

DuckDB also supplies the parser/printer used by `sql_transform`; serialized shapes are
pinned per DuckDB version in `sql_transform/model/_shapes.json`. That role does not make
parser shape capture a differential-oracle comparison.
