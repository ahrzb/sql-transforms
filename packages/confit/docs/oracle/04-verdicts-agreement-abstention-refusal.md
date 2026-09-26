# Campaign verdicts, refusal, and abstention

Settled verdict and refusal-reporting policy comes from the
[oracle policy record](../decisions/oracle-policy.md).

## Case-classification pipeline

A campaign case is classified rather than allowed to disappear as an exception.
Unexpected construction exceptions and backend splits exit before DuckDB runs;
otherwise the order is:

1. construct both confit backends;
2. execute optimizer-off and optimizer-on DuckDB on one connection;
3. classify construction refusal or execute both confit backends;
4. settle backend agreement;
5. compare confit separately with each DuckDB reading;
6. form the optimizer bracket; and
7. for eligible row-path results, run confit-only boundary and ordering legs.

**claim: optimizer-bracket.** The two DuckDB readings share one connection and loaded
tables. This keeps table statistics fixed while changing optimizer state. `UNSHIPPED`
outranks the bracket because no value comparison occurred; neither reading can then be
evidence about an optimizer pass.

*Enforced-by:* `fuzz.oracle._duck_run` and `fuzz.oracle.run_case`.
*Evidence:* `packages/confit/tests/test_fuzz_smoke.py::test_an_unshipped_lane_is_classified_and_never_value_compared`.

## Verdict meanings

**claim: verdict-taxonomy.** `fuzz.oracle` emits eleven kinds; the runner adds two
worker-failure kinds.

| kind | exact campaign meaning |
|---|---|
| `AGREE` | confit returns the same compared schema/values as optimizer-off and optimizer-on DuckDB |
| `AGREE_TRAP` | confit and both DuckDB readings fail at execution rather than return rows |
| `DIVERGE_VALUE` | compared schema/value mismatch, failed confit self-leg, or backend value/trap split |
| `DIVERGE_BUILD` | confit builds where DuckDB rejects at bind/build, or the confit backends split during construction |
| `DIVERGE_TRAP` | confit and DuckDB disagree on returning rows versus failing at execution |
| `DIVERGE_OPT` | confit matches optimizer-off; optimizer-on differs |
| `OPT_EMULATED` | confit matches optimizer-on while the optimizer-off reference differs; this is a finding |
| `BUILD_EXC` | confit construction raises something other than the contract `ValueError` |
| `REFUSED` | both confit backends reject construction with `ValueError` |
| `UNSHIPPED` | an enumerated missing width prevents value comparison |
| `SKIP` | an exception escapes the case harness |
| `TIMEOUT` | a worker exceeds the per-case budget; detail is an 800-byte stderr tail |
| `PANIC` | a worker exits without returning a verdict; detail has the same shape |

*Enforced-by:* `fuzz.oracle.KINDS`, `fuzz.oracle.run_case`, and `fuzz.runner`.
*Evidence:* `packages/confit/tests/test_fuzz_smoke.py::test_verdicts_cover_the_contract_and_reproduce`.

**claim: opt-emulated-classification.** `OPT_EMULATED` is a finding, never agreement or
coverage. It means confit reproduced an optimizer result that differs from the configured
reference; calling it agreement would hide the same defect in both reporting views.

A case classified `OPT_EMULATED` stops there, as any other mismatch does: the primary
finding is what the campaign reports, and no later leg may replace it. Additional
diagnostics on such a case may be reconsidered only if they preserve that original
finding.

*Enforced-by:* `fuzz.oracle.run_case`, which returns `OPT_EMULATED` before the
confit-only boundary legs; intended membership in `fuzz.runner.INTERESTING` and
exclusion from `COVERED`.
*Evidence:* emission is tested by `test_verdicts_cover_the_contract_and_reproduce`;
the stopping rule by
`packages/confit/tests/test_fuzz_smoke.py::test_opt_emulated_is_final_and_no_self_leg_replaces_it`.
Runner tuple membership remains **Unverified**: `tests/test_fuzz_report.py` drives
`fuzz.runner.report`, but no test asserts `INTERESTING` or `COVERED` membership.

## Construction refusal versus runtime trap

Construction and execution are separate contract phases.

- A confit `ValueError` during construction is a refusal. A non-`ValueError` is
  `BUILD_EXC`; disagreement between Cranelift and interpreter construction is
  `DIVERGE_BUILD`.
- A DuckDB exception is classified as bind/build only for the named DuckDB build-error
  classes. Other exceptions are runtime failures. Confit building where DuckDB rejects
  at bind/build is `DIVERGE_BUILD`.
- After successful construction, an exception from `infer_arrow` or `infer_rows` is a
  runtime trap. Agreement or divergence then depends on whether DuckDB also fails at
  execution. Error-message identity is not part of the value comparison.

Thus construction success does not promise that every input returns rows, and a runtime
trap must not be rewritten as a construction refusal. The serving-level statement lives
in [the serving contract](../specs/serving-contract.md).

**claim: refusal-message-prefixes.** The documented construction-refusal surface intends
three `ValueError` prefixes:

- `unsupported:` — valid SQL outside the served product;
- `parse error:` — outside the accepted dialect; and
- `bind error:` — invalid against the declared schema.

Current corpus classification does not match that set. `_CLEAN` accepts `unsupported:`,
`parse error:`, `duplicate map key`, and `NULL in value column`; it omits `bind error:`,
and the final two messages have no documented prefix. This is an implementation/document
inconsistency, not permission to infer a fourth settled policy. See
**ticket: clean-prefix-reconcile**.

*Evidence:* `packages/confit/docs/known-limitations.md:274-285`, P7 and P18 in
`packages/confit/docs/properties.md`, and
`packages/confit/tests/test_corpus_replay.py:36`.

**claim: reason-code-placement.** Audit classifications — codes such as
`unspecified-order`, `tie-break`, `fp-association`, `session-dependent`, and
`oracle-errored` — remain internal report and ledger vocabulary. A public diagnostic
instead names the unsupported construct and explains what the caller can do. These
audit classifications are not a public API. Publishing stable machine-readable codes
requires a concrete consumer and a separate API decision; this does not remove the
existing refusal prefixes.

**claim: refusal-grounds.** Message prefix and product ground are separate axes. The
three grounds used to discuss scope are:

1. **specialization-inherent** — the serving model cannot express it;
2. **scope-by-product-decision** — expressible, but deliberately not served; and
3. **resource** — too expensive per serving row, with a stated budget.

These grounds classify scope choices, not every invalid caller declaration. The
restriction inventory applies them through
[scope classification](../specs/serving-contract.md#scope-classification), which sorts
restrictions without ratifying every existing limit.

*Evidence:* `backlog/milestones/m-8 - duckdbs-type-lattice.md:30-36`,
`packages/confit/docs/known-limitations.md` §§1-2, and
`packages/confit/tests/known_divergences/test_arrow_boundary.py:34-36`.

**claim: refusal-absorb.** The campaign executes both DuckDB readings before returning
a confit refusal. `REFUSED` keeps the optimizer-off reading's outcome — `serves`,
`rejects` (bind/build), or `traps` (run time) — as its `oracle` field, carries a class
derived from the first six message words, and stays absent from `INTERESTING`. The
report groups refusals by that outcome, then by class, so "DuckDB serves, confit
refuses" is visible per refusal class without being promoted to a finding. This closed
**ticket: split-refused-verdict**.

*Enforced-by:* `fuzz.oracle.run_case`, `fuzz.oracle._oracle_outcome`, and
`fuzz.runner.report`.
*Evidence:* `packages/confit/tests/test_fuzz_smoke.py::test_a_refusal_keeps_the_oracle_outcome_it_already_computed`
and `packages/confit/tests/test_fuzz_report.py::test_refusals_are_summarized_by_oracle_outcome_and_class`.

**claim: refusal-outcome-reporting.** A refusal retains the oracle outcome the campaign
has already computed, and the report summarizes refusals by reason: whether DuckDB
served or trapped, and under which refusal class. That is reporting, not adjudication.
A query DuckDB serves but confit refuses is not automatically a correctness defect — its
status follows from the refusal grounds above and from severity rung 4 — so neither a
new top-level verdict kind nor a failure status for every such refusal is required. The
accepted refusal itself, divergence: bind-time-constant-refusals, is unchanged.

## Findings, abstention, and coverage

**claim: abstention-reporting.** Harness failure is `SKIP`, never a pass. `SKIP`,
`TIMEOUT`, and `PANIC` are intended findings and must reach `findings.jsonl`; otherwise a
growing blind spot can look green.

**claim: coverage-accounting.** Only `AGREE` contributes to the construct-coverage
histogram. `AGREE_TRAP` establishes a matched runtime outcome but is not counted as
construct coverage. `REFUSED` is summarized separately, `UNSHIPPED` has its own report
section because values were not compared, and `OPT_EMULATED` remains a finding.

*Enforced-by:* `fuzz.runner.INTERESTING`, `COVERED`, and `report`.
*Evidence:* oracle verdict reachability and `UNSHIPPED` behavior are covered in
`packages/confit/tests/test_fuzz_smoke.py`; runner membership remains **Unverified** under
**ticket: verdict-tuple-test**.

**claim: logged-fallback.** If a checker cannot evaluate its strongest condition,
it may use a weaker check only with an explicit tag. The current legacy example is
an `ORDER BY` expression absent from the output: multiset comparison still runs,
but sortedness is not established and the case receives `order-by-unevaluated`.
That tag must not be reported as evidence that the stronger check passed.

*Evidence:* `fuzz.oracle.run_case` and the ordering work recorded in TASK-129.

**claim: timeout-attribution.** **[FACT]** The current timeout identifies neither the
side nor the SQL. Oracle work timing out and confit work timing out imply opposite
problems, so recovery requires regenerating the seed. On 2026-08-14, seed 4395 made
DuckDB spend 9.0 seconds building a 2 GiB `lpad` while confit refused immediately under
its 1 GiB budget; three other seeds had the same shape.

*Evidence:* `packages/confit/docs/2026-08-13-fuzz-triage.md:124-149`.
*Open work:* record SQL before execution and split oracle-side from engine-side timeout.

**claim: countable-cost.** Disclose an accepted cost honestly and measure the ones that
matter, starting with refusal outcomes under claim: refusal-outcome-reporting. No
universal rule requires every accepted cost to name a counting mechanism, and
conservative-refusal counting is that same reporting work rather than a second rule.
