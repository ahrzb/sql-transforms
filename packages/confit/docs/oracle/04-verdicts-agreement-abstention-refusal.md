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
That it is written as a finding and never counted as coverage is observed through the
report by `packages/confit/tests/test_fuzz_report.py::test_findings_and_coverage_are_what_the_contract_says`.

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

The engine's formerly unprefixed build-time families now carry the prefix of their
class (**ticket: clean-prefix-reconcile**, done 2026-09-26), keeping their old text as a
suffix: a duplicate key in a 1:1 static map is `unsupported: duplicate map key …`
(multiplicity restriction), `shape='map'` blockers are `unsupported: shape='map': …`,
and a NULL in a declared non-null static value column and the build-time UDF
declaration errors (`udf '<name>': …`) are `bind error: …` (inconsistent caller
declaration). UDF errors raised while serving a row are runtime traps and keep their
text. The corpus gate's `_CLEAN` is now exactly `unsupported:` and `parse error:`;
`bind error:` stays a corpus FAIL on purpose, because every corpus statement is one
DuckDB answered, so "invalid against the declared schema" cannot be its reason. No
public error-code API was added.

*Evidence:* `packages/confit/docs/known-limitations.md` §6, P7 and P18 in
`packages/confit/docs/properties.md`, and
`packages/confit/tests/test_corpus_replay.py` (`_CLEAN`), and
`packages/confit/tests/test_known_limitations.py::test_every_refusal_family_carries_a_documented_prefix`.

**claim: refusal-quality-report.** The campaign reports refusal quality, not only
prefix presence: of all refusals, how many carry a documented prefix, how many name the
construct rather than echoing source or AST text, and how many say what the caller can
do; the echoing families are listed by the text before the echo (for example
`unsupported: FROM (SELECT …`, a derived table refused through a message that repeats
the subquery). These are text heuristics for reporting only: they back no gate, and no
blocking KPI is adopted from them.

*Enforced-by:* `fuzz.runner.refusal_quality` and `fuzz.runner.report`.
*Evidence:* `packages/confit/tests/test_fuzz_report.py::test_refusal_quality_reads_naming_and_actionability`
and `::test_refusal_quality_is_reported_as_shares_of_all_refusals`.

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
`packages/confit/tests/test_fuzz_smoke.py`; which kinds reach `findings.jsonl` and which
feed coverage is observed through the report by
`packages/confit/tests/test_fuzz_report.py::test_findings_and_coverage_are_what_the_contract_says`.

**claim: logged-fallback.** If a checker cannot evaluate its strongest condition,
it may use a weaker check only with an explicit tag. The current legacy example is
an `ORDER BY` expression absent from the output: multiset comparison still runs,
but sortedness is not established and the case receives `order-by-unevaluated`.
That tag must not be reported as evidence that the stronger check passed.

*Evidence:* `fuzz.oracle.run_case` and the ordering work recorded in TASK-129.

**claim: timeout-attribution.** A `TIMEOUT` or `PANIC` names its SQL and its side.
The worker writes a phase marker to stderr before each stage — `harness:startup`,
`harness:gen`, `confit:build`, `oracle`, `confit:run`, `confit:legs` — and the runner
reads the last marker a killed worker wrote: the finding's `side` is `oracle`, `confit`,
`harness` or `unknown`, and its class is `timeout:<side>` / `panic:<side>`. The SQL and
inputs are regenerated from the seed in the parent, so they are preserved even though
the worker never returned. Oracle-side and confit-side timeouts imply opposite problems;
on 2026-08-14, seed 4395 made DuckDB spend 9.0 seconds building a 2 GiB `lpad` while
confit refused immediately under its 1 GiB budget. The markers are internal audit
vocabulary, not a public API.

*Enforced-by:* `fuzz.oracle._phase`, `fuzz.worker`, `fuzz.runner.side_of`, and
`fuzz.runner.blame`.
*Evidence:* `packages/confit/tests/test_fuzz_smoke.py::test_a_case_marks_each_phase_on_stderr_before_it_runs`
and `packages/confit/tests/test_fuzz_report.py::test_the_last_phase_marker_attributes_the_side`.

**claim: abstention-report.** The campaign report states `SKIP`, `TIMEOUT` and `PANIC`
counts as rates over the case population, the worker failures split by side, and the
rate of `AGREE` cases tagged `order-by-unevaluated` (agreement whose sortedness was not
checked). `UNSHIPPED` stays in its own section.

*Enforced-by:* `fuzz.runner.report`.
*Evidence:* `packages/confit/tests/test_fuzz_report.py::test_abstentions_are_reported_as_rates_by_kind_and_side`.

**claim: countable-cost.** Disclose an accepted cost honestly and measure the ones that
matter, starting with refusal outcomes under claim: refusal-outcome-reporting. No
universal rule requires every accepted cost to name a counting mechanism, and
conservative-refusal counting is that same reporting work rather than a second rule.
