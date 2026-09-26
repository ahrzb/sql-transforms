# Oracle policy

**Question.** What is the oracle's reference, how is it compared, and what counts as
evidence?

**Ruling.** The rules below. They keep a small core contract rather than make current
limitations or documentation machinery part of its definition. Where the
implementation differs, the [oracle chapters](../../oracle/README.md) state what is
true of the code.

## Reference and comparison

- **Version:** DuckDB 1.5.5. Pin the reproducible oracle/test environment
  and assert its version when opening the oracle. Do not unnecessarily constrain
  unrelated DuckDB consumers. An upgrade is a separate reviewed change.
- **Platform:** the reference is DuckDB as it behaves on
  Linux. Behavior DuckDB shows only on another platform, platform-specific bugs
  included, is not replicated. Tests that pin Linux-measured DuckDB behavior may be
  marked Linux-only; CI runs on Linux.
- **Execution order:** do not change the global thread setting speculatively.
  Define a justified contract when implementing each order-sensitive family;
  refuse a family until that contract is clear. Single-thread execution alone
  does not make unordered SQL ordered. Relational `DOUBLE` `sum` / `avg` is
  compared under the float-reduction bound.
- **Numerical exceptions:** maintain an explicit approved list, with operation,
  comparison rule, valid inputs, and rationale. Tests do not choose new
  tolerances independently. The reduction bound applies only once its algorithms
  and edge domain are defined; no general epsilon substitutes for them.
  Empty/all-NULL and non-finite outcomes need explicit oracle behavior.
- **Unsupported widths:** require behavioral coverage showing that they are
  classified rather than counted as agreement. Reuse existing coverage; do not
  require a duplicate strict-xfail test merely as bookkeeping. A strict xfail
  is useful for a concrete defect whose repair should expire the exception.
- **Scope:** distinguish outside-the-model computation, unimplemented in-scope
  features, explicit product/resource restrictions, and invalid inputs. Current
  syntax bans do not define permanent scope. This classifies restrictions; it
  does not blanket-ratify every existing limit or served divergence.
- **Output nullability:** metadata must be truthful, but exact identity with
  DuckDB's nullable flags is not required. A non-null promise must be sound;
  conservative nullable metadata need not match another engine's inference.
  Output names, types, and field order must match exactly.

## Reporting and measurement

- **Refusal cost:** retain the oracle outcome already computed and summarize it
  by refusal reason. A query DuckDB serves but Confit refuses is not automatically
  a correctness defect. New top-level verdicts or failure status for every such
  refusal are not required.
- **Optimizer mismatch:** stop at the primary `OPT_EMULATED` finding, as for other
  mismatches, rather than letting a later self-check replace it. Additional
  diagnostics may be reconsidered if they preserve the original finding.
- **Reason codes:** keep audit classifications internal. Public diagnostics must
  name the unsupported construct and be actionable. Stable public codes are
  [postponed](../postponed/public-reason-codes.md).
- **Corpus ratchets:** stable corpora allow no unexplained decrease in support.
  Raise floors with growth; a decrease needs a reviewed reason and affected cases.
  A total is not universal SQL compatibility and does not replace case-level
  regression checks.
- **Acceptance and KPIs:** report generated-campaign acceptance with an explicit
  population, not an arbitrary percentage target. Keep C1–C5 and D1–D2. Improve
  refusal-quality and unsupported-width reporting before adding blocking KPIs;
  no new blocking KPI is adopted here.

## Evidence and unresolved observations

- **Coverage claims:** state demonstrated coverage, not universal test totality.
  Fill important behavioral gaps without a mandatory one-test-per-paragraph registry.
- **Divergences:** judge against the contract, not ledger membership. A confirmed
  in-contract mismatch is a defect unless an explicit approved exception permits
  it. Listing a mismatch never approves it; an invalid comparison is not by
  itself an engine defect.
- **Unresolved observations:** permit a visible, separately counted unresolved
  category. It is neither agreement nor an approved exception nor necessarily
  a confirmed defect. `Unresolved` means not yet understood; `unspecified` means
  the contract deliberately leaves the particular aspect unconstrained.
- **Baselines:** a recorded run is never rewritten. Results are dated and
  provenance-bearing: SQL, inputs, generator revision where generated, engine
  revision, and reference configuration. Seeds are not durable identities after
  generator changes. A count whose run cannot be reconstructed is not evidence.

## Limits on process rules

Use fixed/implementation-defined/unspecified distinctions only where they clarify
variation, not as mandatory labels on every claim. Disclose tradeoffs honestly and
measure important costs, starting with refusal outcomes; there is no universal
requirement to count every accepted cost. Conservative-refusal counting is that
same reporting work, not a second rule.

Permit justified platform/build-selected expectations, never a nearest-answer
choice after seeing Confit's output. Keep rules against concealing mismatches or
substituting another authority for the oracle; do not ban hashing as an incidental
tool merely because a hash alone is poor diagnostic evidence.

Approved exceptions have one home beside their comparison rule, linked from the
ledger; investigations and defects stay in the ledger. A severity-1/2 defect may
remain open while it is worked without becoming an accepted result. Differences in
genuinely unspecified aspects are not parity defects; unclassified differences remain unresolved, not silently
relabelled unspecified.
