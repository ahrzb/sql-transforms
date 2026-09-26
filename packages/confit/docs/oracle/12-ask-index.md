# Decision index

The owner accepted the recommendations in the [oracle policy decision](../decisions/oracle-policy.md)
on 2026-09-21. The questions below are resolved as policy; that does not mean all
implementation work is complete. Current gaps belong in the
[work register](11-proposed-tickets.md), not in repeated owner-choice questions.

## Resolved

### Reference, scope, and comparison

| Former question | Adopted decision | Definition |
|---|---|---|
| **ask: version-pin** | Keep 1.5.5; pin the oracle/test environment and assert the runtime version. Review upgrades separately. | [Reference enforcement](01-what-the-oracle-is.md) |
| **ask: threads-and-value-order** | No speculative global thread change. Require a justified contract before serving each order-sensitive family. | [Ordering](03-nondeterminism.md) |
| **ask: float-tolerance-list** | Explicitly approved numerical exceptions only; behavioral coverage without mandatory duplicate xfails; resolve the adopted reduction bound's domain before implementation. | [Comparison](05-the-comparison-contract.md) |
| **ask: exclusion-ratification** | Separate semantic scope, unimplemented features, explicit product/resource restrictions, and invalid inputs. Do not blanket-ratify current bans. | [Scope classification](../specs/serving-contract.md#scope-classification) |
| Output nullability, previously unindexed | Require truthful metadata, not identical DuckDB nullable flags. A non-null promise must be sound. | [Serving output](../specs/serving-contract.md#api-and-output-shape) |

### Reporting and evidence

| Former question | Adopted decision | Definition |
|---|---|---|
| **ask: refusal-cost-counting** | Retain and summarize the already-computed oracle outcome by refusal reason; do not make every oracle-serves refusal a correctness defect. | [Refusal reporting](04-verdicts-agreement-abstention-refusal.md) |
| **ask: opt-emulated-branch** | Stop at the primary `OPT_EMULATED` finding instead of allowing a later self-leg to replace it. | [Verdicts](04-verdicts-agreement-abstention-refusal.md) |
| **ask: reason-code-visibility** | Keep audit codes internal; public diagnostics remain actionable and construct-naming. | [Refusal diagnostics](04-verdicts-agreement-abstention-refusal.md) |
| **ask: doc-twin-overstatement** | State demonstrated test coverage and fill important behavioral gaps; no universal test-totality claim or registry requirement. | [Coverage evidence](07-the-divergence-ledger.md) |
| **ask: unlisted-divergence** | Judge against the contract, not ledger membership. Neither listing nor omitting an entry changes correctness. | [Divergence ledger](07-the-divergence-ledger.md) |
| **ask: tentative-bucket** | Allow visible, separately counted unresolved observations, never implicit passes or approved exceptions. | [Evidence classification](07-the-divergence-ledger.md) |
| **ask: baseline-as-evidence** | Freeze historical runs; use dated, provenance-bearing future runs with SQL and inputs, not seeds alone. | [Evidence lifecycle](09-version-bumps-and-mutability.md) |
| **ask: width-residual-classification** | Retire the unsupported historical count from current evidence; replay available stored cases or produce a labelled fresh measurement. | [Campaign evidence](10-campaign-validity-and-blind-spots.md) |
| **ask: match-count-ratchet** | Stable-corpus support may not decrease without a reviewed reason and affected cases; totals do not replace case-level checks. | [Corpus gates](10-campaign-validity-and-blind-spots.md) |
| **ask: acceptance-target** | Ratchets for stable corpora, reporting for generated campaigns, no arbitrary percentage target. | [Measurement policy](../specs/success-measures.md#measurement-policy) |
| **ask: kpi-set-change** | Keep C1–C5 and D1–D2; prioritize refusal-quality and unsupported-width reporting without new blocking KPIs. | [Measurement policy](../specs/success-measures.md#measurement-policy) |

### Process proposals

**ask: proposed-rules-adoption — RULED individually, not as a blanket package.**

| Proposal | Disposition |
|---|---|
| Mandatory target-status labels | Not adopted; use distinctions only where they clarify variation. |
| Count every accepted cost | Not adopted as a universal rule; disclose tradeoffs and measure important costs. |
| Multiple expected answers | Permit justified platform/build-selected expectations, never nearest-answer selection. |
| Blanket mechanism bans | Reject concealed mismatches and reference substitution, not incidental hashing. |
| Divergence placement | Approved exceptions live beside the comparison rule; the ledger links to them and holds investigations/defects. |
| Never retain severity-1/2 defects | Blanket policy rejected; an open defect remains a defect, not agreement. |
| Count conservative refusals | Merged into refusal-outcome reporting, not a second process rule. |
| Unspecified residuals | Differences in genuinely unconstrained aspects are not parity defects; unexplained differences remain unresolved. |

Details live in [ordering](03-nondeterminism.md), [verdicts](04-verdicts-agreement-abstention-refusal.md),
[comparison](05-the-comparison-contract.md), [ledger](07-the-divergence-ledger.md),
[severity](08-the-severity-ladder.md), and [campaigns](10-campaign-validity-and-blind-spots.md).

### Earlier decisions

| Question | Result |
|---|---|
| **ask: unshipped-never-compared** | Normalization stays outside the oracle answer and verdict; unsupported widths are classified, never cast into agreement. |
| **ask: engine-fold-reading** | Superseded by [fold retirement](../decisions/trustworthy-fold.md), not a choice of optimizer for a retained fold. |
| **ask: frozen-row-order** | Superseded by the same scope decision. |

## Remaining work, not reopened policy

- Establish the reduction bound's valid algorithms and edge domain before implementing it.
- Define and justify a contract when adding another order-sensitive value family.
- Version enforcement, refusal reporting, early mismatch return, nullability checks,
  and the adopted evidence/reporting improvements were implemented on 2026-09-26; see the
  [work register](11-proposed-tickets.md). The deferred proposals there remain open.
- Review individual restrictions and ledger entries against their evidence; the policy
  decision did not approve every historical disposition or numeric resource limit.

The remaining proposals for corpus-wide pin metadata, generic re-record tooling,
and mutability classes were not blanket-adopted. They remain optional engineering
proposals, not prerequisites to understanding the core oracle contract.

## Stable names

Old ASK slugs identify the decisions above rather than still-open questions.
[The citation map](13-old-ids.md) preserves numeric IDs and merged subjects.
