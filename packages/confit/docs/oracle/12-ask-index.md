# Decision index

An ASK changes status only through an owner ruling or an explicit superseding product
decision. Shipped code is evidence, not a ruling.

## Resolved

| ask | result | authoritative detail |
|---|---|---|
| **ask: unshipped-never-compared** | **ruled:** normalization is outside the oracle answer and verdict; an unshipped feature fails or is classified, and raw equality changes only under a named reviewed bound | claim: unshipped-verdict and divergence: decimal-cast-artifact; numeric bounds remain governed by [the comparison contract](05-the-comparison-contract.md) |
| **ask: engine-fold-reading** | **superseded:** no optimizer reading was selected; the fold left the target and current removal remains an implementation gap | [trustworthy-fold decision](../decisions/trustworthy-fold.md) |
| **ask: frozen-row-order** | **superseded:** fold retirement removed the target freeze-order question | [trustworthy-fold decision](../decisions/trustworthy-fold.md) |

## Open

| ask | decision required | principal bindings |
|---|---|---|
| **ask: version-pin** | choose exact dependency pinning, runtime assertion, or both; separately decide whether to retain 1.5.5 or deliberately move to LTS through a reviewed reference change | claim: oracle-identity; claim: oracle-version-constant; claim: capture-outside-the-oracle; pins |
| **ask: threads-and-value-order** | decide the disposition of future retained order-sensitive families; pinning oracle `threads=1` remains a candidate change to the fixed identity, not a per-case override | claim: disposition-table; claim: threads-setting; ticket: threads-one-setting |
| **ask: refusal-cost-counting** | measure accepted severity-4 refusal cost or amend the RFC to call it unmeasured | claim: refusal-absorb; claim: countable-rung-four; divergence: bind-time-constant-refusals |
| **ask: opt-emulated-branch** | decide whether `OPT_EMULATED` should continue to boundary self-checks; it remains a finding, not agreement or coverage | claim: opt-emulated-classification; claim: unshipped-verdict |
| **ask: reason-code-visibility** | keep refusal reason codes internal or expose them in user-visible text | claim: refusal-message-prefixes |
| **ask: float-tolerance-list** | govern every exception to bit equality, including the edge domain required before the adopted but unimplemented float-reduction bound can ship, and decide whether `_type_delta`'s `UNSHIPPED` arm needs its own gate | [comparison contract](05-the-comparison-contract.md); claim: feature-in-flight; divergence: decimal-literal-typing; divergence: decimal-cast-rounding |
| **ask: exclusion-ratification** | ratify, narrow, or reject the proposed serving exclusions | [serving contract](../specs/serving-contract.md) |
| **ask: doc-twin-overstatement** | complete TASK-95 and totality checking, or narrow prose to enumerated twin coverage | claim: doc-twin-totality |
| **ask: unlisted-divergence** | decide whether every unlisted divergence is a bug by definition | claim: ledger-adjudication |
| **ask: tentative-bucket** | admit a measured-but-unruled tag or require classification before campaign closure | future threads disposition; divergence: phase-two-width-residuals; divergence: snapshot-baseline |
| **ask: baseline-as-evidence** | regenerate the campaign baseline, freeze it as named dated history, or replace it with dated reports | claim: regexp-fuzz-gate; divergence: snapshot-baseline; ask: width-residual-classification |
| **ask: width-residual-classification** | rerun and classify width residuals before treating the unreconstructible historical count as defects | divergence: phase-two-width-residuals; divergence: snapshot-baseline |
| **ask: match-count-ratchet** | ratify the shipped `MATCH_FLOOR = 547` mechanism as policy or keep it as implementation evidence, and define approved adjustments | claim: zero-fails-gate; claim: dialect-gate-oracle |
| **ask: proposed-rules-adoption** | rule independently on target-status vocabulary, countable cost, multi-answer sets, standing rejections, divergence placement, absolute severity rungs 1/2, countable rung four, and unspecified residuals | individual proposal sections in [ordering](03-nondeterminism.md), [verdicts](04-verdicts-agreement-abstention-refusal.md), [comparison](05-the-comparison-contract.md), [ledger](07-the-divergence-ledger.md), [severity](08-the-severity-ladder.md), and [campaigns](10-campaign-validity-and-blind-spots.md) |

Proposed KPIs and their acceptance decisions are indexed in
[success measures](../specs/success-measures.md); API, UDF, restriction, and exclusion
decisions are indexed in [the serving contract](../specs/serving-contract.md).

## Ledger rulings

A proposed ledger disposition is not an ASK ruling, and a blank ruling remains unruled.
Divergence: decimal-cast-artifact is ruled through ask: unshipped-never-compared. The
phase-two width residuals and snapshot baseline remain unruled; decimal literal typing
and cast rounding remain tied to ask: float-tolerance-list. No other status may be
inferred from tests, directories, or proposed text. See
[the divergence index](07-the-divergence-ledger.md).

## Stable names

Claims, ASK blocks, divergences, and tickets use semantic slugs. Retired numeric IDs
remain resolvable through [old ID aliases](13-old-ids.md). Retired claim:
multiset-default remains a tombstone, and closed or refuted ticket slugs remain in the
ticket register.
