# Citation map

These are navigation aliases, not additional requirements. Old names retain their
subjects when definitions move or merge; no identifier is reassigned.

`goal:` and `exclusion:` name product intent; `kpi:` names a measure; `claim:` can
name a rule or an observation. `ask:` names a decision, `divergence:` a recorded
difference, and `ticket:` proposed work. `gap:` and `finding:` name dated report
entries. Status comes from the text, not the prefix: evidence does not adopt a
proposal, and an open decision is not a rule.

## Former goal sections

Historical reports use these names for sections formerly in `goal.md`.

| Former section | Current location |
|---|---|
| `engine-purpose` | [Goal](../goal.md) |
| `public-surface` | [API and output shape](../specs/serving-contract.md#api-and-output-shape) |
| `scope-edge` | [Scope and restrictions](../specs/serving-contract.md#scope-and-restrictions) |
| `scope-redirects` | [Implementation gap ledger](../reports/2026-09-02-goal-baseline.md#gap-ledger) |
| `beyond-duckdb` | [UDF and model boundary](../specs/serving-contract.md#udf-and-model-boundary) |
| `acceptance-frame` | [Acceptance and measurement](../specs/success-measures.md#acceptance-and-measurement) |
| `four-yardsticks` | [Measurement methods](../specs/success-measures.md#measurement-methods) |
| `measurement-and-kpis` | [Success measures](../specs/success-measures.md) |
| `standing-law` | [Standing rule](../specs/success-measures.md#standing-rule) |
| `controls-in-force` | [Correctness controls](../specs/success-measures.md#correctness-controls) |
| `drives-in-force` | [Optimization measures](../specs/success-measures.md#optimization-measures) |
| `enforcing-suites` | [Controls and their gates](../specs/success-measures.md#correctness-controls) |
| `proposed-kpis` | [Proposed additions](../specs/success-measures.md#proposed-additions-and-open-decisions) |
| `ask-index` | [Measurement decisions](../specs/success-measures.md#proposed-additions-and-open-decisions) · [Scope decision](../specs/serving-contract.md#open-scope-decision) |
| `document-set` | [Supporting specifications](../goal.md#supporting-specifications) |

## Moved or merged goal identifiers

| Former identifier | Current location |
|---|---|
| `goal: serving-without-skew`, `goal: two-outcome-contract` | [Required behavior](../goal.md#required-behavior) |
| `goal: pack-time-only-work` | [Construction-time preparation](../goal.md) |
| `goal: growing-accepted-surface`, `goal: request-latency-budget` | [Correctness and optimization priorities](../goal.md#required-behavior) |
| `goal: engine-half-only` | [Package ownership](../goal.md#scope) |
| `claim: model-surface-split`, `claim: udf-parity-is-still-the-oracle` | [UDF and model boundary](../specs/serving-contract.md#udf-and-model-boundary) |
| `claim: kpi-pointers-resolve` | [Gate evidence](../specs/success-measures.md#measurement-methods); a resolving path does not establish execution |
| `claim: float-reduction-bound` | [Comparison contract](05-the-comparison-contract.md) |
| `claim: sklearn-is-the-reference` | [Independent model references](../specs/success-measures.md#transformer-parity-c4) |
| `exclusion: whole-relation-shapes`, `exclusion: per-row-general-work`, `exclusion: resource-ceilings`, `exclusion: optimizer-on-answers`, `exclusion: statistics-dependent-kernels` | [Scope and proposed grounds](../specs/serving-contract.md#scope-and-restrictions) |
| `exclusion: multiplicity-by-default` | [Output multiplicity](../specs/serving-contract.md#api-and-output-shape) |
| `exclusion: unshipped-decimal-arithmetic`, `exclusion: wide-integer-lanes`, `exclusion: non-scalar-values`, `exclusion: parse-divergence-guards` | [Gap ledger](../reports/2026-09-02-goal-baseline.md#gap-ledger); each retains its subject slug under `gap:`, not as a permanent exclusion |
| `kpi: training-round-trip`, `kpi: engine-parity`, `kpi: binding-parity`, `kpi: transformer-parity`, `kpi: no-third-mode` | [C1–C5](../specs/success-measures.md#correctness-controls) |
| `kpi: coverage-ladder`, `kpi: serving-latency` | [D1–D2](../specs/success-measures.md#optimization-measures) |
| `kpi: acceptance-rate`, `kpi: findings-per-campaign`, `kpi: unshipped-burndown`, `kpi: named-refusal-share`, `kpi: ladder-ratchet`, `kpi: bench-refresh-cadence` | [Proposed measures](../specs/success-measures.md#proposed-additions-and-open-decisions) |
| `ask: acceptance-target`, `ask: kpi-set-change` | [Open measurement decisions](../specs/success-measures.md#proposed-additions-and-open-decisions) |
| `ask: exclusion-ratification` | [Open scope decision](../specs/serving-contract.md#open-scope-decision) |
| `ask: kpis-absorb-or-defer` | Historical ruling consolidated the KPI set; its definitions now live in [success measures](../specs/success-measures.md), with the set unchanged |

## Merged oracle identifiers

| Earlier identifier | Current location |
|---|---|
| `claim: pseudo-oracle`, `claim: oracle-identity` | [Oracle definition](README.md) |
| `claim: duckdb-three-roles` | [Reference and nearby uses](01-what-the-oracle-is.md); the retired fold's alternatives remain in the [decision record](../decisions/trustworthy-fold.md) |
| `claim: row-limit-refusal`, `claim: build-vs-build-repeatability` | Retired-fold concerns superseded by the [scope decision](../decisions/trustworthy-fold.md); historical measurements remain in [TASK-128](../../../../backlog/tasks/task-128%20-%20A-row-limit-without-ORDER-BY-freezes-an-answer-DuckDB-itself-cannot-repeat.md) and [TASK-129](../../../../backlog/tasks/task-129%20-%20Define-order-sensitivity-per-query-and-make-the-oracle-honor-it.md) |
| `claim: multiset-default` | Retired duplicate of [compare modes](03-nondeterminism.md#row-ordering) and [value canonicalization](05-the-comparison-contract.md#rows-and-values) |
| `ask: engine-fold-reading`, `ask: frozen-row-order`, `ask: unshipped-never-compared` | [Resolved decisions](12-ask-index.md#resolved) |

## Numeric aliases

Earlier PRs, tickets, reports, and `confit/oracle.py` (`ASK-1`) use these IDs.
Each keeps its original semantic subject below.

### Claims

| was | cited now as |
|---|---|
| `ORC-01` | `claim: pseudo-oracle` |
| `ORC-02` | `claim: oracle-identity` |
| `ORC-03` | `claim: optimizer-on-reading` |
| `ORC-04` | `claim: unoptimized-verifier` |
| `ORC-05` | `claim: disable-optimizer-scope` |
| `ORC-06` | `claim: contract-surface-gap` |
| `ORC-07` | `claim: no-raw-connections` |
| `ORC-08` | `claim: optimizer-flip-in-place` |
| `ORC-09` | `claim: oracle-version-constant` |
| `ORC-10` | `claim: reproduce-not-fix` |
| `ORC-11` | `claim: enumerated-quirks` |
| `ORC-12` | `claim: unlisted-oddity` |
| `ORC-13` | `claim: nondeterminism-axiom` |
| `ORC-14` | `claim: disposition-table` |
| `ORC-15` | `claim: target-status-vocabulary` |
| `ORC-16` | `claim: serving-row-order` |
| `ORC-17` | `claim: row-limit-refusal` |
| `ORC-18` | `claim: compare-modes` |
| `ORC-19` | `claim: join-output-order` |
| `ORC-20` | `claim: statistics-dependent-exclusion` |
| `ORC-21` | `claim: evaluation-path-disagreement` |
| `ORC-22` | `claim: build-vs-build-repeatability` |
| `ORC-23` | `claim: verdict-taxonomy` |
| `ORC-24` | `claim: optimizer-bracket` |
| `ORC-25` | `claim: opt-emulated-classification` |
| `ORC-26` | `claim: abstention-reporting` |
| `ORC-27` | `claim: logged-fallback` |
| `ORC-28` | `claim: coverage-accounting` |
| `ORC-29` | `claim: refusal-message-prefixes` |
| `ORC-30` | `claim: refusal-absorb` |
| `ORC-31` | `claim: countable-cost` |
| `ORC-32` | `claim: float-bit-equality` |
| `ORC-33` | `claim: signed-zero` |
| `ORC-34` | `claim: modulo-nan-sign` |
| `ORC-35` | `claim: multi-answer-sets` |
| `ORC-36` | `claim: multiset-default` (retired; see merged oracle identifiers above) |
| `ORC-37` | `claim: duplicate-name-dedup` |
| `ORC-38` | `claim: schema-comparison` |
| `ORC-39` | `claim: error-texts` |
| `ORC-40` | `claim: backend-agreement` |
| `ORC-41` | `claim: unadopted-mechanisms` |
| `ORC-42` | `claim: pin-as-data` |
| `ORC-43` | `claim: no-semantics-from-memory` |
| `ORC-44` | `claim: over-generalized-summary` |
| `ORC-45` | `claim: phase-separated-probes` |
| `ORC-46` | `claim: pin-provenance` |
| `ORC-47` | `claim: generator-version-stamps` |
| `ORC-48` | `claim: pin-back-reference` |
| `ORC-49` | `claim: under-determined-token` |
| `ORC-50` | `claim: keep-vs-change` |
| `ORC-51` | `claim: strict-xfail` |
| `ORC-52` | `claim: empty-change-ledger` |
| `ORC-53` | `claim: keep-entry-reason` |
| `ORC-54` | `claim: doc-twin-totality` |
| `ORC-55` | `claim: ledger-adjudication` |
| `ORC-56` | `claim: divergence-placement` |
| `ORC-57` | `claim: severity-ladder` |
| `ORC-58` | `claim: directional-rungs` |
| `ORC-59` | `claim: countable-rung-four` |
| `ORC-60` | `claim: bump-object` |
| `ORC-61` | `claim: re-record-diff-report` |
| `ORC-62` | `claim: diff-triage-classes` |
| `ORC-63` | `claim: mutability-classes` |
| `ORC-64` | `claim: changed-pin-record` |
| `ORC-65` | `claim: unspecified-residuals` |
| `ORC-66` | `claim: regexp-fuzz-gate` |
| `ORC-67` | `claim: zero-fails-gate` |
| `ORC-68` | `claim: blind-spots` |
| `ORC-69` | `claim: metamorphic-self-legs` |
| `ORC-70` | `claim: coverage-denominator` |
| `ORC-71` | `claim: abstention-rate` |
| `ORC-72` | `claim: fit-serving-oracle` |
| `ORC-73` | `claim: dialect-gate-oracle` |
| `ORC-74` | `claim: duckdb-three-roles` |
| `ORC-75` | `claim: threads-setting` |
| `ORC-76` | `claim: cbrt-ulp-tolerance` |
| `ORC-77` | `claim: corpus-exclusion-sets` |
| `ORC-78` | `claim: timeout-attribution` |
| `ORC-79` | `claim: refusal-grounds` |
| `ORC-80` | `claim: feature-in-flight` |
| `ORC-81` | `claim: platform-libm` |
| `ORC-82` | `claim: oracle-extracted-tables` |
| `ORC-83` | `claim: interpreter-backend` |
| `ORC-84` | `claim: standing-rejections` |
| `ORC-85` | `claim: pin-re-runnability` |
| `ORC-86` | `claim: capture-outside-the-oracle` |
| `ORC-87` | `claim: mined-corpus-provenance` |
| `ORC-88` | `claim: remeasure-guard` |
| `ORC-89` | `claim: campaign-as-acceptance` |
| `ORC-90` | `claim: repr-equality` |
| `ORC-91` | `claim: one-door-bypass` |
| `ORC-92` | `claim: unshipped-verdict` |
| `ORC-93` | `claim: native-tables` |

### ASK blocks

| was | cited now as |
|---|---|
| `ASK-1` | `ask: version-pin` |
| `ASK-2` | `ask: frozen-row-order` |
| `ASK-3` | `ask: refusal-cost-counting` |
| `ASK-4` | `ask: opt-emulated-branch` |
| `ASK-5` | `ask: reason-code-visibility` |
| `ASK-6` | `ask: float-tolerance-list` |
| `ASK-7` | `ask: doc-twin-overstatement` |
| `ASK-8` | `ask: unlisted-divergence` |
| `ASK-9` | `ask: tentative-bucket` |
| `ASK-10` | `ask: width-residual-classification` |
| `ASK-11` | `ask: match-count-ratchet` |
| `ASK-12` | `ask: unshipped-never-compared` |
| `ASK-13` | `ask: threads-and-value-order` |
| `ASK-14` | `ask: baseline-as-evidence` |
| `ASK-15` | `ask: proposed-rules-adoption` |
| `ASK-16` | `ask: engine-fold-reading` |

### Divergence-ledger rows

| was | cited now as |
|---|---|
| `D1` | `divergence: dedup-on-both-sides` |
| `D2` | `divergence: approximate-error-text` |
| `D3` | `divergence: ilike-nul` |
| `D4` | `divergence: trap-elision` |
| `D5` | `divergence: nan-sign-per-platform` |
| `D6` | `divergence: schema-qualifiers` |
| `D7` | `divergence: decimal-literal-typing` |
| `D8` | `divergence: decimal-cast-rounding` |
| `D9` | `divergence: bind-time-constant-refusals` |
| `D10` | `divergence: regex-size-guard` |
| `D11` | `divergence: narrow-lane-overflow` |
| `D12` | `divergence: decimal-cast-artifact` |
| `D13` | `divergence: phase-two-width-residuals` |
| `D14` | `divergence: string-builder-budget` |
| `D15` | `divergence: arrow-batch-ceiling` |
| `D16` | `divergence: snapshot-baseline` |

### Proposed tickets

| was | cited now as |
|---|---|
| `T-1` | `ticket: oracle-docstring-corrections` |
| `T-2` | `ticket: version-assert` |
| `T-3` | `ticket: axiom-as-property` |
| `T-4` | `ticket: exclusion-count-correction` |
| `T-5` | `ticket: split-refused-verdict` |
| `T-6` | `ticket: phase-probing-in-methodology` |
| `T-7` | `ticket: uniform-pin-header` |
| `T-8` | `ticket: pin-decision-field` |
| `T-9` | `ticket: pin-field-token` |
| `T-10` | `ticket: ambiguity-class-closed` |
| `T-11` | `ticket: severity-definition-merge` |
| `T-12` | `ticket: corpus-drift-report` |
| `T-13` | `ticket: match-count-single-home` |
| `T-14` | `ticket: coverage-triples` |
| `T-15` | `ticket: per-kind-abstention-report` |
| `T-16` | `ticket: threads-one-setting` |
| `T-17` | `ticket: clean-prefix-reconcile` |
| `T-18` | `ticket: value-preserving-normalization` |
| `T-19` | `ticket: string-budget-ground-fix` |
| `T-20` | `ticket: convert-unrunnable-pins` |
| `T-21` | `ticket: mined-corpus-stamp` |
| `T-22` | `ticket: fuzzer-gate-correction` |
| `T-23` | `ticket: fold-reading-decision` |
| `T-24` | `ticket: verdict-tuple-test` |
| `T-25` | `ticket: static-only-schema-check` |

