# Proposed ticket register

A row records proposed work, not authorization. `editorial` means correcting a measured
falsehood; an ASK or `owner ruling` is an unresolved blocker.

## Active proposals

| ticket | proposed work | basis / blocker |
|---|---|---|
| **ticket: oracle-docstring-corrections** | correct the stated `disable_optimizer` scope and remove the stale claim that `OPT_EMULATED` is purposeful coverage | claim: disable-optimizer-scope; claim: opt-emulated-classification / editorial |
| **ticket: version-assert** | assert `duckdb.__version__ == Oracle.VERSION` beside oracle setup | claim: oracle-version-constant / ask: version-pin |
| **ticket: axiom-as-property** | add the nondeterminism axiom to `properties.md` and cite it from its three sites | claim: nondeterminism-axiom / owner ruling |
| **ticket: exclusion-count-correction** | correct `pins-first-methodology.md:89`: one excluded source covers two statements, not two sources | claim: statistics-dependent-exclusion / editorial |
| **ticket: split-refused-verdict** | split `REFUSED` into oracle-traps and oracle-serves; make the latter interesting | claim: refusal-absorb / ask: refusal-cost-counting |
| **ticket: phase-probing-in-methodology** | document PREPARE/EXECUTE and zero-row phase probes | claim: phase-separated-probes / owner ruling |
| **ticket: uniform-pin-header** | add version, settings profile, capture date, and harness commit | claim: pin-provenance / owner ruling |
| **ticket: pin-decision-field** | record the evidentiary claim slug in every pin | claim: pin-back-reference / owner ruling |
| **ticket: pin-field-token** | mark under-determined or discriminator-dependent fields | claim: under-determined-token / owner ruling |
| **ticket: ambiguity-class-closed** | correct the dated triage report: the ambiguous-reference class is TASK-121, Done, though its acceptance criteria remain unchecked | [ledger evidence](07-the-divergence-ledger.md#evidence-notes) / editorial |
| **ticket: severity-definition-merge** | replace partial ladder copies with claim: severity-ladder | editorial |
| **ticket: corpus-drift-report** | generalize `pin_ast_shapes.py`'s reviewable diff to the pin corpus | claim: re-record-diff-report / owner ruling, independent of ask: version-pin |
| **ticket: match-count-single-home** | generate dated displayed match counts in one place; keep the test floor separate | claim: zero-fails-gate / ask: match-count-ratchet for lasting policy |
| **ticket: coverage-triples** | report `(operator, argument-type, edge-class)` coverage | claim: coverage-denominator / owner ruling |
| **ticket: per-kind-abstention-report** | report abstention rates, record SQL before execution, and attribute timeout side in the runner | claim: abstention-rate; claim: timeout-attribution / ask: reason-code-visibility |
| **ticket: threads-one-setting** | consider pinning oracle `threads=1` to support future retained order-sensitive families | claim: threads-setting / ask: threads-and-value-order; an unadopted change to the fixed identity, not a per-case override |
| **ticket: clean-prefix-reconcile** | reconcile `_CLEAN`'s two unprefixed messages with the three-prefix rule | claim: refusal-message-prefixes / editorial |
| **ticket: string-budget-ground-fix** | replace the false spelling-dependent reason with the measured deterministic DuckDB behavior | claim: keep-entry-reason; divergence: string-builder-budget / editorial |
| **ticket: convert-unrunnable-pins** | inventory and convert pins that cannot be replayed mechanically | claim: pin-re-runnability / owner ruling |
| **ticket: mined-corpus-stamp** | stamp DuckDB version, date, and settings profile during mining | claim: mined-corpus-provenance / owner ruling |
| **ticket: fuzzer-gate-correction** | state that the campaign fuzzer is a manual CLI and smoke tests gate machinery, not zero findings | claim: regexp-fuzz-gate / editorial |
| **ticket: verdict-tuple-test** | test `fuzz.runner.INTERESTING` and `COVERED`, whose memberships currently lack a direct test | claims: contract-surface-gap, optimizer-bracket, opt-emulated-classification, abstention-reporting, coverage-accounting / owner ruling |

The fold decision did not adopt `threads=1`; it does not settle that candidate
for future retained families.

## Closed, refuted, or superseded slugs

These concise entries remain so old references resolve; the fold history lives only in
[the decision record](../decisions/trustworthy-fold.md).

| ticket | result |
|---|---|
| **ticket: value-preserving-normalization** | **closed:** ask: unshipped-never-compared kept normalization outside the answer and verdict; the cast was deleted and `UNSHIPPED` replaced it |
| **ticket: static-only-schema-check** | **refuted by measurement:** the current static-only path returns DuckDB rows and schema; that path is nevertheless outside the target and remains an implementation gap until removed |
| **ticket: fold-reading-decision** | **superseded, not implemented:** fold retirement removed the target optimizer-choice question |

## Dated evidence correction

The committed 2026-08-17 snapshot contained 7 `DIVERGE_OPT` seeds among 28 findings when
counted on 2026-08-25 (312, 812, 1196, 1563, 1564, 2174, 2805).
`known-limitations.md:248-249` says 8, while
`2026-08-17-fuzz-triage.md:87, :89-94, :102` disagrees with itself. Correcting those
historical documents would not create a fresh campaign result.
