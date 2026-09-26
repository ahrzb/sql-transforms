# Oracle work register

The [oracle policy decision](../decisions/oracle-policy.md) settles the former
owner-choice questions. This register separates required follow-through from
unadopted mechanisms. It records work, not proof that code changed or a gate ran.

This is the active TODO list for oracle follow-through. The legacy `backlog/tasks/`
directory was removed by owner request; `TASK-*` citations elsewhere are historical
evidence, not active instructions or dependencies.

## Completed policy work

- [x] Record the [accepted decisions](../decisions/oracle-policy.md).
- [x] Resolve the discussed questions in the [decision index](12-ask-index.md).
- [x] Keep the core goal and oracle entry point unchanged while updating supporting rules.

## Follow-through from adopted policy

- [x] **ticket: version-assert** — Pin the oracle/test environment to 1.5.5 and
  assert `duckdb.__version__ == Oracle.VERSION` at startup. Keep upgrades separate
  and unrelated consumers unconstrained.
- [x] **ticket: stop-on-opt-emulated** — Preserve the primary `OPT_EMULATED`
  finding by stopping before boundary self-legs can replace it.
- [x] **ticket: split-refused-verdict** — Retain the already-computed oracle
  outcome and summarize by refusal reason. The historical name does not require
  a new verdict kind or make every oracle-serves refusal a correctness defect.
- [x] **ticket: truthful-output-nullability** — Enforce sound non-null promises,
  not equality with DuckDB's nullable flags. The existing checkers do not
  establish the adopted invariant in full.
- [x] **ticket: dated-campaign-evidence** — Keep old runs historical; record SQL,
  inputs, generator revision where applicable, engine revision, and reference
  configuration in dated future results. Seeds are not durable case identities.
- [x] **ticket: unresolved-observation-reporting** — Show unresolved observations
  separately from agreement and confirmed defects. Do not downgrade established
  mismatches or manufacture cases from retired counts.
- [x] **ticket: refusal-quality-reporting** — Report construct naming and
  actionability, not just prefix presence. No new blocking KPI is adopted.
- [x] **ticket: unsupported-width-reporting** — Report unsupported widths and
  generator reachability. An empty bucket is not proof of support; reuse
  behavioral coverage rather than require duplicate bookkeeping xfails.
- [x] **ticket: scope-inventory-classification** — Separate out-of-model queries,
  implementation gaps, explicit product/resource restrictions, and invalid
  inputs. Current syntax refusals do not justify permanent exclusions by themselves.
- [x] **ticket: behavioral-coverage-gaps** — Correct unsupported totality claims
  and fill important behavioral gaps, starting with the recorded schema-qualifier
  gap. Reuse adequate coverage; no one-test-per-document-entry registry.

The reduction bound's algorithms and edge domain remain prerequisites to its
implementation. Policy approval is not a proof of the bound. Corpus floor
changes must now carry a reviewed reason and the affected cases; the existing
floor mechanism does not need a second parallel policy system.

## Existing editorial work

These correct recorded inaccuracies without creating new oracle policy. All seven
were applied on 2026-09-26.

| Ticket | Correction |
|---|---|
| ~~**ticket: oracle-docstring-corrections**~~ done | Correct the stated `disable_optimizer` scope and remove the stale claim that `OPT_EMULATED` is purposeful coverage |
| ~~**ticket: exclusion-count-correction**~~ done | Correct `pins-first-methodology.md:89`: one excluded source covers two statements, not two sources |
| ~~**ticket: ambiguity-class-closed**~~ done | Correct the dated triage description of TASK-121, distinguishing its completed status from unchecked acceptance criteria; see [ledger evidence](07-the-divergence-ledger.md#evidence-notes) |
| ~~**ticket: severity-definition-merge**~~ done | Replace partial severity definitions with the common rule; no blanket never-retain-defects policy |
| ~~**ticket: clean-prefix-reconcile**~~ done | Reconcile `_CLEAN`'s two unprefixed messages with the three-prefix description without introducing a new public code API |
| ~~**ticket: string-budget-ground-fix**~~ done | Replace the false spelling-dependent explanation with measured deterministic DuckDB behavior and Confit's resource limit |
| ~~**ticket: fuzzer-gate-correction**~~ done | State that the campaign fuzzer is a manual CLI; machinery smoke tests do not establish zero campaign findings |

## Deferred implementation proposals

These mechanisms were not blanket-adopted by resolving the policy questions.
They are not owner-decision blockers for the settled contract.

| Ticket | Proposed mechanism and boundary |
|---|---|
| ~~**ticket: axiom-as-property**~~ done | Add the nondeterminism axiom to `properties.md`; the rule already has a canonical definition |
| ~~**ticket: phase-probing-in-methodology**~~ done | Add the existing PREPARE/EXECUTE and zero-row probe guidance to the methodology report |
| **ticket: uniform-pin-header** | Standardize metadata across the legacy pin corpus; provenance for new campaign runs does not imply a full retrospective conversion |
| **ticket: pin-decision-field** | Add a claim back-reference to every pin; mandatory per-pin governance metadata remains unadopted |
| **ticket: pin-field-token** | Add a general token for discriminator-dependent or unspecified fields; the semantic distinction does not require this encoding |
| **ticket: corpus-drift-report** | Generalize `pin_ast_shapes.py`'s reviewable diff to the whole pin corpus; reference upgrades still need separate review |
| ~~**ticket: match-count-single-home**~~ done | Generate dated display counts in one place; the adopted floor policy does not require generated documentation |
| ~~**ticket: coverage-triples**~~ done | Report `(operator, argument-type, edge-class)` coverage; no universal coverage-metadata scheme was adopted |
| ~~**ticket: per-kind-abstention-report**~~ done | Choose a reporting format for unanswered cases, preserve SQL before execution, and attribute timeout side; audit codes remain internal |
| **ticket: convert-unrunnable-pins** | Inventory and convert old pins that cannot be replayed mechanically; do not rewrite history as a fresh run |
| ~~**ticket: mined-corpus-stamp**~~ done | Add provenance during future corpus mining; existing optimizer-on expectations do not become optimizer-off evidence |
| ~~**ticket: verdict-tuple-test**~~ done | Historical name for missing finding/coverage checks; prefer observable reporting regressions over assertions about tuple membership |

## Closed or superseded mechanisms

| Ticket | Disposition |
|---|---|
| **ticket: threads-one-setting** | No speculative global pin adopted. Revisit settings only if implementing a retained family requires a justified reference-contract change. |
| **ticket: value-preserving-normalization** | Closed: normalization stays outside the answer and verdict; the cast was deleted and `UNSHIPPED` replaced it. |
| **ticket: static-only-schema-check** | Refuted by measurement: the current static-only path returns DuckDB rows and schema; its removal remains a separate implementation gap. |
| **ticket: fold-reading-decision** | Superseded by fold retirement, not a selected optimizer reading for a retained fold. |

## Historical evidence correction

The committed 2026-08-17 snapshot contained 7 `DIVERGE_OPT` seeds among 28 findings
when counted on 2026-08-25 (312, 812, 1196, 1563, 1564, 2174, 2805).
`known-limitations.md` said 8 (corrected to 7 on 2026-09-26), while
`2026-08-17-fuzz-triage.md:87, :89-94, :102` disagrees with itself. Correcting those
historical descriptions would not create a fresh campaign result. The
unreconstructible “79 of 84” residual count is retired from current evidence;
replay available stored cases or report a labelled new measurement instead.
