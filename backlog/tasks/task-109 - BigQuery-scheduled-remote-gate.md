---
id: TASK-109
title: >-
  BigQuery scheduled remote gate — date-stamped pins, then unlock the printer's calls
status: To Do
assignee: []
created_date: '2026-08-13 20:30'
labels:
  - m-9
dependencies: []
type: feature
ordinal: 101000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 4 of the dialect-logical-plan spec. The BigQuery printer exists but
refuses every function call because no gate can verify a spelling — by
design (unprobed row = refusal). This task builds the gate: a small pinned
corpus run against real BigQuery on a schedule (not per-commit — the
service is metered and unversionable), pins date-stamped because the
oracle can move between runs.

Decisions parked for this phase, per the spec: the narrow-int
guard-expression question (INT64-only vs per-width overflow traps via
`ERROR()`), the null-safe-join expansion spelling, NUMERIC/BIGNUMERIC
fit rules. Needs credentials/billing from AmirHossein before any remote
run — blocked on his go for that part.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 scheduled gate exists (pinned mini-corpus, three-outcome accounting, date-stamped pins-bigquery/)
- [ ] #2 first probe batch lands; probed spellings unlock in the BQ printer, everything else stays refused by name
- [ ] #3 the narrow-int question is decided and recorded in the spec
- [ ] #4 kpis.md gets the bigquery column
<!-- AC:END -->
