---
id: TASK-102
title: >-
  Align || with a foldable NULL operand to SQLNULL (int32)
status: Done
assignee: []
created_date: '2026-08-13 12:00'
labels:
  - m-8
dependencies: []
type: task
ordinal: 94000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
DECIDED 2026-08-13 (option B, bug-for-bug; xfail-strict pin in
test_integer_widths.py). Measured (DESCRIBE agrees with execution on
every spelling — bind-time, not optimizer): `||` with an operand that
is a foldable constant evaluating to NULL — bare NULL,
CAST(NULL AS VARCHAR), upper(NULL), nullif('a','a'), constant CASE —
collapses the whole expression to SQLNULL, INTEGER at the boundary.
Concat-specific: `+`, LIKE, unary minus and function calls keep their
promoted type; a CASE holding a column is not foldable and stays
VARCHAR. Supersedes the §5 "NULL || NULL types as VARCHAR" contract
row (delete it; the row also understated the divergence). Fix is a
fold-pass arm: a concat operand folded to a NULL literal retypes the
node to null_of(int32) — value-correct, || propagates NULL. Spec:
packages/confit/docs/specs/2026-08-13-bind-fold-alignment-design.md.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 the TASK-102 xfail-strict pin flips; the measured 5-spelling battery pinned live-oracle
- [ ] #2 column-bearing CASE operand stays VARCHAR (foldability boundary pinned)
- [ ] #3 §5 known-limitations row deleted (RFC'd removal — this decision)
- [ ] #4 20k campaign: no new classes
<!-- AC:END -->

## Notes

2026-08-13 hygiene: merged to master (7240d4a); pin flipped, no || class in the 2026-08-13 20k campaign.
