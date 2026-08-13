---
id: TASK-104
title: >-
  Join node — general joins bound, verified, printed (closes the v0 node set)
status: Done
assignee: []
created_date: '2026-08-13 20:30'
labels:
  - m-9
dependencies: []
type: feature
ordinal: 96000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Design: `docs/superpowers/specs/2026-08-13-dialect-join-node-design.md`
(approved 2026-08-13). The general case in one slice: INNER/LEFT/RIGHT/
FULL/CROSS, ON as a bound boolean expression (`Join{left, right, kind,
on: Option<Expr>}` — supersedes the epic spec's JoinKey sketch), USING/
NATURAL desugared in the binder, comma-joins as CROSS, chained joins
left-nested. Printers move to qualified ordinal refs (approach A) so
dup-named join outputs print; SEMI/ANTI/ASOF/APPLY/positional refuse by
name. ~90 of 678 corpus statements contain a join form.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [x] #1 Join node + verifier rules (on XOR Cross, BOOLEAN predicate, combined-schema ordinals); canonical text round-trips
- [x] #2 frontend binds ON/USING/NATURAL/comma/chained joins per the spec's binder rules; ambiguity and duplicate qualifiers are Bind errors
- [x] #3 printers emit qualified ordinal refs; frontend∘printer fixpoint holds on SQL-derived join plans
- [x] #4 L2 gate floor rises from 235 (record old → new in the commit message); L3 spark floor rises from 213
- [x] #5 three-outcome accounting: no statement changes class downward, zero wrong answers; divergences become xfail pins + findings
<!-- AC:END -->
