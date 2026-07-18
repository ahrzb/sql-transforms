---
id: TASK-12
title: 'Conformance harness S2: allowlist extractor plus dual-engine runner'
status: To Do
assignee: []
created_date: '2026-07-18 15:38'
labels:
  - tests
  - conformance
milestone: m-5
dependencies: []
ordinal: 12000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Extractor pulls query records whose function set is a subset of the supported allowlist and are self-contained (no FROM/DDL/aggregate); runnable-by-construction, near-zero skips. Runner executes each through both serving engines (native + codegen) and live DataFusion, asserting values match (oracle-truth; decision-1 + decision-2). Seed allowlist = current shipped surface: upper/lower/trim/concat/coalesce/nullif/abs/round/substr/substring/struct + operators + cast to numeric/string/bool.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 extractor emits runnable cases for the current allowlist; near-zero skips
- [ ] #2 runner diffs both engines vs live DataFusion on values
- [ ] #3 value divergences become findings/tickets, not silent passes
<!-- AC:END -->
