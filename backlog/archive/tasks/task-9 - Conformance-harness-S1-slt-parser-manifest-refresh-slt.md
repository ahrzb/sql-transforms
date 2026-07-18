---
id: TASK-9
title: 'Conformance harness S1: slt parser + manifest + --refresh-slt'
status: To Do
assignee: []
created_date: '2026-07-18 15:37'
labels:
  - tests
  - conformance
milestone: m-5
dependencies: []
ordinal: 9000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Test-only package. Parse DataFusion sqllogictest .slt files, extracting query records only (no DDL/statement). Manifest {our_function -> source file, upstream revision, case hashes}. --refresh-slt re-fetches upstream at a pinned ref and rebumps. Seed corpus = expr.slt @ DataFusion 67947b6 (48.0.0-rc2). Zero engine changes.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 parser yields query records from expr.slt (no DDL/statement)
- [ ] #2 manifest maps our_function -> (source file, revision, case hashes)
- [ ] #3 --refresh-slt re-fetches upstream at a given ref and rebumps the manifest
<!-- AC:END -->
