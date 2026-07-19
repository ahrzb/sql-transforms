---
id: DRAFT-6
title: Outer joins in authored SQL
status: Draft
assignee: []
created_date: '2026-07-19 01:09'
labels:
  - sql-surface
dependencies: []
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Decide if/how LEFT/RIGHT/FULL OUTER joins matter for real feature-engineering SQL before investing -- not supported in the native interpreter today. Prioritize by what authoring (goal 1) actually needs. Split from the CASE WHEN item (CASE is ticketed as TASK-27 native / TASK-30 codegen); outer joins stay the separate, still-parked concern.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 decision: are outer joins in scope, driven by real authoring demand; if yes, scope the interpreter + rewrite work
<!-- AC:END -->
