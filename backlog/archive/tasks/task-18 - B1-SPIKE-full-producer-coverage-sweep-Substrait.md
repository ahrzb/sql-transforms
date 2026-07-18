---
id: TASK-18
title: 'B1 SPIKE: full producer-coverage sweep (Substrait)'
status: To Do
assignee: []
created_date: '2026-07-18 16:08'
labels:
  - epic-b
milestone: m-6
dependencies: []
ordinal: 18000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Push the FULL rewrite output (every supported function/cast) through the datafusion-python Substrait producer; catalog coverage gaps (unnest known-blocked). Output GATES B2 (IR choice). Cheap, highest-signal. See doc-4.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 producer-coverage report over the full current rewrite surface; gaps catalogued
<!-- AC:END -->
