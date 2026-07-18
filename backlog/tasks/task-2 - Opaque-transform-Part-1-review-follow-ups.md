---
id: TASK-2
title: Opaque transform Part-1 review follow-ups
status: To Do
assignee: []
created_date: '2026-07-18 13:44'
labels:
  - rust
  - parity
  - opaque
dependencies: []
references:
  - docs/BACKLOG.md
ordinal: 2000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Follow-ups from the Part-1 (engine capability) review. Detail in docs/BACKLOG.md 'Opaque transform support'.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria
<!-- AC:BEGIN -->
- [ ] #1 Out-of-projection transformer calls: build-time guard/consistent resolve so Rust and DF agree
- [ ] #2 Add a single-field 1-in/1-out transformer differential parity case (oracle y[:,i] assumes 2-D)
- [ ] #3 Enforce or reconcile out_schema vs the transform natural output dtype
<!-- AC:END -->
