---
id: DRAFT-26
title: >-
  Reverse frontends — parse Spark / BigQuery SQL into the dialect plan
status: Draft
assignee: []
created_date: '2026-08-13 20:30'
labels:
  - m-9
dependencies: []
type: feature
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
Phase 5 of the dialect-logical-plan spec, parked as a draft because it is
explicitly demand-driven — nothing earlier depends on it, and no user has
asked for "author in Spark" yet. When demand arrives: each reverse
frontend must satisfy L1 (round-trip) and its own-engine L2 (invisibility
on that engine's oracle) before it ships. Do not start unbidden.
<!-- SECTION:DESCRIPTION:END -->
