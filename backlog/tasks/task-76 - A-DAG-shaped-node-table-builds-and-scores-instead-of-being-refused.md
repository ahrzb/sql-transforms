---
id: TASK-76
title: >-
  A DAG-shaped node table builds and scores instead of being refused
status: To Do
assignee: []
created_date: '2026-08-08 03:00'
labels:
  - bug
  - validation
dependencies: []
documentation:
  - packages/confit/tests/test_known_divergences.py
type: bug
ordinal: 69000
---

## Description

<!-- SECTION:DESCRIPTION:BEGIN -->
**DISPUTED by the sweep's own verifiers** - one refuter broke the finding, one
could not, and it has NOT been adjudicated by hand. Treat the claim as unproven
until someone reproduces it.

The spec lists "a node reachable from two parents" among the build-time refusals
that name the offending row. The finder reports that a node table where two
internal nodes share a child builds and scores anyway.

If true, the fix is a reachability/parent-count check in `ensemble`. If false,
the spec overstates what is checked and the SPEC is what needs correcting.

Found by the 2026-08-08 adversarial sweep (6 finders over distinct surfaces,
then two independent refute-by-default verifiers per finding; 18 raw, 12
verified, 9 confirmed, 2 disputed, 1 refuted).

Pinned xfail-strict, so it cannot silently start or stop failing. Full context
for every finding is in the module docstring of
`packages/confit/tests/test_known_divergences.py`.
<!-- SECTION:DESCRIPTION:END -->

## Acceptance Criteria

<!-- AC:BEGIN -->
- [ ] #1 The claim is adjudicated by hand first - reproduced or refuted, in
      writing
- [ ] #2 If real: a DAG is refused naming the offending node id
- [ ] #3 If not real: the spec's refusal list is corrected instead
- [ ] #4 Whichever way it goes, every OTHER refusal the spec claims is checked
      by construction too, not assumed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The pinned test asserts a refusal, so it is red both when the DAG is accepted
and when it is accepted-but-harmless. Adjudicate before writing code.
<!-- SECTION:NOTES:END -->
