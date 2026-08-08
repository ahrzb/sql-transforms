---
id: TASK-76
title: >-
  A DAG-shaped node table builds and scores instead of being refused
status: Done
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
- [x] #1 The claim is adjudicated by hand first - reproduced or refuted, in
      writing
- [x] #2 If real: a DAG is refused naming the offending node id
- [x] #3 If not real: the spec's refusal list is corrected instead
- [x] #4 Whichever way it goes, every OTHER refusal the spec claims is checked
      by construction too, not assumed
<!-- AC:END -->

## Implementation Notes

<!-- SECTION:NOTES:BEGIN -->
The pinned test asserts a refusal, so it is red both when the DAG is accepted
and when it is accepted-but-harmless. Adjudicate before writing code.
<!-- SECTION:NOTES:END -->

## Adjudication (2026-08-08): REFUTED as a code bug — the SPEC was wrong

Reproduced by hand. A DAG-shaped table does build, and it scores
`p(x = .1, .4, .6, .9) = [10, 20, 10, 30]` — which is exactly what the table
describes. So the observation was right and the conclusion was not.

**A node with two parents is not a cycle.** The kernel already forces every
child index to STRICTLY FOLLOW its parent, which rules out cycles by
construction and is what makes traversal terminate without a depth counter.
Under that rule a shared child is an ordinary decision DAG: the walk from the
root still takes exactly one path, still terminates, and still yields the value
the table names. There is no wrong number to prevent, no library we target
emits one, and a refusal would cost a parent-count pass at build for nothing.

The spec's refusal list read "a cycle: a node reachable from two parents, or
unreachable from its tree's root", which conflates three different properties
into one bullet. Corrected in place, and the acceptance is now stated
explicitly and covered by a positive test so nobody re-derives this.

Note `unreachable from its tree's root` IS checked — the finder's report
lumped the two together, and only one half was ever missing.

### AC #4 done properly

Every other refusal the spec claims is now exercised by construction rather
than assumed, parametrised in `test_known_divergences.py`. All nine hold:
child index out of range; child preceding its parent (how a cycle would have
to be spelled); node unreachable from the root; leaf with children; split node
missing a child; feature beyond the declared width; node id out of dense
order; unknown agg; unknown link.
