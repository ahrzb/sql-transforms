# Decisions

One record per question. Each states the question, the ruling or the options, and
the ground.

- [open/](open/): waiting on a ruling. [Next query classes](open/next-query-classes.md),
  [C1 depth](open/c1-depth.md),
  [native transform parity bounds](open/native-transform-parity-bounds.md).
- [closed/](closed/): rulings in force. [Oracle policy](closed/oracle-policy.md),
  [static-only queries](closed/static-only-queries.md).
- [postponed/](postponed/): decided "not now", each with the condition that reopens
  it. [Public refusal codes](postponed/public-reason-codes.md),
  [pin governance metadata](postponed/pin-governance-metadata.md).

A ruled question moves from `open/` to `closed/` or `postponed/`; one whose outcome
is fully written into the goal, specs or known limitations is removed.
