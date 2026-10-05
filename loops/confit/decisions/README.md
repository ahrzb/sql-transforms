# Decisions

One record per question. Each states the question, the ruling or the options, and
the ground.

- `open/`: waiting on a ruling (none at the moment; a new record starts here).
- [closed/](closed): rulings in force. [Oracle policy](closed/oracle-policy.md),
  [static-only queries](closed/static-only-queries.md),
  [C1 depth](closed/c1-depth.md),
  [next query classes](closed/next-query-classes.md),
  [native transform parity bounds](closed/native-transform-parity-bounds.md),
  [empty-static join trap timing](closed/empty-static-join-trap-timing.md).
- [postponed/](postponed): decided "not now", each with the condition that reopens
  it. [Public refusal codes](postponed/public-reason-codes.md),
  [pin governance metadata](postponed/pin-governance-metadata.md).

A ruled question moves from `open/` to `closed/` or `postponed/`; one whose outcome
is fully written into the goal, specs or known limitations is removed.
