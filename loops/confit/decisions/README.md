# Decisions

This folder holds one record for each question. Each record states the question, the ruling or the options,
and the ground.

- [open/](open) holds records that wait for a ruling:
  - [Empty-static witness: trap order](open/empty-static-witness-trap-order.md)
  - [the oracle times out where confit traps first](open/oracle-timeout-before-a-trap.md)
- [closed/](closed) holds rulings in force:
  - [Oracle policy](closed/oracle-policy.md)
  - [static-only queries](closed/static-only-queries.md)
  - [C1 depth](closed/c1-depth.md)
  - [next query classes](closed/next-query-classes.md)
  - [native transform parity bounds](closed/native-transform-parity-bounds.md)
  - [empty-static join trap timing](closed/empty-static-join-trap-timing.md)
- [postponed/](postponed) holds questions decided "not now". Each record names the condition that reopens it:
  - [Public refusal codes](postponed/public-reason-codes.md)
  - [pin governance metadata](postponed/pin-governance-metadata.md)

When the owner rules on a question, move its record from `open/` to `closed/` or `postponed/`. If the goal,
the specs or the known limitations fully hold the outcome, remove the record.
