# Decisions

This folder holds one decision record for each question of contract or
scope ([catalog contract](../../../packages/sql-transform/spec/native/catalog-contract.md)) to the owner. A record states the question,
the options, the provisional choice that the loop works under, and the
evidence. After the owner rules, the record states the ruling.

- [open/](open) holds the records that wait for a ruling:
  - [where a sum overflows on one side only](open/traps-where-a-sum-overflows.md)
    (2026-10-17): must the entry and its twin trap on the same rows after a
    step with a parity bound above 0?
- [closed/](closed) holds the owner's rulings. Each ruling below is of
  2026-10-06, unless the list gives a different date:
  - [a step with a parity bound in a
    composition](closed/bounded-steps-in-compositions.md) (2026-10-15): such
    a step may sit in any position. The check proves the bound where a
    derivation gives it, and reports the flip rate elsewhere.
  - [the worker environment](closed/worker-environment.md) (2026-10-15):
    the Nix dev shell of #428 gives uv and CPython, and no setup script for
    uv is necessary
  - [the parity bound of the matvec families](closed/matvec-parity-bound.md):
    the general form of the parity bound. Bit-exact is the default, and a
    bound above 0 serves only when the caller asks for it.
  - [the parity bound of `PowerTransformer`](closed/power-parity-bound.md)
  - [the parity bound of `AdditiveChi2Sampler`](closed/additive-chi2-parity-bound.md)
  - [where the twin raises](closed/tolerated-differences.md): the entry
    traps too, through an input guard
  - [sparse outputs](closed/sparse-outputs.md): the Python step densifies a
    sparse output
- One ruling in force for this loop is in the confit loop's folder. The
  owner made it before each loop had its own folder. Its amendment of
  2026-10-06 introduces the parity bound:
  - [the parity bounds of catalog entries](../../confit/decisions/closed/native-transform-parity-bounds.md)
- [research/](research/2026-10-06/README.md) holds the measurements and the
  derivations behind the Methodology and Recommendation of each closed
  record.

When the owner rules on a question, move its record from `open/` to
`closed/`. If the ruling is "not now", move it to `postponed/`, and name the
condition that opens it again. If the catalog contract or another spec
states the whole ruling, remove the record.
