# Decisions

This folder holds one decision record for each question of contract or
scope ([catalog contract](../../../packages/sql-transform/spec/native/catalog-contract.md)) to the owner. A record states the question,
the options, the provisional choice that the loop works under, and the
evidence. After the owner rules, the record states the ruling.

- [open/](open) holds the records that wait for a ruling:
  - [a step with a parity bound in a
    composition](open/bounded-steps-in-compositions.md): may such a step sit
    in a Pipeline, a ColumnTransformer or a FeatureUnion?
  - [the worker environment](open/worker-environment.md): a setup script
    that installs a current uv and Python 3.14.8
- [closed/](closed) holds the owner's rulings of 2026-10-06:
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
