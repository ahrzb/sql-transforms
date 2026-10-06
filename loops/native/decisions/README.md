# Decisions

This folder holds one decision record for each question of contract or
scope ([goal.md](../goal.md)) to the owner. A record states the question,
the options, the provisional choice that the loop works under, and the
evidence. After the owner rules, the record states the ruling.

- [open/](open) holds the records that wait for a ruling:
  - [the ulp bound of the matvec families](open/matvec-parity-bound.md):
    the families whose `transform` is a matrix-vector product, such as
    `PCA`
  - [the ulp bound of `PowerTransformer`](open/power-parity-bound.md)
  - [the ulp bound of `AdditiveChi2Sampler`](open/additive-chi2-parity-bound.md)
  - [the tolerated differences](open/tolerated-differences.md): may an
    entry answer where the twin raises an error?
  - [sparse outputs](open/sparse-outputs.md): should the Python step
    densify a sparse output, such as the output of `OneHotEncoder()`?
  - [the worker environment](open/worker-environment.md): a setup script
    that installs a current uv and Python 3.14.8
- One ruling in force for this loop is in the confit loop's folder. The
  owner made it before each loop had its own folder:
  - [the parity bounds of catalog entries](../../confit/decisions/closed/native-transform-parity-bounds.md)

When the owner rules on a question, move its record from `open/` to
`closed/`. If the ruling is "not now", move it to `postponed/`, and name the
condition that opens it again. If goal.md or the specs state the whole
ruling, remove the record.
