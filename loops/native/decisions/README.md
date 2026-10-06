# Decisions for the native catalog

Each record holds one question of contract or scope ([goal.md](../goal.md)).
It gives the question, the options, the provisional choice of the loop, and
the ground. When the owner rules, the record moves from `open/` to `closed/`.

- [open/](open/): [may a step with a parity bound sit in a
  composition](open/bounded-steps-in-compositions.md).
- [closed/](closed/): the owner rulings of 2026-10-06.
  - [Where the twin raises](closed/tolerated-differences.md): the entry
    traps too, through an input guard.
  - [How close a matvec entry must be](closed/matvec-parity-bound.md): the
    parity bound and its general form.
  - [How close a power transform must be](closed/power-parity-bound.md).
  - [How close an additive chi2 sampler must
    be](closed/additive-chi2-parity-bound.md).
  - [The Python step densifies a sparse output](closed/sparse-outputs.md).
  - The first parity ruling is in confit's record,
    `loops/confit/decisions/closed/native-transform-parity-bounds.md`. Its
    amendment of 2026-10-06 introduces the parity bound.
- [research/](research/2026-10-06/README.md): the measurements and the
  derivations behind the Methodology and Recommendation of each closed
  record.
