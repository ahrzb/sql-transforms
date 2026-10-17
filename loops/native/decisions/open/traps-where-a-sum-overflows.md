# Where a sum overflows on one side only, must the traps agree?

**Question.** In a composition, a step with a parity bound above 0 can come
before a step that validates its input. On rows of values near the largest
double, the entry and its twin can then trap on different rows. May they?

**What it decides.** It decides what `native.check` holds on such rows, and
what the catalog contract claims there.

- Since T59, a step with a parity bound above 0 can feed later steps, for
  example in `make_pipeline(PCA(), StandardScaler())`. The owner ruled this
  in [bounded-steps-in-compositions.md](../closed/bounded-steps-in-compositions.md).
- A later step can validate its input. For example, the validation of
  `StandardScaler` raises an error on an infinite value.
- The twin of a projection such as `PCA` sums the products x_i·c_ki of each
  output field in the order of its BLAS kernel. BLAS is the library of
  linear algebra that numpy calls. That order can change with the number of
  rows in the call. The entry sums the same products in a balanced tree
  (`project.py`).
- The parity bound of each family assumes that no partial sum overflows
  (`project.py`). A partial sum can overflow only on a row where some
  product x_i·c_ki is at least DBL_MAX/n. DBL_MAX is the largest double,
  about 1.8e308, and n is the number of features.
- On such a row, a sum can overflow to infinity in one order and stay
  finite in the other. Then the validation of the later step raises an
  error on one side only.
- The ruling on where the twin raises
  ([tolerated-differences.md](../closed/tolerated-differences.md)) asks
  for two things. The entry traps where the validation of its twin raises,
  and the entry never traps where its twin answers. On such a row, no entry
  can do both in all calls.

**Evidence.** The scripts and the data are in
`/mnt/project-files/transforms-loop/t59/`.

- The twin does not agree with itself on such rows. `band.py` fitted 80
  pipelines of a projection and a `StandardScaler`. The projection was a
  `PCA` (42 fits) or a `GaussianRandomProjection` (38 fits), with 2 to 512
  features and 1 to 16 components. Each fit served 400 rows, 32,000 rows
  in all. Each value of a row was 5% to 100% of DBL_MAX, with a random
  sign.
- The twin raised on 16,198 rows when it read each row alone. It raised on
  15,385 rows when it read each row in a call with 63 rows drawn as the
  fit, and on 17,103 rows in a call with 1,023 such rows. The 63 and the
  1,023 rows make the twin raise on none of them alone.
- On 2,137 rows, the twin raised alone and answered in the call of 64
  rows, or answered alone and raised in that call. 56 of the 80 fits had
  such rows.
- The entry trapped on 15,007 rows. On 721 of them the twin answered alone,
  and on 1,912 other rows the entry answered where the twin raised alone.

**Options.**

1. *Pass such a row.* `native.check` passes a row where one side traps and
   the other answers, if a value that the later step validates can
   overflow on one side only. On the twin's side, that value is infinite or
   within its carried bound of DBL_MAX. The contract states this
   exception. The ruling on the matvec parity bound
   ([matvec-parity-bound.md](../closed/matvec-parity-bound.md), item 6)
   has the same rule for an output field that is infinite on one side only.
2. *Keep the rule strict.* The entry serves as it does now. `native.check`
   reports such a row as a `ParityError`. The contract states that its
   claim does not hold on such rows.
3. *Trap on each such row.* The entry traps on each row where a partial sum
   of a step with a bound above 0 can overflow. It then never answers where
   the twin raises. But it traps on some rows where the twin answers, and
   the ruling forbids that. Each projection also computes a second sum for
   each output field, of the sizes of its products.

**Provisional choice.** Option 2. The entry serves as it does now, and
`native.check` reports such a row as a `ParityError`.

**What would close it.** An owner ruling on options 1 to 3. The loop
recommends option 1. The twin itself raises on such a row in one call and
answers in another, so no entry can match it in every call. Option 1 is
the rule that the owner ruled for an output field that is infinite on one
side only.
