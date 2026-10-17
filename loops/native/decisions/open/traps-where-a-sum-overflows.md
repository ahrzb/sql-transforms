# Where a sum overflows on one side only, must the traps agree?

**Question.** A composition is a `Pipeline`, `ColumnTransformer` or
`FeatureUnion` of catalog entries. In a composition, a step that is not
bit-exact (its parity bound is above 0) can come before a step that
validates its input. On rows of values near the largest double, the entry
and its twin can then trap on different rows. May they?

**What it decides.** It decides what `native.check` holds on such rows.
`native.check` is the function that holds a catalog entry to its twin on
test rows. It also decides what the catalog contract claims there. The
catalog contract is the spec of what a catalog entry must equal
(`packages/sql-transform/spec/native/catalog-contract.md`).

- Since T59, a ticket of the native loop, a step that is not bit-exact can
  feed later steps, for example in `make_pipeline(PCA(), StandardScaler())`.
  The owner ruled this in
  [bounded-steps-in-compositions.md](../closed/bounded-steps-in-compositions.md).
- A later step can validate its input. For example, the validation of
  `StandardScaler` raises an error on an infinite value.
- A projection such as `PCA` computes each output field k as a sum of
  products x_i·c_ki. Here x_i is feature i of the row, and c_ki is its
  coefficient in component k. The twin adds these products in the order of
  its BLAS kernel. BLAS is the library of linear algebra that numpy calls.
  That order can change with the number of rows in the call. The entry adds
  the same products in a balanced tree (`project.py`).
- The parity bound of each family assumes that no partial sum overflows
  (`project.py`). A partial sum can overflow only on a row where some
  product x_i·c_ki is about DBL_MAX/n or more. DBL_MAX is the largest
  double, about 1.8e308, and n is the number of features.
- On such a row, a sum can overflow to infinity in one order and stay
  finite in the other. Then the validation of the later step raises an
  error on one side only.
- The ruling on where the twin raises
  ([tolerated-differences.md](../closed/tolerated-differences.md)) has two
  requirements. The entry traps where the validation of its twin raises.
  The entry never traps where its twin answers. The result of the twin on
  such a row changes with the number of rows in the call. So no entry can
  meet both requirements in every call.

**Evidence.** The scripts and the data are in
`/mnt/project-files/transforms-loop/t59/`.

- The twin does not agree with itself on such rows. `band.py` made 80
  `Pipeline`s of a projection and a `StandardScaler`, and fitted each on
  its own random data. The projection was a `PCA` (42 `Pipeline`s) or a
  `GaussianRandomProjection` (38), with 2 to 512 features and 1 to 16
  components. Each `Pipeline` served 400 rows, 32,000 rows in all. Each
  value of a row was 5% to 100% of DBL_MAX, with a random sign.
- The twin raised on 16,198 rows when it read each row alone. It raised on
  15,385 rows in a call of 64 rows: the row and 63 rows drawn like the
  training data. It raised on 17,103 rows in a call of 1,024 rows: the row
  and 1,023 such rows. The twin raises on none of the drawn rows.
- On 2,137 rows, the twin gave a different result alone and in the call of
  64 rows. It raised in one of the two calls and answered in the other. 56
  of the 80 `Pipeline`s had such rows.
- The entry answers a row in the same way in any call. It trapped on
  15,007 rows. On 721 of them, the twin answered when it read the row
  alone. On 1,912 other rows, the entry answered and the twin raised alone.

**Options.**

1. *Pass such a row.* `native.check` passes a row where one side traps and
   the other answers, under one condition. On the twin's side, a value that
   the later step validates must be infinite, or within its carried bound
   of DBL_MAX. Only such a value can overflow on one side only. The
   contract states this exception. Item 6 of the ruling on the parity bound
   of the projections
   ([matvec-parity-bound.md](../closed/matvec-parity-bound.md)) has the
   same rule for an output field. It passes a field that is infinite on one
   side and within its bound of DBL_MAX on the other. Option 1 applies that
   rule to a value inside the composition, where the value decides a trap.
2. *Keep the rule strict.* The entry serves as it does now. `native.check`
   reports such a row as a `ParityError`. The contract states that its
   claim does not hold on such rows.
3. *Trap on each such row.* The entry traps on each row where a partial sum
   of a step that is not bit-exact can overflow. It then never answers
   where the twin raises. But it traps on some rows where the twin answers,
   and the ruling forbids that. To find such rows, each projection also
   adds the absolute values of the products of each output field. This is
   a second sum for each output field.

**Provisional choice.** Option 2. The entry serves as it does now, and
`native.check` reports such a row as a `ParityError`.

**What would close it.** An owner ruling on options 1 to 3. The loop
recommends option 1. The twin itself raises on such a row in one call and
answers in another. So no entry can trap on exactly the rows where the twin
raises, in every call. Option 1 is the rule that the owner ruled for an
output field that is infinite on one side only.
