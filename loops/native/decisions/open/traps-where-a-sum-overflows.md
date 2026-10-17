# Where a sum overflows on one side only, must the traps agree?

**Question.** A composition is a `Pipeline`, `ColumnTransformer` or
`FeatureUnion` of catalog entries. In a composition, a step that is not
bit-exact (its parity bound is above 0) can come before a step that
validates its input. On rows of values near the largest double, the entry
and its twin can then trap on different rows. May they?

**What it decides.** It decides what `native.check` requires on rows of
values near the largest double. `native.check` is the function that holds
a catalog entry to its twin on test rows. Where the two differ, it raises
a `ParityError`. The question also decides what the catalog contract
claims on these rows. The catalog contract is the spec of what a catalog
entry must equal (`packages/sql-transform/spec/native/catalog-contract.md`).

- The native loop numbers its tickets in order, and T59 is the ticket of
  this work. Since T59, a step that is not bit-exact can feed later steps.
  An example is `make_pipeline(PCA(), StandardScaler())`. The owner ruled
  this in [bounded-steps-in-compositions.md](../closed/bounded-steps-in-compositions.md).
- A later step can validate its input. For example, the validation of
  `StandardScaler` raises an error on an infinite value.
- A projection such as `PCA` computes each output field k as a sum of
  products x_i·c_ki. Here x_i is feature i of the row, and c_ki is its
  coefficient in component k. The twin adds these products in the order of
  its BLAS kernel. BLAS is the library of linear algebra that numpy calls.
- The twin reads rows in calls of `transform`. A call of one row reads the
  row alone. The order of the BLAS kernel can change with the number of
  rows in the call. The entry adds the same products in a balanced tree.
  The catalog module of the projections does this
  (`packages/sql-transform/sql_transform/native/project.py`).
- Let DBL_MAX be the largest double, about 1.8e308, and let n be the number
  of features. The parity bound of each family assumes that no partial sum
  overflows. A partial sum can overflow only on a row where some product
  x_i·c_ki is about DBL_MAX/n or more.
- On such a row, a sum can overflow to infinity in one order and stay
  finite in the other. Then the validation of the later step raises an
  error on one side only.
- The ruling on where the twin raises
  ([tolerated-differences.md](../closed/tolerated-differences.md)) has two
  requirements:
  1. Where the validation of the twin raises an error, the entry traps.
  2. Where the twin answers, the entry does not trap.
- confit computes each row on its own, so the entry answers a row in the
  same way in any call. But on such a row, the twin can raise in one call
  and answer in another. Then the entry
  breaks one of the two requirements in one of the two calls.

**Evidence.** The scripts and the data are in the project's shared folder,
at `/mnt/project-files/transforms-loop/t59/`.

- `band.py` made 80 `Pipeline`s of a projection and a `StandardScaler`,
  and fitted each on its own random rows. The projection was a `PCA` (42
  `Pipeline`s) or a `GaussianRandomProjection` (38), with 2 to 512
  features and 1 to 16 components. Each `Pipeline` read 400 test rows,
  32,000 test rows in all. Each value of a test row was 5% to 100% of
  DBL_MAX, with a random sign.
- The twin raised on 16,198 test rows when it read each test row alone. It
  raised on 15,385 test rows when it read each in its own call of 64 rows:
  the test row and 63 rows drawn like the rows that the `Pipeline` was
  fitted on. It raised on 17,103 test rows in calls of 1,024 rows: the
  test row and 1,023 such rows. The drawn rows are small, and the twin
  raises on none of them. So each error in these calls comes from the test
  row.
- So the twin does not agree with itself on these rows. On 2,137 test rows,
  the twin raised in one of two calls and answered in the other: the call
  of the test row alone, and the call of 64 rows. 56 of the 80 `Pipeline`s
  had such test rows.
- The entry trapped on 15,007 test rows. On 721 of them, the twin answered
  when it read the test row alone. On 1,912 other test rows, the entry
  answered and the twin raised alone.

**Options.**

1. *Pass such a row.* `native.check` passes a row where one side traps and
   the other answers, under one condition. Let v be the twin's value of an
   output field of the bounded step that the later step validates, and let
   e be the carried bound of v. Then v must be infinite, or |v| + e must be
   at least DBL_MAX. Only then can v overflow on one side only. The contract
   states this exception. This option extends item 6 of the ruling on the
   parity bound of the projections
   ([matvec-parity-bound.md](../closed/matvec-parity-bound.md), "matvec"
   for matrix times vector). Item 6 passes an output field that is infinite
   on one side, if the other side is finite, of the same sign and within
   the parity bound of DBL_MAX. Option 1 applies that rule to a value
   inside the composition, where the value decides a trap.
2. *Keep the rule strict.* The entry serves as it does now. `native.check`
   raises a `ParityError` on such a row. The contract states that its claim
   does not hold on such rows.
3. *Trap on each row where a sum can overflow.* The entry traps on each row
   where a partial sum of a step that is not bit-exact can overflow. It
   then never answers where the twin raises. But it traps on some rows
   where the twin answers, and the ruling forbids that. To find these rows,
   each projection also adds the absolute values of the products of each
   output field. This is a second sum for each output field.

**Provisional choice.** Option 2. The entry serves as it does now, and
`native.check` raises a `ParityError` on such a row.

**What would close it.** An owner ruling on options 1 to 3. The loop
recommends option 1. The twin itself raises on such a row in one call and
answers in another. So no entry can trap on exactly the rows where the twin
raises, in every call. Option 1 extends item 6 of the matvec ruling from an
output field to a value inside the composition.
