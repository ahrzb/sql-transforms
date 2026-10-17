# Where a sum overflows on one side only, must the traps agree?

**Question.** A composition is a `Pipeline`, `ColumnTransformer` or
`FeatureUnion` of catalog entries. For a composition, `to_native` gives one
function that composes the entries of its steps, and this record calls that
function the entry. In a composition, a step that is not bit-exact (its
parity bound is above 0) can come before a step that validates its input.
On rows of values near the largest double, the entry can then trap where
its twin answers, or answer where its twin raises an error. May it?

**What it decides.** It decides what `native.check` requires on such rows.
`native.check` is the function that holds an entry to its twin on test
rows. It raises a `ParityError` where it finds a difference. For example,
the entry is not within its parity bound of the twin, or only one of the
two answers. The question also decides what the catalog contract claims on these rows. The catalog
contract is the spec of what a catalog entry must equal
(`packages/sql-transform/spec/native/catalog-contract.md`).

- The native loop is the loop that grows the native catalog. It numbers its
  tickets in order, and T59 is the ticket of this work. With T59, a step
  that is not bit-exact can feed later steps. An example is
  `make_pipeline(PCA(), StandardScaler())`. The owner ruled this in
  [bounded-steps-in-compositions.md](../closed/bounded-steps-in-compositions.md).
- A later step can validate its input. For example, the validation of
  `StandardScaler` raises an error on an infinite value.
- A projection is a linear map of the features, such as `PCA`. Its output
  field k is the sum over i of the products x_i·c_ki. Here x_i is feature
  i of the row, and c_ki is the coefficient of feature i in component k.
  The twin adds these products in the order of its BLAS kernel. BLAS is the
  library of linear algebra that numpy calls.
- The twin reads rows in calls of `transform`. A call of one row reads the
  row alone. The order of the BLAS kernel can change with the number of
  rows in the call. The entry adds the same products in a balanced tree.
  The catalog module of the projections does this
  (`packages/sql-transform/sql_transform/native/project.py`).
- Let DBL_MAX be the largest double, about 1.8e308, and let n be the number
  of features. The derivation of the parity bound of each family assumes
  that no partial sum overflows (`project.py` states this). A partial sum
  can overflow only if some product x_i·c_ki is at least about DBL_MAX/n.
  This condition is necessary, but not sufficient.
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
  and answer in another. If the entry traps on that row, it breaks
  requirement 2 in the call where the twin answers. If the entry answers,
  it breaks requirement 1 in the call where the twin raises.

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
  test row and 1,023 such rows. The drawn rows have values near 0, far
  below DBL_MAX, and the twin raises on none of them (`band.py` asserts
  this). So each error in these calls comes from the test row.
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
   output field of the step that is not bit-exact, where the later step
   validates v. Let e be the bound of v: its parity bound, or its carried
   bound where an earlier step is not bit-exact either. Then v must be
   infinite, or |v| + e must be at least DBL_MAX. These are the rows where
   v can overflow on one side only. The catalog contract states this
   exception. This option extends item 6 of the ruling on the parity bound
   of the projections
   ([matvec-parity-bound.md](../closed/matvec-parity-bound.md), "matvec"
   for matrix times vector). Item 6 passes an output field that is
   infinite on one side only, if the other side meets three conditions:
   - It is finite.
   - It has the same sign as the infinite side.
   - Its distance to DBL_MAX is at most the parity bound of the field.

   Option 1 applies that rule to a value inside the composition, where the
   value decides a trap.
2. *Keep the rule strict.* The entry serves as it does now. `native.check`
   raises a `ParityError` on such a row. The catalog contract states that
   its claim does not hold on such rows.
3. *Trap on each row where a sum can overflow.* The entry traps on each row
   where a partial sum of a step that is not bit-exact can overflow. It
   then never answers where the twin raises. But it traps on some rows
   where the twin answers, and requirement 2 forbids that. To find these
   rows, each projection also adds the absolute values of the products of
   each output field. Where that second sum is below DBL_MAX, no partial
   sum can overflow.

**Provisional choice.** Option 2. The entry serves as it does now, and
`native.check` raises a `ParityError` on such a row.

**What would close it.** An owner ruling on options 1 to 3. The loop
recommends option 1. The twin itself raises on such a row in one call and
answers in another. So in every call, no entry can trap on exactly the
rows where the twin raises. Option 1 extends item 6 of the matvec ruling
from an output field to a value inside the composition.
