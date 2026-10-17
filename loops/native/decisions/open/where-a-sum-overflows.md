# Where a sum overflows on one side only, what must hold?

**Question.** On some rows of values near the largest double, a sum of a
projection overflows in one order of addition and not in another. There an
output field can be infinite or NaN on one side only. In a composition, a
later step can then raise an error on one side only. The twin itself
answers such a row one way in one call and another way in another call.
What must the entry give on such rows?

**What it decides.** It decides what `native.check` requires on such rows.
`native.check` is the function that holds an entry to its twin on test
rows. It raises a `ParityError` where it finds a difference. The question
also decides what the catalog contract claims on such rows. The catalog
contract is the spec of what a catalog entry must equal
(`packages/sql-transform/spec/native/catalog-contract.md`). Two of its
claims now say that they can fail on such rows, and link to this record.

- A projection is a linear map of the features, such as `PCA`. Its output
  field k is a sum with one term for each feature i, such as x_i·c_ki.
  Here x_i is feature i of the row, and c_ki is the coefficient of feature
  i in component k.
- The twin adds these terms in the order of its BLAS kernel. BLAS is the
  library of linear algebra that numpy calls. This order can change with
  the number of rows in a call of `transform`. The entry adds the same
  terms in a balanced tree
  (`packages/sql-transform/sql_transform/native/project.py`).
- confit computes each row on its own. So the entry answers a row in the
  same way in any call.
- Let DBL_MAX be the largest double, about 1.8e308. If a partial sum of a
  sum passes DBL_MAX, the partial sum overflows and becomes infinite. If
  two partial sums overflow with opposite signs, the sum is NaN (not a
  number). An infinity plus an infinity of the other sign is NaN.
- `FactorAnalysis` computes its output with two products of a matrix and
  a vector. A sum of either product can overflow.
- A composition is a `Pipeline`, `ColumnTransformer` or `FeatureUnion` of
  catalog entries. For a composition, `to_native` gives one function that
  composes the entries of its steps. This record calls that function the
  entry too.
- In a composition, a later step can read the output of a projection. The
  owner ruled this in
  [bounded-steps-in-compositions.md](../closed/bounded-steps-in-compositions.md).
  The native loop is the loop that grows the native catalog. It numbers its
  tickets in order, and its ticket T59 serves such compositions. Each
  output field of the later step then has a carried bound
  (`packages/sql-transform/sql_transform/native/_carry.py`).
- In this record, "a row where a value can overflow" is a row where one of
  these is true:
  - The absolute values of the terms of a sum add to about DBL_MAX or
    more.
  - In a composition, |v| + e is about DBL_MAX or more. Here v is the
    twin's value of a later output field, and e is its carried bound.

  On every other row, no value overflows on one side only. The derivation
  of each parity bound assumes this (`project.py` and `_carry.py` state
  it).
- Item 6 of the ruling on the parity bound of the matvec families
  ([matvec-parity-bound.md](../closed/matvec-parity-bound.md)) passes one
  kind of difference on such a row. "Matvec" is short for the product of a
  matrix and a vector. If exactly one side of a field is infinite, the
  field passes under three conditions:
  - The other side is finite.
  - The other side has the same sign as the infinite side.
  - The distance of the other side to DBL_MAX is at most the parity bound
    of the field.

  `native.check` raises a `ParityError` on each other difference of an
  infinite or NaN value.
- A later step can validate its input. For example, the validation of
  `StandardScaler` raises an error on an infinite value. So on a row where
  a value can overflow, the twin can raise an error where the entry
  answers. The entry can also trap where the twin answers.
- The ruling on where the twin raises
  ([tolerated-differences.md](../closed/tolerated-differences.md)) has two
  requirements:
  1. Where the validation of the twin raises an error, the entry traps.
  2. Where the twin answers, the entry does not trap.

**Evidence.** The scripts and the data are in the project's shared folder,
at `/mnt/project-files/transforms-loop/t59/`. Its `README.md` names each
script.

- `fields.py` fitted 80 projections, each on its own random rows: 24
  `PCA`, 16 `TruncatedSVD`, 13 `GaussianRandomProjection`, 11 `FastICA`
  and 16 `FactorAnalysis`. Each had 2 to 512 features and 1 to 16
  components. Each read 400 test rows, 32,000 test rows in all. Each value
  of a test row was 5% to 100% of DBL_MAX, with a random sign. A value can
  overflow on 19,487 of these test rows.
- `fields.py` held each output field as `native.check` holds it, with item
  6. It compared the entry with two answers of the twin: the answer to the
  test row alone, and the answer to the test row in a call of 64 rows. The
  other 63 rows of the call were drawn like the rows of the fit.
  `fields.py` also compared the two answers of the twin. The table gives
  the output fields past the bound, of 180,400 in each comparison:

  | kind of the two values | entry and twin alone | entry and twin in the call | twin alone and twin in the call |
  |---|---|---|---|
  | NaN and an infinity | 4,630 | 3,938 | 4,515 |
  | NaN and a finite value | 369 | 403 | 261 |
  | an infinity and a finite value | 3,471 | 3,601 | 2,457 |
  | two infinities of opposite signs | 67 | 78 | 56 |
  | two finite values | 0 | 0 | 0 |
  | all | 8,537 | 8,020 | 7,289 |

- Each of these output fields was on a row where a value can overflow.
  They were in 58 of the 80 projections. The two answers of the twin
  differed so in 45 of them.
- `band.py` made 80 `Pipeline`s of a projection and a `StandardScaler`,
  and fitted each on its own random rows. The projection was a `PCA` (42
  `Pipeline`s) or a `GaussianRandomProjection` (38), with 2 to 512
  features and 1 to 16 components. Each `Pipeline` read 400 test rows, as
  `fields.py` draws them. A sum of the projection can overflow on 22,268
  of the 32,000 test rows.
- The twin raised an error on 15,226 test rows when it read each test row
  alone. It raised on 14,326 test rows in calls of 64 rows, and on 16,285
  test rows in calls of 1,024 rows. The other rows of each call were drawn
  like the rows of the fit. The twin raises on none of them alone
  (`band.py` asserts this).
- On 2,224 test rows, the twin raised in one of two calls and answered in
  the other: the call of the test row alone, and the call of 64 rows. 54
  of the 80 `Pipeline`s had such test rows.
- The entry trapped on 13,623 test rows. On 713 of them, the twin answered
  when it read the test row alone. On 2,316 other test rows, the entry
  answered and the twin raised alone. Each of these rows was a row where a
  sum of the projection can overflow.
- The parity run of T59 (`measure.py`) served 264 random compositions,
  each with a step that is not bit-exact. Each composition read rows of
  nine kinds, and one kind has values near DBL_MAX. On the eight other
  kinds, no output field of 8.9 million was past its bound. No row of
  these kinds trapped or raised an error on one side only. On the kind
  near DBL_MAX, 447 output fields of 174,778 were past their bound, against
  the twin of each row alone. On 231 rows of that kind, one side trapped or
  raised an error alone.

**Options.**

1. *Pass the differences of an overflow.* On a row where a value can
   overflow, `native.check` passes two kinds of differences:
   - an output field that is infinite or NaN on one side or on both sides;
   - a trap or an error on one side only.

   On such a row, `native.check` still holds a field that is finite on
   both sides to its parity bound. On every other row, the rule does not
   change. This option replaces item 6 of the matvec ruling. Item 6 passes
   only one of these differences.
2. *Keep the rule strict.* The entry serves as it does now. `native.check`
   raises a `ParityError` on such a row, unless item 6 passes the field.
   The catalog contract states that its claims can fail on such rows.
3. *Trap on each such row.* The entry traps on each row where a value can
   overflow. To find these rows, it adds the absolute values of the terms
   of each sum, and it computes each carried bound. Then the entry never
   answers where the twin raises. But it traps where the twin answers, and
   requirement 2 forbids that. In `fields.py`, the twin of each projection
   alone answered each test row. In `band.py`, the twin answered 7,042 of
   the 22,268 test rows where a sum of the projection can overflow, when
   it read each test row alone.

**Provisional choice.** Option 2. The entry serves as it does now, and
`native.check` raises a `ParityError` on such a row, unless item 6 passes
the field.

**What would close it.** An owner ruling on options 1 to 3. The loop
recommends option 1. On such a row, the twin itself gives different
answers in two calls. So no entry can give the twin's answer in every
call. Option 1 holds the entry to its twin on every other row, and on each
field that is finite on both sides.
