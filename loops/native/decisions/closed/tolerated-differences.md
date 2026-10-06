# Where a native entry may differ from its twin

**Question.** Where exact parity needs sklearn's own behaviour, what may
the entry do meanwhile?

1. **The twin raises on input sklearn's validation rejects** (most
   estimators refuse an infinite feature). Provisional: the entry may answer
   there; never the reverse.
   *Ground:* the serving query already trapped for the twin, so no served
   answer changes; making the entry trap needs a per-estimator copy of
   sklearn's validation, raised with `error()`.

Two further edges were tolerated until confit could serve them, and are
exact now: an unknown instance id raises (confit's `error()`), and a NULL
id with a struct return is a NULL struct (`SqlFunction(null_when=...)`).

**What would close it.** An owner ruling that this is acceptable as stated,
or that it must block the entry instead.

**Methodology (2026-10-06).** The research note is
[research/2026-10-06/tolerated.md](../research/2026-10-06/tolerated.md). An
adversarial verifier re-ran it with a different design of the probe
(item 1). The verifier is a second agent that tried to refute each claim of
the note.

The research used this setup and these terms:

- **Environment.** The software is scikit-learn (sklearn) 1.9.0, DuckDB
  1.5.5 and a release build of confit. The code is master at commit 113fba7.
- **The provisional rule.** This is the ruling above: the entry may answer
  where the twin raises, and never the reverse.
- **check.** `native.check` is the test that serves the same query with the
  twin and with the entry and compares the results. This record calls it
  `check`.
- **EDGES.** `EDGES` (in `catalog_test.py`) is the list of extreme input
  values that the catalog tests serve to the entries in the gate.
- **Values.** The value inf means infinity. NaN is the floating-point value
  "not a number".
- **Field.** A field is one output field of the struct that an entry
  returns. Field 0 is the first output field.

1. **The probe.**
   - The probe is the experiment of the note. It covers all 160 fixture
     configurations of the catalog tests.
   - It set each feature in turn to +inf, −inf, NaN, NULL, ±1e308, 0, −1,
     ±1e6 and 0.5. That gave 5,104 rows in all.
   - It served each row alone through confit, once with the twin and once
     with the entry. It used the same query that `check` serves.
   - The verifier used its own design, with 4 features and other data. That
     design gave 6,688 rows and the same pattern of results.
2. **What the entry answers where the twin raises.**
   - The twin raised an error on 1,396 rows. The entry trapped on none of
     them.
   - On 565 of those rows, every field of the entry was finite:
     - On 425 rows, a field that reads the bad input is finite.
     - On 140 rows, no field reads the bad input. An example is a feature
       selector that drops the column.
   - The twin answered on 3,708 rows. On those rows, the entry differed from
     the twin on 0 rows and trapped on 0 rows.
3. **NULL, the common case in SQL.**
   - In 48 of the 160 configurations, a NULL input comes back with every
     field finite.
   - These are examples:
     - KBinsDiscretizer puts the NULL in the top bin.
     - Binarizer answers 1.0.
     - IsotonicRegression with `out_of_bounds="clip"` answers its top value.
     - SplineTransformer with `handle_missing="error"` answers 0.0 in its
       fields.
   - In a 1,000-row call with one NULL, the twin's query fails. The entry
     returns all 1,000 rows.
4. **What those finite answers are.** Of the 565 rows:
   - 525 rows equal sklearn's own arithmetic with validation off (the
     sklearn setting `assume_finite=True`).
   - 36 rows are artefacts of the NaN order of DuckDB. DuckDB puts NaN above
     every number.
   - On 4 rows, sklearn raises an error even with its validation off.
5. **What the gate sees.**
   - `check` skips every row where the twin raises. That is 624 of the
     12,800 rows that the gate serves (4.9%).
   - `EDGES` contains no ±inf.
   - The probes found a real parity breach where the twin answers. The
     breach is at ±inf, in a SplineTransformer with periodic extrapolation,
     `handle_missing="zeros"` and n_knots ≥ degree + 3. Here n_knots is the
     number of knots.
   - At those inputs, the twin answers NaN. The entry answers 0.0.
   - To show the breach, the gate needs both ±inf in `EDGES` and a fixture
     with `handle_missing="zeros"`.
6. **Input guards** (measured).
   - An input guard built from the tags that sklearn declares for each
     transformer is not exact. It misses 156 rows where the twin raises.
   - That guard also traps 22–112 times where the twin answers. The record
     forbids a trap where the twin answers.
   - A guard for each leaf is possible. A leaf is a transformer that holds
     no other transformer. These are the steps to build the guard:
     1. When the entry translates the fitted transformer, probe which of
        ±inf and NaN each column of the leaf rejects.
     2. Apply each test to the input expressions of the leaf itself.
     3. Join the tests with OR into field 0, as `CASE … THEN error()`.
        `error()` is the SQL function that raises an error.
   - That trap fires in confit and in DuckDB when the query reads any single
     field.
   - In the cheapest SQL spelling, the guard costs 0–31 ns per feature per
     row. Other spellings cost up to 108 ns. The twin costs 100–220 µs per
     row.
   - The research did not build a prototype of the guard across the catalog.
   - One case where the twin raises is not validation at all. A
     SplineTransformer of degree 0 with constant extrapolation raises a
     broadcast error. This error is an sklearn bug.
7. **A middle option** is the condition "non-finite in, non-finite out". It
   costs nothing for an entry whose arithmetic already propagates non-finite
   values. The verifier refuted the claim that it costs as much as the
   guard. But the condition still turns errors into answers. Also, the NaN
   order of DuckDB turns a NaN back into a branch downstream. For example,
   Binarizer on NaN answers 1.

**Recommendation.** Do not accept the provisional rule as stated. Do not
accept it with only the non-finite condition either. Require the entry to
trap where the validation of the twin raises an error.

- **What the guard covers.** The guard covers the non-finite checks for each
  column. It also covers these domain checks:
  - PowerTransformer with Box-Cox, on x ≤ 0
  - unknown categories
  - IsotonicRegression with `out_of_bounds="raise"`
  - SplineTransformer with `extrapolation="error"`
  - MissingIndicator with `error_on_new`
- **Build it** as the probed guard for each leaf above, joined with OR into
  field 0.
- **Keep the provisional rule only where the twin raises for a reason other
  than validation,** such as an sklearn bug. Name each such case in this
  record.
- **Before this ruling replaces the provisional one,** these must be true:
  - `check` asserts on every row that the entry traps if and only if the
    twin raises. It skips no row.
  - The gate has ±inf in `EDGES`. It also has a fixture of a
    SplineTransformer with periodic extrapolation and
    `handle_missing="zeros"`.
  - A prototype shows across the catalog that the guard never traps where
    the twin answers.

These are the reasons, judged against the goal that inference does not
change:

- **The twin's error is part of what inference returns.** A query that
  fails with the twin succeeds with the entry. It returns plausible numbers.
  For example, a NULL lands in the top bin of KBinsDiscretizer. Nothing
  downstream can tell the difference.
- **The gate cannot see it.** The gate skips these rows.
- **The guard is cheap.** It costs under 2% of the cost of the twin.
- **The obstacle that the record states is smaller than it reads.** The
  record names a copy of sklearn's validation for each transformer. A probe
  of each leaf replaces most of that copy.

This recommendation reverses the provisional rule in favour of exactness.
