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

**Methodology (2026-10-06).** The note is
[research/2026-10-06/tolerated.md](../research/2026-10-06/tolerated.md),
re-run by an adversarial verifier with a different probe design. The
environment is scikit-learn 1.9.0, DuckDB 1.5.5 and a release build of
confit; master 113fba7.

1. **The probe.**
   - It covers all 160 catalog fixture configurations.
   - Each feature in turn was set to +inf, −inf, NaN, NULL, ±1e308, 0, −1,
     ±1e6 and 0.5: 5,104 rows in all.
   - Each row was served one at a time through confit, once with the twin and
     once with the entry, using the query `check` serves.
   - The verifier's own design (4 features, other data) gave 6,688 rows and
     the same picture.
2. **What the entry answers where the twin raises.**
   - The twin raised on 1,396 rows. The entry raised on none of them.
   - 565 of those rows came back entirely finite:
     - 425 where a lane that reads the bad input is finite;
     - 140 where no lane reads it, for example a selector that drops the
       column.
   - Where the twin answers (3,708 rows), 0 differ and 0 go the other way.
3. **NULL, the common case in SQL.**
   - A NULL comes back entirely finite in 48 of the 160 configurations.
   - Examples: KBins puts it in the top bin, Binarizer answers 1.0,
     Isotonic(`clip`) answers its top value, and a strict spline answers 0.0
     lanes.
   - In a 1,000-row call with one NULL, the twin's query fails. The entry
     returns all 1,000 rows.
4. **What those finite answers are.** Of the 565:
   - 525 equal sklearn's own arithmetic with validation off
     (`assume_finite=True`);
   - 36 are artefacts of DuckDB's NaN ordering, which puts NaN above every
     number;
   - 4 still raise in sklearn even then.
5. **What the gate sees.**
   - `check` skips every row where the twin raises: 624 of 12,800 gate rows
     (4.9%).
   - EDGES contains no ±inf.
   - Probing found a real breach where the twin answers: a periodic
     SplineTransformer with `handle_missing="zeros"` and
     n_knots ≥ degree + 3, at ±inf. The twin answers NaN; the entry answers
     0.0.
   - Showing it needs both ±inf in EDGES and a `"zeros"` fixture.
6. **Guards** (measured).
   - A guard read off sklearn's tags is not exact. It misses 156 twin raises
     and over-raises 22–112 times, and the record forbids raising where the
     twin answers.
   - A guard per leaf can be constructed:
     - at translation, probe which of ±inf and NaN each column rejects;
     - apply the test to the leaf's own input expressions;
     - OR the tests into lane 0 as `CASE … THEN error()`.
   - That trap fires on any single field read, in confit and in DuckDB.
   - It costs 0–31 ns per feature per row in the cheapest spelling, up to
     108 ns, against the twin's 100–220 µs per row.
   - It has not been prototyped across the catalog. One twin raise is not
     validation at all: the degree-0 constant spline's broadcast error, an
     sklearn bug.
7. **A middle ground,** "non-finite in, non-finite out", costs nothing for entries whose arithmetic already propagates
   non-finite values. The verifier refuted the claim that it costs as much
   as the guard. But it still turns errors into answers, and DuckDB's NaN
   ordering turns a NaN back into a branch downstream: Binarizer on NaN
   answers 1.

**Recommendation.** Do not accept the tolerance as stated, and do not accept
it with only the non-finite condition. Require the entry to raise where the
twin's validation raises.

- **What the guard covers:** the per-column non-finite checks and the domain
  checks: Box-Cox x ≤ 0, unknown categories, Isotonic `raise`, Spline
  `error`, and MissingIndicator `error_on_new`.
- **Build it** as the per-leaf probed guard above, OR-ed into lane 0.
- **Keep the tolerance only for twin raises that are not validation,** such
  as sklearn bugs. Name each one in this record.
- **Before this ruling replaces the provisional one:**
  - `check` asserts "raises iff the twin raises" on every row, and skips
    none;
  - the gate gains ±inf in EDGES and a `handle_missing="zeros"` periodic
    spline fixture;
  - a prototype shows zero over-raises across the catalog.

Why, judged against the goal that inference does not change:

- **The twin's error is part of what inference returns.** A query that fails
  with the twin succeeds with the entry and returns plausible numbers: a
  NULL lands in the top bin. Nothing downstream can tell.
- **The gate cannot see it.** It skips these rows.
- **The guard is cheap:** under 2% of the twin's cost.
- **The record's stated obstacle is smaller than it reads.** It names a
  per-estimator copy of sklearn's validation. Probing each leaf replaces
  most of that copy.

This reverses the provisional reading in favour of exactness.
