# The oracle times out where confit traps first

**Question.** In some cases confit traps on an ordinary error (an overflow, a
failed cast) while DuckDB, evaluating the same query column at a time, first
builds a huge string (`lpad`/`repeat` to about 2 GiB) and runs past the
per-case budget. The case is filed as `TIMEOUT: timeout:oracle`, a gated class.
Should it be EXCLUDED, like the resource ceilings?

**Evidence.**
- Nightly 2026-10-05, seed 4226438, 10 cases. It shrinks to
  `SELECT (-9223372036854775808 + c0) FROM __THIS__`, which AGREE_TRAPs: the
  timeout needs the `lpad(…, greatest(c0, c2), …)` column beside it.
- Campaign 940000–949999, 2 cases (seed 946454, same shape).
- The existing exclusion `resource-ceilings` covers cases where **confit**
  hits the ceiling, so DuckDB is not run. Here confit traps on something
  else first, so DuckDB is run and does not finish in time.

**Options.**
1. **Exclude with a witness.** EXCLUDED when DuckDB times out, confit traps on
   both backends, and the query contains a string builder (`lpad`, `rpad`,
   `repeat`) whose length argument is not constant. The witness is the same
   query with that column removed, which must trap on DuckDB like confit.
2. **Raise the oracle budget.** DuckDB does finish eventually (minutes per
   case), but this lengthens the nightly for little information.
3. **Keep it gated.** The class reappears in most nightlies and campaigns,
   and each one costs a triage.

**Recommendation.** Option 1: the witness keeps it to this mechanism, as the
empty-static rule does.
