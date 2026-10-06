# The oracle times out where confit traps first

**Question.** In some cases, confit traps on an ordinary error, such as an overflow or a failed cast. DuckDB
evaluates the same query one column at a time. It first builds a huge string (`lpad` or `repeat`, up to about
2 GiB). It then runs past the time budget for one case. The campaign files the case as
`TIMEOUT: timeout:oracle`, which is a gated class.

Should the campaign EXCLUDE this case, as it does for the resource ceilings?

**Evidence.**
- Nightly 2026-10-05, seed 4226438, 10 cases. The case shrinks to
  `SELECT (-9223372036854775808 + c0) FROM __THIS__`. That query is AGREE_TRAP. The timeout needs the
  `lpad(…, greatest(c0, c2), …)` column beside it.
- Campaign 940000–949999, 2 cases (seed 946454, same shape).
- The existing exclusion `resource-ceilings` covers cases where **confit** reaches the ceiling. In those
  cases the campaign does not run DuckDB. In the cases here, confit traps on something else first. So the
  campaign runs DuckDB, and DuckDB does not finish in time.

**Options.**
1. **Exclude with a witness.** The campaign marks the case EXCLUDED if all of these are true:
   - DuckDB times out.
   - Confit traps on both backends.
   - The query contains a string builder (`lpad`, `rpad` or `repeat`) whose length argument is not constant.

   The witness is the same query with that column removed. It must trap on DuckDB as it does on confit.
2. **Raise the oracle budget.** DuckDB does finish after some minutes for each case. But this makes the
   nightly run longer for little information.
3. **Keep it gated.** The class reappears in most nightly runs and campaigns. Each one costs a triage.

**Recommendation.** Choose option 1. The witness limits the exclusion to this mechanism, as the
empty-static rule does.
