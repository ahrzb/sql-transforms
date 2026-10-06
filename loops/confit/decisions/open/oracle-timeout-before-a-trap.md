# The oracle times out where confit traps first

**Question.** Should the campaign exclude a case where confit traps, and the oracle runs past its time
limit before it reaches the same trap?

**What happens.** The SQL functions `lpad` and `repeat` build a string with a length that the query
computes. In these cases, the length is about 2,147,483,647 characters (2 GiB) in one row. Confit and the
oracle evaluate the query in a different order:

- Confit evaluates the whole query for one request, then for the next request. It traps on an ordinary
  error, such as an overflow or a failed cast, in an earlier request. So it never builds the long string.
- The oracle evaluates one expression for all the requests, then the next expression. So it builds the long
  string before it reaches the error.

The oracle then runs past the time limit of the campaign for one case, which is 20 s. The campaign gives
such a case the verdict TIMEOUT, which means that a reading did not finish in time. This verdict is gated.

The campaign already has an exclusion for strings that are longer than the limits of the serving contract.
It applies when both backends of confit trap on such a limit. The campaign then does not run the oracle. In
the cases here, confit traps on a different error first, so that exclusion does not apply.

**Evidence.**
- The nightly of 2026-10-05 had 10 such cases. In one of them (seed 4226438), confit traps on
  `-9223372036854775808 + c0`. Alone, this column gets the verdict AGREE_TRAP: confit and the oracle trap
  with the same error. Beside it, a column computes an `lpad` whose length is 2,147,483,647 in one row, and
  then the oracle runs out of time.
- A campaign over the seeds 940000 to 949999 had 2 cases of the same form (one is seed 946454).
- The seeds of the nightly of 2026-10-06 (4600000 to 4999999), run again on the current code, give 8
  TIMEOUT cases. In seven of them, confit traps first:

| where confit traps | cases (seeds) |
|---|---|
| In a different column or clause than the long string | 4 (4617250, 4658758, 4895703, 4975674) |
| In the same expression as the long string, for example an overflow in the length of an `lpad` | 3 (4657943, 4733641, 4780094) |

  The eighth case (seed 4667128) has no trap, and the oracle agrees with confit. Only the optimizer-on
  reading runs out of time. That case does not need a ruling, so the loop handles it in its work plan.

**Options.**
1. **Exclude with a witness.** The campaign marks the case EXCLUDED if all of these are true:
   - The oracle times out.
   - Confit traps on both backends.
   - The query contains a string builder (`lpad`, `rpad` or `repeat`) whose length is not a constant.

   The witness is the same query with each length above 1000 replaced by 1000. The oracle must trap on the
   witness with confit's error category. This witness covers both rows of the table above.
2. **Raise the time limit of the oracle.** DuckDB finishes each case after some minutes. But this makes the
   nightly run longer for little information.
3. **Keep the verdict gated.** The cases come back in most nightly runs and campaigns, and each one costs a
   triage.

**Recommendation.** Choose option 1. The witness limits the exclusion to this cause, as the empty-static
ruling does.
