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

The campaign already has an exclusion for strings that are longer than the limits of the serving contract
(1 GiB for a string that a function builds). It applies when both backends of confit trap on such a limit.
The campaign then does not run the oracle. In the cases here, confit traps on a different error first, so
that exclusion does not apply.

**Evidence.**
- The nightly of 2026-10-05 had 10 such cases. In one of them (seed 4226438), confit traps on
  `-9223372036854775808 + c0`. Alone, this column gets the verdict AGREE_TRAP: confit and the oracle trap
  with the same error. Beside it, a column computes an `lpad` whose length is 2,147,483,647 in one row, and
  then the oracle runs out of time.
- A campaign over the seeds 940000 to 949999 had 2 cases of the same form (one is seed 946454).
- The seeds of the nightly of 2026-10-06 (4600000 to 4999999), run again on master f72214b, give 8
  TIMEOUT cases. In seven of them, confit traps first:

| where confit traps | cases (seeds) |
|---|---|
| In a different column or clause than the long string | 4 (4617250, 4658758, 4895703, 4975674) |
| In the same expression as the long string, for example an overflow in the length of an `lpad` | 3 (4657943, 4733641, 4780094) |

  The eighth case (seed 4667128) has no trap, and the oracle agrees with confit. Only the optimizer-on
  reading runs out of time. That case does not need a ruling. Pull request #442 fixed it (see the last item
  of this list).
- The eight nightlies from 2026-10-07 to 2026-10-14 (seeds 5000000 to 8199999, all on master b4b8d03, before
  #442) had 43 TIMEOUT cases in a run of DuckDB. The loop has the seeds of 26 of them. In 24 of the 26,
  confit traps on both backends, and the query has `lpad` or `repeat`: 5302968, 5416440, 5517213, 5652903,
  5710437, 5827768, 5907077, 6242314, 6385245, 6434710, 6654683, 6710997, 6854666, 6994881, 7015472,
  7157597, 7257342, 7345367, 7429356, 7570725, 7830621, 7964417, 8021654 and 8151365. Run again on master
  b4b8d03, 21 of the 24 run out of time again. The other 3 (6654683, 6710997, 8021654) finish in time with
  the verdict AGREE_TRAP. So the time that the oracle needs is near the limit. In the last 2 cases (5068704,
  7731793), confit returns no rows, and only the optimizer-on reading runs out of time, as in seed 4667128.
  Pull request #442 fixed these 2 cases too (see the last item of this list).
- A campaign over the seeds 9150000 to 9209999 ran on master 65f401b (2026-10-14, before #442). It had 2
  more cases in which confit traps and the oracle runs out of time: 9195711 and 9200287.
- Pull request #442 (2026-10-14) gives the optimizer-on reading a time limit of its own, which is 5 s. The
  optimizer-on reading runs after the oracle. If it runs out of time, the case keeps the verdict of the
  oracle. Before #442, the two readings shared the time limit of the case. With #442, the 3 cases without a
  trap (4667128, 5068704, 7731793) get the verdict AGREE.

  The loop ran again each case above in which confit traps, 35 cases in all. They are 4226438, 946454, the 7
  cases of the nightly of 2026-10-06, the 24 cases of the eight nightlies, 9195711 and 9200287. It ran them
  one case at a time, with the limit of the nightly (20 s):

| commit | AGREE_TRAP | TIMEOUT |
|---|---|---|
| master 65f401b, before #442 | 1 | 34 |
| master f938de5, with #442 | 16 | 19 |

  With #442, 16 cases get the verdict AGREE_TRAP. In each of them, the oracle reaches the trap in time, and
  only the optimizer-on reading runs out of time. In the other 19 cases, the oracle runs out of time. When
  the loop runs four of the 35 cases at a time on the same machine, 34 of them run out of time with #442. So
  the verdict of the 16 cases depends on the load of the machine.
- The nightly of 2026-10-15 (seeds 8200000 to 8599999) ran on master df653be, which has #442. It had one case
  of this kind, seed 8249704, with the verdict TIMEOUT. In the first request, `c2` is -2,147,483,648, and
  confit traps when one field of the output struct computes `- c2`. In the fifth request, another field calls
  `lpad` with a length of 2,147,483,647. The loop ran the case again alone, without a time limit. The case took
  about 55 s, and it got the verdict AGREE_TRAP.

**Options.**
1. **Exclude with a witness.** The campaign gives the case the verdict EXCLUDED if all of these are true:
   - The oracle times out.
   - Confit traps on both backends.
   - The query contains a string builder (`lpad`, `rpad` or `repeat`) whose length is not a constant.

   The witness is the same query with each length above 1000 replaced by 1000. The oracle must trap on the
   witness with the same kind of error as confit, such as an overflow. This witness covers both places
   where confit traps (the first table above).
2. **Raise the time limit of the oracle.** The oracle finishes each case after some minutes. But this makes
   the nightly run longer for little information.
3. **Keep the verdict gated.** The cases come back in most nightly runs and campaigns, and each one costs a
   triage.

**Recommendation.** Choose option 1. The witness limits the exclusion to this cause, as the ruling on the
[trap timing of a join on an empty static table](../closed/empty-static-join-trap-timing.md) does.
