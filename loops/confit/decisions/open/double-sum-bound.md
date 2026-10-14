# The open parts of the float-reduction bound

**Question.** The owner ruled an order of five kinds of queries for confit to serve next
([next query classes](../closed/next-query-classes.md)). The fifth kind is an aggregate over the static rows
that one request matches. For `sum` and `avg` over DOUBLE values, the answer depends on the order of the
additions. The comparison contract says when an answer of confit counts as equal to the oracle's answer
([`packages/confit/docs/oracle/05-the-comparison-contract.md`](../../../../packages/confit/docs/oracle/05-the-comparison-contract.md#floating-point-comparisons)).
Its claim `float-reduction-bound` covers this case. The claim accepts a DOUBLE `sum` or `avg` of confit
if its distance from the oracle's answer is at most a bound B. But the claim leaves some parts open, and
it says that these parts need a definition before use. No code checks the bound yet. Do you approve the
definitions below?

**The bound now.** A group is the set of static rows that match one key value. For the n values v_i of a
group that are not NULL, and eps = 2^-52 (the distance from 1.0 to the next DOUBLE):

$$
\left|\mathrm{confit}-\mathrm{oracle}\right| \le B = 2(n-1)\,\mathrm{eps}\sum_i |v_i|.
$$

For `avg`, the claim divides B by n. The claim leaves these parts open:

- the algorithm of the sum, and its number of additions;
- values and results that are not finite (NaN and infinity);
- overflow of the sum, or of the sum of the absolute values;
- underflow;
- n = 0 and n = 1;
- the final division of `avg`.

**What the oracle does** (DuckDB 1.5.5, optimizer off, measured 2026-10-14). Each query is
`SELECT (SELECT sum(s.v) FROM s WHERE s.k = t.k) FROM t`, and `avg` in the same form. The oracle uses
its default number of threads, 4 on the machine of the measurement. Table order is the order of the rows in the Arrow
table that the caller gives for a static table. The table-order sum adds the values of a group from the
first row of the table to the last.

- **The order changes the answer.** The measurement makes six static tables. Each has three rows with
  the key 1 and the values 1e16, 1 and -1e16, in a different order. The oracle's sum is 0 for four of
  the tables and 1 for the other two. For each table, the oracle's sum equals the table-order sum.
- **Small tables.** A static table has 5,000 rows in 50 groups. In each of 3 runs, the oracle's `sum` and
  `avg` of each of the 50 groups equal the table-order `sum` and `avg` bit for bit.
- **Large tables.** A static table has 300,000 rows in 300 groups. In each of 3 runs, the table-order sum
  equals the oracle's sum in the same 11 of the 300 groups. Another static table has 600,000 rows in 64
  groups. In each of 3 runs, the table-order sum equals the oracle's sum in 2 or 3 of the 64 groups. For
  this table, the oracle's own sum for a group changes from one run to the next.
- **The bound holds.** The 9 runs above are 3 runs for each of the 3 tables. In each of them, the
  distance between the oracle's answer and the table-order answer is in the bound, for `sum` and for
  `avg`. For `avg`, this uses the bound of the definitions below. The largest distance is 0.0012 of the
  bound.
- **The start of the sum.** The oracle's sum of one value -0.0 is +0.0. A sum that starts at its first
  value would give -0.0. So the oracle starts the sum at +0.0.

So a table-order sum is bit-exact with the oracle only for small tables. For large tables, the oracle
itself can give different sums in different runs. The bound accepts each sum that the measurement saw.

**Proposed definitions.**

| open part | definition |
|---|---|
| algorithm and number of additions | Confit gives the table-order sum. It starts at +0.0 and adds each matched value that is not NULL with one IEEE addition (round to nearest). n is the number of these values. The first addition, +0.0 + v_1, is exact, so the sum rounds at most n − 1 times. `avg` divides the sum by n with one IEEE division. |
| n = 0 | Confit and the oracle both answer NULL. The comparison is exact. |
| n = 1 and n = 2 | The comparison is exact. One value needs no rounding. For two values, the order cannot change the sum, because an IEEE addition gives the same result in both orders of its operands. |
| values that are not finite, and overflow | If a group has a NaN or an infinite value, confit refuses the build by name. Confit also refuses it if the sum of the absolute values of a group is not finite. Confit knows the static values when it builds the function. |
| underflow | An IEEE addition with a result below the normal range is exact. So the bound of the sum needs no extra term for underflow. |
| the final division of `avg` | The bound is B/n + 2^-52 · max(\|confit\|, \|oracle\|) + 2^-1074. Each engine rounds its division once. The term with 2^-52 covers these two roundings. 2^-1074 is the smallest positive DOUBLE, and it covers a result below the normal range. |
| the sum of the absolute values | The checker computes it with Python's `math.fsum`, which rounds the exact sum once. |

The checker is the comparison function that the loop adds to the campaign and the tests, for DOUBLE
`sum` and `avg` over matched static rows. It accepts confit's answer if the distance is in the bound.

**Options.**

1. **Approve the definitions.** Confit serves DOUBLE `sum` and `avg` over matched static rows. The loop
   adds the checker.
2. **Serve only the groups whose sum the order cannot change.** A group qualifies if it has at most two
   values. A group also qualifies if each of its values is an integer and the sum of their absolute
   values is below 2^53. In such a group, each partial sum is an integer below 2^53, and a DOUBLE holds
   each such integer exactly. So each addition is exact. If a static table has a group that does not
   qualify, confit refuses the build by name. This option needs no bound.
3. **Keep refusing DOUBLE `sum` and `avg`.**

**Recommendation.** Choose option 1. It serves each group whose values and sum of absolute values are
finite, and its bound accepts the sums that change between runs of the oracle. Option 2 refuses most
groups of values with fractions, such as prices with cents.
