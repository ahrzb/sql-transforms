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

**Ruling (owner, 2026-10-15).** The owner chose "Approve definitions" on a decision card in the project
chat at 18:18 UTC. This is option 1. The card added two notes to the definitions above:

- The reason that B holds in each order of the additions is the error bound of N. J. Higham for a sum
  in any order.
- One amendment: confit refuses the build if B is not finite, and not only if the sum of the absolute
  values is not finite, because a partial sum can round past the sum of the absolute values.

The card also reported a check on the oracle: on 30 random tables of up to 200,000 rows, whose sums
cancel a lot, the largest distance was 1.4e-6 of B.

Later on the same day, the owner wrote that a ruling is an opinion, and that the loop can object to it,
also later. The owner named this bound as a question that needs numerical analysis. The section
"Numerical analysis" below gives this analysis. It changes no definition. It proves each definition, and
it applies the amendment as a rule that it proves sufficient.

**How the loop applies the ruling.**

1. The build adds the values of a group that are not NULL in table order, from +0.0 (`Acc::Float` in
   `packages/confit/src/duckdb/mod.rs`). For `avg`, it divides the sum by n once.
2. The build refuses a group by name if a value is a NaN or an infinity, or if Σ|v_i| + B is not finite.
   It computes Σ|v_i| exactly and rounds it once, with the algorithm of Python's `math.fsum` (`ExactSum`
   in the same file). A test compares the two on 300 random groups, bit for bit.
3. Confit serves the aggregate only as the whole SELECT item of a subquery that is a whole output column.
   A derived table or a CTE can pass the column on with no change. Confit refuses each other read of the
   value by name: an expression, a WHERE, a join condition, a struct or a list output, and a value that a
   SQL function reads more than once (`float_bounds` in `packages/confit/src/specializer/mod.rs`).
4. `DuckDBInferFn.plan_facts["bounds"]` names each such column. Then the campaign builds the query a
   second time, with `float_bounds=True` (`fuzz.oracle.run_case`). In this build, the column holds the
   bound of the group in place of the value: 0.0 for n ≤ 2, B for `sum`, B/n for `avg`, and NULL for
   n = 0.
5. `confit.compare.within_float_bound` compares the two answers. It pairs the rows as the multiset
   comparison does, and it computes the distance exactly with Python's `Fraction`. If a value needs the
   bound, the verdict is AGREE with the `klass` `float-reduction-bound`. The campaign report counts these
   verdicts, with their seeds.

**Numerical analysis.** The notation is u = 2^-53 (the unit roundoff of a DOUBLE), eps = 2u = 2^-52,
A = Σ|v_i| (the exact sum), and γ_k = ku/(1 − ku). Â and B̂ are the values that confit computes for A and
B. An order of the additions is a binary tree over the n values. The parts of the oracle start at +0.0,
and an addition of +0.0 is exact. So each order of the oracle is such a tree.

1. **The bound in each order.** For each tree, the computed sum differs from the exact sum by at most
   γ_{n−1}·A (N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd edition, 2002, section
   4.2). For a sum in one sequence, S. M. Rump proves (n − 1)·u·A, without the denominator ("Error
   estimation of floating-point summation and dot product", BIT 52, 2012). So two orders differ by at most
   2γ_{n−1}·A. This is about (n − 1)·eps·A, which is B/2.
2. **The rounding of B.** Confit rounds A once to Â. Then it rounds the product of Â and (n − 1)·2^-51
   once (`float_bound` in `duckdb/mod.rs`). So B̂ ≥ (1 − u)²·B, and B̂ is above 2γ_{n−1}·A for each n
   below 2^51. Below the normal range, the product can lose at most 2^-1075 more. Each DOUBLE is a
   multiple of 2^-1074, so a distance that is not 0 is at least 2^-1074. The factor 2 of B also covers
   this loss.
3. **Overflow.** The amendment refuses a group if B is not finite, because a partial sum can round past
   A. The loop applies this reason directly: the build refuses a group if Â + B̂ is not finite. This rule
   refuses each group that the amendment refuses, because B̂ is a part of the sum. The rule is sufficient
   for each order, by induction over the tree. Each addition rounds the exact sum of two computed partial
   sums, and the magnitude of this exact sum is at most A·(1 + γ_{n−1}). For n ≥ 2, this magnitude is
   at most Â + B̂. The reason is that B̂ is about 4(n − 1)·u·Â. This is more than γ_{n−1}·A and the
   rounding of Â together, which are about (n − 1)·u·Â + u·Â. A rounding to nearest gives an infinity
   only from a magnitude of 2^1024 − 2^970 or more. Â + B̂ is below this magnitude, because its rounding
   is finite. For n = 1, the sum is the one value.
4. **The division of `avg`.** Each engine divides its sum by n once. If a quotient q is in the normal
   range, the rounding moves it by at most u·|fl(q)|. So the two divisions add at most
   u·(|confit| + |oracle|), which is at most eps·max(|confit|, |oracle|). Below the normal range, each
   rounding moves a quotient by at most 2^-1075, and the term 2^-1074 covers both. The distance of the
   two sums adds at most B/n. The bound build rounds B/n once more, and the factor 2 of B covers it.
5. **Underflow.** The exact sum of two DOUBLEs is a multiple of 2^-1074. If its magnitude is below
   2^-1022, a DOUBLE holds it exactly, and the addition does not round.
6. **The sign of zero.** The sum starts at +0.0. With a rounding to nearest, x + (−x) is +0.0, and
   +0.0 + (−0.0) is +0.0. So no partial sum is −0.0, and −0.0 and 0.0 give the same sum. Thus the value
   that represents the DISTINCT class of zero does not change the sum.

**Measurement** (2026-10-15, master 17316e0 with this change).

- **Large tables.** The loop measured three static tables on the oracle of the campaign, which copies
  each table into DuckDB (optimizer off). Each value has a random sign and a random magnitude. For each
  table, the loop ran `sum` and `avg` 3 times each. No answer of the oracle was past the bound.

  | rows | groups | magnitudes | groups bit-exact in one run | largest distance |
  |---|---|---|---|---|
  | 300,000 | 300 | 10^-3 to 10^3 | 12 to 15 | 0.00095 of B |
  | 600,000 | 64 | 10^-8 to 10^8 | 1 to 4 | 0.00011 of B |
  | 200,000 | 16 | 10^-15 to 10^15 | 0 | 0.000019 of B |

  If the measurement registers an Arrow table of one chunk in place of the copy, the oracle reads it in
  one sequence. Then each group of the three tables is bit-exact in each run.
- **Campaign.** On the 50,000 seeds 31,000,000 to 31,049,999, confit refused 14 cases with a DOUBLE
  `sum` or `avg` before this change. Now 7 of them agree bit for bit. 5 refuse by name because an
  expression reads the value, and 2 refuse for another reason that comes first. No other verdict
  changed, and no verdict is gated. The static tables of the generator have 8 rows or fewer, so no case
  needed the bound.
