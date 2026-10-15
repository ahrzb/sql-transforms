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

**Ruling (owner, 2026-10-15).** In the project chat at 18:18 UTC, the owner chose option 1, "Approve
definitions". The question in the chat added two notes to the definitions above:

- The reason that B holds in each order of the additions is the error bound of N. J. Higham for a sum
  in any order.
- One amendment: confit refuses the build if B is not finite, and not only if the sum of the absolute
  values is not finite, because a partial sum can round past the sum of the absolute values.

The question also reported a check on DuckDB 1.5.5 by the agent that asked it. Over 30 random tables of
up to 200,000 rows, whose sums cancel a lot, the largest distance was 1.4e-6 of B. The question does not
say which two answers it compared, and the repository does not keep this check. The section
"Measurement" below gives the check of the loop.

At 18:26 UTC, the owner wrote that a ruling is an opinion. The loop can object to a ruling at any time,
also after the loop applies it. The loop asks the owner before it acts against a ruling. The owner named
this bound as an example of a question that needs numerical analysis. The section "Numerical analysis"
below gives this analysis. It changes no definition, and it shows why each definition holds.

**Where the loop refuses more than the text of the ruling.** Two steps below add refusals that the text
of the ruling does not name. The loop reads both steps as a part of the ruling, for these reasons:

- **The overflow rule** (step 2). The build refuses a group if Σ|v_i| + B is past the range of a DOUBLE.
  The amendment names B alone. But B = (n − 1)·2^-51·Σ|v_i|, so B is smaller than Σ|v_i| for each n
  below 2^51. Thus the text of the amendment refuses no group that the definitions serve, and a partial
  sum can still round past the range. For example, the three values 2^1023 − 2^970, 2^1022 and
  2^1022 − 2^970 have a Σ|v_i| equal to the largest DOUBLE, and their B is finite. In table order, the
  oracle answers an infinity (measured on DuckDB 1.5.5). In the reverse order, each addition is exact,
  and the sum is the largest DOUBLE. Item 4 of "Numerical analysis" proves that the rule of step 2
  prevents such a sum. The rule refuses a group that the definitions serve only if Σ|v_i| is within
  about B of the largest DOUBLE.
- **The whole output column** (step 3). B bounds the distance of the value itself. After an expression
  or a comparison reads the value, a distance in B can become any distance. For example, `v > 0` can be
  true in confit and false in the oracle. So the checker can compare only the value itself, and confit
  refuses each other read of the value.

**How the loop applies the ruling.**

1. The build adds the values of a group that are not NULL in table order, from +0.0. The accumulator of
   a DOUBLE sum does this (`Acc::Float` in `packages/confit/src/duckdb/mod.rs`). For `avg`, the build
   divides the sum by n once.
2. The build refuses a group by name if a value is a NaN or an infinity, or if Σ|v_i| + B is past the
   range of a DOUBLE. It computes Σ|v_i| exactly and rounds it once, with the algorithm of Python's
   `math.fsum` (the type `ExactSum` in the same file). A test compares `ExactSum` with `math.fsum` on
   300 random groups, and the two are bit-exact. The function `float_bound` in the same file computes B
   from the rounded Σ|v_i|.
3. Confit serves the aggregate only as the whole SELECT item of a subquery, and only if this subquery is
   a whole output column. A derived table or a CTE can pass the column on with no change. Confit refuses
   each other read of the value by name: an expression, a WHERE, a join condition, a struct or a list
   output, and a value that a SQL function reads more than once. The function `float_bounds` in
   `packages/confit/src/specializer/mod.rs` finds these reads.
4. `DuckDBInferFn` is the Python class that builds a confit function. Its `plan_facts["bounds"]` names
   each column of step 3. For such a query, the campaign builds the query a second time, with the
   argument `float_bounds=True` (`fuzz.oracle.run_case`). This second build is the bound build. In the
   bound build, each such column answers the bound of the group in place of the value. This bound is
   NULL for n = 0, and 0.0 for n = 1 and n = 2. For n of 3 or more, it is B̂ for `sum` and B̂/n for
   `avg`. B̂ is the value of B that confit computes (item 3 of "Numerical analysis").
5. The function `within_float_bound` in `confit.compare` compares the answers of confit and the oracle.
   It compares the rows in any order, as the comparison of other answers in the campaign does
   (`multiset` in `confit.compare`). It computes each distance exactly, with Python's `Fraction`. A
   value with the bound 0.0 must be bit-exact. For another value of a `sum`, the distance can be at most
   the bound. For `avg`, it can be at most the bound + 2^-52·max(|confit|, |oracle|) + 2^-1074, as the
   definitions say.
6. If the two answers are bit-exact, the verdict is `AGREE`, as before. If they are not bit-exact, but
   each distance is in the bound, the verdict is `AGREE` with the reason `float-reduction-bound` in its
   field `klass`. The campaign report counts these verdicts, with their seeds. If a distance is past the
   bound, the verdict is `DIVERGE_VALUE`, as for any other value that differs.

**Numerical analysis.** The notation is u = 2^-53 (the unit roundoff of a DOUBLE), eps = 2u = 2^-52,
A = Σ|v_i| (the exact sum of the absolute values), and γ_k = ku/(1 − ku). Â and B̂ are the values that
confit computes for A and B. δ = 2^-1074 is the smallest positive DOUBLE, and each DOUBLE is a multiple
of δ. The analysis is for n below 2^51.

An order of the additions is a binary tree over the n values. Each inner node adds the sums of its two
children, with one rounding. Table order is the tree that adds each value to the sum of the values
before it. The oracle can divide the rows of a group into parts, and threads add the parts at the same
time. Each part starts its sum at +0.0, and an addition to +0.0 is exact. So each order of the oracle
is such a tree, with at most n − 1 additions that round.

1. **n of 1 or 2.** For n = 1, each tree gives the one value (item 7 gives the case of −0.0). For n = 2,
   each tree adds the two values with one rounding. An IEEE addition gives the same result in both
   orders of its operands. So for n of 1 or 2, the comparison is exact.
2. **The bound in each order.** For each tree, the computed sum differs from the exact sum by at most
   γ_{n−1}·A (N. J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd edition, 2002, section
   4.2). For table order, S. M. Rump proves the smaller bound (n − 1)·u·A ("Error estimation of
   floating-point summation and dot product", BIT 52, 2012). So the sums of two orders differ by at
   most 2γ_{n−1}·A. For a small n, this is about (n − 1)·eps·A, which is B/2. For each n below 2^51,
   (n − 1)·u is at most 1/4, so 2γ_{n−1}·A is at most 2B/3.
3. **The rounding of B.** The checker uses B̂, and B̂ can be smaller than B. Confit rounds A once to Â.
   Then `float_bound` multiplies Â by (n − 1)·2^-51, and the product rounds once. In the normal range,
   B̂ ≥ (1 − u)²·B, which is more than 2B/3. Below the normal range, B̂ ≥ (1 − u)²·B − δ/2. A distance
   d that is not 0 is at least δ, and d is at most 2B/3 by item 2. So B ≥ 1.5δ, and then B̂ − d > −δ.
   B̂ and d are multiples of δ, so B̂ ≥ d. Thus the factor 2 in B covers the rounding of B.
4. **Overflow.** Let T = 2^1024 − 2^970. A rounding to nearest gives an infinity only from a magnitude
   of T or more. Confit computes Â + B̂ with one rounding. If this rounding is finite, the exact Â + B̂
   is below T, and the rule of step 2 serves the group. Then no addition of any tree gives an infinity.
   The proof is an induction over the tree, from the leaves. For a node, let k be the number of
   additions that compute its sum, and A_k the sum of the absolute values of its leaves. The claim is
   that the computed sum s of the node has |s| ≤ (1 + γ_k)·A_k. The claim holds for a leaf, with k = 0.
   At an inner node, the claim for the two children gives a magnitude of at most (1 + γ_{k−1})·A_k for
   their exact sum. This is at most (1 + γ_{n−1})·A. Also A ≤ (1 + u)·Â, and
   B̂ ≥ (1 − u)·4(n − 1)·u·Â. For n of 2 or more, these give (1 + γ_{n−1})·A ≤ Â + B̂ < T. So the
   addition does not give an infinity, and it rounds with a relative error of at most u. Thus
   |s| ≤ (1 + u)(1 + γ_{k−1})·A_k ≤ (1 + γ_k)·A_k, and the claim holds for the node.
5. **The division of `avg`.** Each engine divides its sum by n with one rounding. If a quotient q
   rounds to a value in the normal range, the rounding moves it by at most u·|fl(q)|. So the two
   divisions add at most u·(|confit| + |oracle|) to the distance. This is at most
   eps·max(|confit|, |oracle|). Below the normal range, each division moves its quotient by at most
   δ/2, and the term δ = 2^-1074 covers the two divisions. Before the divisions, the two sums differ by
   at most 2B/3 (item 2). So the exact quotients differ by at most 2B/(3n). The bound build rounds B̂/n
   once more. As in item 3, the margin from 2B/3 to B and the term δ cover this rounding.
6. **Underflow.** The exact sum of two DOUBLEs is a multiple of δ. If its magnitude is below 2^-1022, a
   DOUBLE holds it exactly, so the addition does not round. Thus the bound of a sum needs no extra term
   for underflow.
7. **The sign of zero.** Both engines start each sum at +0.0. With a rounding to nearest, x + (−x) is
   +0.0, and +0.0 + (−0.0) is +0.0. So no partial sum is −0.0, and a value −0.0 does not change a sum,
   as 0.0 does not. For `sum(DISTINCT v)`, −0.0 and 0.0 are one value, and the order of the rows
   decides which of the two the sum adds. This choice does not change the sum.

**Measurement** (2026-10-15, master 17316e0 with this change).

- **Large tables.** The loop measured three static tables on the oracle of the campaign
  (`confit.oracle.Oracle`), which copies each table into DuckDB. Each value has a random sign and a
  random magnitude. For each table, the loop ran `sum` and `avg` 3 times each. For each group, the loop
  compared the oracle's answer with the table-order answer. Each distance was in the float-reduction
  bound.

  | rows | groups | magnitudes | groups with bit-exact answers, in one run | largest distance of the two answers |
  |---|---|---|---|---|
  | 300,000 | 300 | 10^-3 to 10^3 | 12 to 15 | 0.00095 of B |
  | 600,000 | 64 | 10^-8 to 10^8 | 1 to 4 | 0.00011 of B |
  | 200,000 | 16 | 10^-15 to 10^15 | 0 | 0.000019 of B |

  The loop also gave the three tables to DuckDB in a second way. It registered each table as an Apache
  Arrow table of one chunk, which DuckDB reads in place, with no copy. In this way, the two answers were
  bit-exact for each group in each run. The campaign copies its tables, so the table above uses the
  copy.
- **Campaign.** On the 50,000 seeds 31,000,000 to 31,049,999, confit refused 14 cases with a DOUBLE
  `sum` or `avg` before this change. Now 7 of them agree, and in each of the 7, the two answers are
  bit-exact. 5 refuse by name because an expression reads the value. 2 refuse for another reason first:
  they have a WHERE clause that can remove the row of a request, and `shape="map"` answers one row for
  each request. No other verdict changed, and no verdict is gated. No case needed the bound. The static
  tables of the generator have 8 rows or fewer, and the oracle adds the rows of such a small table in
  table order (see "What the oracle does").
