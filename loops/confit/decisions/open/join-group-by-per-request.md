# JOIN with GROUP BY as an aggregate for each request

**Question.** The owner ruled an order of five kinds of queries for confit to serve next
([next query classes](../closed/next-query-classes.md)). The fifth kind is an aggregate over the static rows
that one request matches. SQL has two spellings for it:

1. A scalar subquery: `SELECT (SELECT sum(s.v) FROM s WHERE s.k = t.k) FROM __THIS__ AS t`.
2. A join with GROUP BY: `SELECT t.k, sum(s.v) FROM __THIS__ AS t LEFT JOIN s ON s.k = t.k GROUP BY t.k`.

Confit refuses both spellings today. The loop starts with the scalar subquery, which needs no ruling.
Should confit also serve the join with GROUP BY, and if so, how?

**How the campaign compares today.** A campaign case has a batch of requests. The campaign runs the
oracle on the whole batch, and it runs confit on the same batch. Then it compares the two answers.

**What the oracle does** (DuckDB 1.5.5, optimizer off, measured 2026-10-14). The static table `s` has
three rows (k, v): (1, 10), (1, 20) and (2, 5). A batch has four requests (id, k): (1, 1), (2, 1), (3, 2)
and (4, 3). The table below gives the oracle's answer for the whole batch. The last column gives the
oracle's answer when the batch holds only one request, in four runs, one for each request.

| query | answer for the batch | answer in the run of each request alone |
|---|---|---|
| the scalar subquery | 4 rows. The sum is 30 for ids 1 and 2, 5 for id 3, and NULL for id 4. | one row in each run, with the same sum as in the batch |
| `LEFT JOIN` with `GROUP BY t.k` | 3 rows. The sum is 60 for k = 1, 5 for k = 2, and NULL for k = 3. | one row in each run. The sum is 30 in the run of id 1, 30 in the run of id 2, 5 in the run of id 3, and NULL in the run of id 4. |
| `LEFT JOIN` with `GROUP BY t.id, t.k` | 4 rows, with the same sums as for the scalar subquery | one row in each run, with the same sum as in the batch |

- The scalar subquery gives each request its own answer, whatever the other requests of the batch are.
- With `GROUP BY t.k`, the two requests with k = 1 make one group. The oracle answers one row for the two
  requests. Its sum, 60, adds the matches of both requests. So the answer for one request depends on the
  other requests of the batch.
- The goal of the confit loop (`loops/confit/goal.md`, section Scope) says: "The answer for one request
  must not depend on other requests in the batch." The same section puts aggregation across request rows
  outside the goal.
- If a GROUP BY lists a request column with a different value in each request, such as `t.id` in the
  third row of the table, the batch gives the same rows as each request alone. Confit builds the
  function before it sees a request. So the build cannot know that a column has a different value in
  each request.
- The two spellings also count differently. For a request that matches no static row, `count(*)` gives 1
  after a `LEFT JOIN`, because the join keeps the request with NULL values. The scalar subquery gives 0.

**Options.**

1. **Serve only the scalar subquery.** Confit keeps the refusal of a join with GROUP BY. The refusal
   names the scalar subquery as the spelling that confit serves.
2. **Answer each request alone.** If the GROUP BY lists only request columns, confit answers each request
   as if the batch held only that request. For these queries, the campaign changes: it runs the oracle on
   each request alone, and compares confit's answer with these runs. If two requests of a batch have the
   same GROUP BY values, a user who runs DuckDB on the batch gets a different answer from confit.
3. **Serve it with a declared key.** `DuckDBInferFn` is the Python class that builds a confit function. A
   new argument of it names a request column. The caller promises that this column has a different value
   in each request of a batch. If the GROUP BY lists that column, confit serves the query. Then the batch
   and each request alone give the same answer, and the campaign compares as it does today.

**Recommendation.** Choose option 1. The campaign then compares confit with the oracle as it does today,
and the scalar subquery gives the same aggregate for each request. If the transforms that people write for
confit use a join with GROUP BY, option 3 can come later.
