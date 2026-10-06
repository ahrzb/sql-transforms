# An INNER join whose static keys are all NULL

**Question.** In the [empty-static ruling](../closed/empty-static-join-trap-timing.md), the owner decided
that the campaign excuses a trap difference when an INNER or CROSS join reads a static table with no rows.
Should the ruling also excuse the same difference when an INNER join reads a static table whose join keys
are all NULL?

**What DuckDB does** (1.5.5, optimizer off, measured 2026-10-06). Take an INNER join on a key, such as
`ON x = s.k`. DuckDB first reads the static table. A static row whose key is NULL cannot match a request,
because `NULL = x` is never true in SQL. So DuckDB leaves that row out. If DuckDB leaves out every static
row, it acts as it does for a static table with no rows. This table repeats the three INNER queries of the ruling with three static tables:

| query | no rows | two rows, key NULL | one row, key 5 |
|---|---|---|---|
| `SELECT 1 FROM (SELECT c0 + c0 AS x FROM t) sub JOIN s ON x = s.k` | traps | traps | traps |
| `SELECT 1 FROM (SELECT c0 + c0 AS x FROM t WHERE c0 > 0) sub JOIN s ON x = s.k` | 0 rows | 0 rows | traps |
| `SELECT 1 FROM t JOIN s ON c0 + c0 = s.k` (key computed in the join) | 0 rows | 0 rows | traps |

DuckDB answers the same for the static table with no rows and for the table whose keys are all NULL.
Confit answers differently. It computes `c0 + c0` for each request, whatever the static table holds, so it
traps in all nine cases.

A CROSS join has no key. DuckDB pairs each request row with every static row, NULL values or not. So this
question does not cover CROSS joins.

**Cases.** The nightly of 2026-10-06 found one such case. Its seed is 4889739. Its static table has three
rows, and the join key is NULL in each row. The campaign gave the case the verdict OPT_EMULATED. This
verdict means that confit agrees with the optimizer-on reading but not with the oracle. Other cases of this
kind get the verdict DIVERGE_TRAP: confit traps, and the oracle returns rows.

**Options.**
1. **Extend the ruling.** The ruling added an exclusion to `fuzz/exclusions.py`. This option makes the
   exclusion also cover an INNER join on a key whose static table has no row with a non-NULL key. The
   witness of the ruling stays the same. DuckDB runs the case again with one more static row, whose values
   are plain and not NULL. The exclusion applies only if DuckDB then traps with confit's error category.
2. **Keep the ruling as it is.** Such cases stay findings. They come back each time that a generated static
   table has only NULL keys, and each one costs a triage.

**Recommendation.** Choose option 1. A static table whose keys are all NULL leaves DuckDB no row to match,
as a static table with no rows does. So the trap difference has the cause that the ruling already excuses.
