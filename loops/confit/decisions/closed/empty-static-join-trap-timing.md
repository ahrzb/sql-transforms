# Trap timing of a join on an empty static table

**Question.** When an INNER or CROSS join reads a static table with no rows,
does a trap on the row side count?

**What DuckDB does** (1.5.5, optimizer off, measured 2026-10-05). The answer is
empty either way, but whether the row side is evaluated depends on its pipeline
shape:

| row side | empty static |
|---|---|
| `SELECT 1 FROM (SELECT c0 + c0 AS x FROM t) sub JOIN s ON x = s.k` | traps |
| `SELECT 1 FROM (SELECT c0 + c0 AS x FROM t WHERE c0 > 0) sub JOIN s ON x = s.k` | 0 rows |
| `SELECT 1 FROM t JOIN s ON c0 + c0 = s.k` (key computed in the join) | 0 rows |
| `SELECT 1 FROM (SELECT c0 + c0 AS x FROM t) sub CROSS JOIN s` | traps |
| `SELECT 1 FROM t CROSS JOIN s WHERE CAST(c1 AS DOUBLE) = s.k` | 0 rows |

With one or more static rows every row above traps. confit evaluates the row
side per row, so it traps in all five. The difference is when DuckDB's executor
stops pulling chunks, not what the query means.

**Seeds.** Nightly #305: 1994509, 2286807, 4001288, 4100483 (DIVERGE_TRAP;
DuckDB returns no rows, confit traps).

**Options.**
1. Name the class in the oracle: a DIVERGE_TRAP whose case joins an empty
   static is EXCLUDED, like the resource ceilings.
2. Refuse at construction: an INNER or CROSS join whose static table is
   empty refuses by name. This costs the agreeing cases too.
3. Model it: run the program over zero rows when such a static is empty. This
   matches the rows-0 shapes above and diverges on the trapping ones, so it
   only moves the mismatch.

**Ruling (owner, 2026-10-05).** Option 1, with a witness: the difference is
excluded only when it is this mechanism. The rule `empty-static-trap-timing` in
`fuzz/exclusions.py` excuses a case when an INNER or CROSS join in its plan reads an
empty static, confit traps on both backends, the optimizer-off baseline returns zero
rows, and DuckDB, re-run with one row of plain non-NULL values added to each such
static, traps with confit's error category. Any other difference in such a case (a
different trap, rows, a backend split) stays a finding. Its canaries are the four seeds
above; `tests/test_exclusions.py` also plants faults inside the scope and checks that
they are not excused.
