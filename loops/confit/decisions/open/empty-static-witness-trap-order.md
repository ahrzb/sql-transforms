# Empty-static witness: must the padded trap match confit's category?

**Question.** The [empty-static ruling](../closed/empty-static-join-trap-timing.md) excuses a case only under
one condition. DuckDB must trap "with confit's error category" when it re-runs the case with one plain row in
each empty static table. This re-run is the witness. A padded trap is the trap that the witness raises.

Sometimes the row side has two different traps in two different rows. DuckDB evaluates one column at a time.
Confit evaluates one row at a time. So the two engines can name different traps first.

Should the witness accept any trap in this case, as the ordinary trap comparison (`AGREE_TRAP`) does? Or
should it keep the requirement for the same category?

**Evidence** (DuckDB 1.5.5, optimizer off, measured 2026-10-06). The evidence is campaign seed 4313391,
shrunk to this query:

```sql
SELECT '0' AS o0 FROM (
  SELECT (- c0.f1) AS i0,
         (CAST(CAST(c0.f0 AS DOUBLE) AS VARCHAR) LIKE '%e%') AS i1
  FROM __THIS__ WHERE c2) AS sub
JOIN s0 ON (i0 = s0.c0)
```

Row 1 has `c0.f0 = 'abcdefghij...'`. The cast to DOUBLE fails for this row. Row 2 has
`c0.f1 = -2147483648`. The negation overflows for this row.

| static `s0` | confit | DuckDB optimizer off | verdict today |
|---|---|---|---|
| empty | traps: Conversion Error (row 1, column i1) | 0 rows | OPT_EMULATED |
| one row | traps: Conversion Error | traps: Out of Range, negation overflow (row 2, column i0) | AGREE_TRAP |

If the static table has one row, the two engines agree under the ordinary comparison. So the empty-static
difference comes from this ruling's mechanism. The witness rejects the case for one reason only. DuckDB
evaluates `i0` over the whole chunk before it evaluates `i1`. Confit evaluates row 1 fully first.

**Options.**
1. **The witness accepts any trap,** as `AGREE_TRAP` does. The excused class grows. It then includes cases
   that combine an empty static table with this multi-trap order.
2. **Keep the category.** Such cases stay findings (OPT_EMULATED or DIVERGE_TRAP). They recur in campaigns.
   Each one costs a triage.
3. **Model DuckDB's column order** for the first trap of a stage. For each projection column, confit would
   evaluate the whole batch before the next column. This changes the serving order everywhere, not only
   here. So it is a separate and larger decision.

**Recommendation.** Choose option 1. The witness exists to show that DuckDB runs the row side and traps once
it does. It does so here.
