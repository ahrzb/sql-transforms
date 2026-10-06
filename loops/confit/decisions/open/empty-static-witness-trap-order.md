# Empty-static witness: must the padded trap match confit's category?

**Question.** The [empty-static ruling](../closed/empty-static-join-trap-timing.md)
excuses a case only when DuckDB, re-run with one plain row in each empty static,
traps "with confit's error category". When the row side has two different traps
in different rows, DuckDB (column at a time) and confit (row at a time) can name
different ones first. Should the witness accept any trap there, as the ordinary
trap comparison (`AGREE_TRAP`) does, or keep requiring the same category?

**Evidence** (DuckDB 1.5.5, optimizer off, measured 2026-10-06). Campaign seed
4313391, shrunk:

```sql
SELECT '0' AS o0 FROM (
  SELECT (- c0.f1) AS i0,
         (CAST(CAST(c0.f0 AS DOUBLE) AS VARCHAR) LIKE '%e%') AS i1
  FROM __THIS__ WHERE c2) AS sub
JOIN s0 ON (i0 = s0.c0)
```

Row 1 has `c0.f0 = 'abcdefghij...'` (the cast to DOUBLE fails); row 2 has
`c0.f1 = -2147483648` (the negation overflows).

| static `s0` | confit | DuckDB optimizer off | verdict today |
|---|---|---|---|
| empty | traps: Conversion Error (row 1, column i1) | 0 rows | OPT_EMULATED |
| one row | traps: Conversion Error | traps: Out of Range, negation overflow (row 2, column i0) | AGREE_TRAP |

With the row present the two engines agree under the ordinary comparison, so
the empty-static difference is this ruling's mechanism. The witness rejects it
only because DuckDB evaluates `i0` over the whole chunk before `i1`, and confit
evaluates row 1 fully first.

**Options.**
1. **The witness accepts any trap,** as `AGREE_TRAP` does. The excused class
   grows by cases that combine an empty static with this multi-trap order.
2. **Keep the category.** Such cases stay findings (OPT_EMULATED or
   DIVERGE_TRAP). They recur in campaigns, and each costs a triage.
3. **Model DuckDB's column order** for the first trap of a stage: evaluate each
   projection column over the batch before the next. This changes serving
   order everywhere, not just here, so it is a separate, larger decision.

**Recommendation.** Option 1: the witness exists to show that DuckDB runs the
row side and traps once it does, and it does here.
