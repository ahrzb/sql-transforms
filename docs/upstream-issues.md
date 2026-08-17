# Upstream issues — findings about other people's software

Third-party behaviour we believe is a defect, collected here for
**AmirHossein to review and file**. Nothing here is reported upstream by the
agent; this file is the queue, not the submission.

Each entry states what was MEASURED and what is INFERRED, because the two
carry different weight in a bug report. A "measured" line means a probe was
run in this repo and the output pasted verbatim; "inferred" means we reasoned
about a mechanism we cannot see.

| # | Project | Status | One line |
|---|---|---|---|
| 1 | DuckDB 1.5.5 | ready to file | `substr` with a negative start disagrees with DuckDB's own constant folder |
| 2 | DuckDB 1.5.5 | needs a decision | optimizer changes whether a trapping subexpression runs |
| 3 | DuckDB | relayed, unverified | arrow scan + constant filter uses IEEE NaN order, native tables do not |
| 4 | DuckDB | relayed, unverified | feeding a connection its own undrained `.arrow()` reader neither serves nor refuses |

---

## 1. `substr` with a negative start: four code paths, two answers

**Measured**, DuckDB 1.5.5, 2026-08-17. One expression, four ways of reaching
it:

```sql
CREATE TABLE t (s VARCHAR);
INSERT INTO t VALUES ('hello'), ('x');

SELECT substr(s, -10, 8) FROM t;          -- optimizer ON  -> 'hello', 'x'
SELECT substr('hello', -10, 8);           -- optimizer ON  -> 'hel'
-- with PRAGMA disable_optimizer:
SELECT substr(s, -10, 8) FROM t;          --               -> 'hel', ''
SELECT substr('hello', -10, 8);           --               -> 'hel'
```

Three of the four agree on `'hel'`. The outlier is the optimized vectorized
column path, which returns the whole string.

The consistent rule — and the one DuckDB's own constant folder implements —
maps a negative start end-relative to a (possibly non-positive) position and
then intersects the window `[pos, pos+len)` with the string. Under it,
`substr('hello', -10, 8)` covers positions -4..3, of which 1..3 survive:
`'hel'`. The outlier instead clamps the start to 1 and keeps the full length,
which is the same rule `substr(s, 0, 3)` does NOT follow (it correctly gives
`'he'`, not `'hel'`).

More rows, all measured, showing it only bites when the mapped position falls
at or below 0:

| input | on/lit | on/col | off/lit | off/col |
|---|---|---|---|---|
| `('hello',-10,8)` | `'hel'` | **`'hello'`** | `'hel'` | `'hel'` |
| `('ab',-4,2)` | `''` | **`'ab'`** | `''` | `''` |
| `('ab',-3,2)` | `'a'` | **`'ab'`** | `'a'` | `'a'` |
| `('hello',-4,8)` | `'ello'` | `'ello'` | `'ello'` | `'ello'` |
| `('hello',0,3)` | `'he'` | `'he'` | `'he'` | `'he'` |

**Inferred**: a constant-folded call and the vectorized kernel implement the
negative-start clamp differently. We have not read DuckDB's source to confirm.

**Why it is worth filing**: a query returns a different answer depending on
whether an argument is a literal, which is invisible to the user and survives
into results.

**Note on provenance**: this repo already knew half of it. An xfail-strict
pin (`test_substr_constant_fold_divergence`) recorded the literal-vs-column
split before 2026-08-17; what is new is the optimizer-on/off dimension, which
identifies WHICH of the two paths is the odd one out.

## 2. The optimizer decides whether a trapping subexpression runs

**Measured**, DuckDB 1.5.5, 2026-08-17:

```sql
CREATE TABLE t (i INTEGER); INSERT INTO t VALUES (2147483647);
SELECT (i + 1) > 5 FROM t;
-- optimizer ON : true
-- optimizer OFF: Out of Range Error: Overflow in addition of INT32 (2147483647 + 1)!
```

and a sharper one, where the answer depends on the table's *insert history*
rather than its contents:

```sql
CREATE TABLE t (c0 TINYINT);
INSERT INTO t VALUES (-128);                     SELECT (c0*32) IS NOT NULL FROM t;  -- true
-- fresh table:
INSERT INTO t VALUES (-128), (NULL);
DELETE FROM t WHERE c0 IS NULL;                  SELECT (c0*32) IS NOT NULL FROM t;  -- OVERFLOW
```

After the `DELETE` the table's rows are identical to the first case, and the
answers differ. `statistics_propagation` proves the predicate from a stored
null statistic that the delete does not clear.

**This may well be intended.** Most engines treat "an expression that would
error is optimized away" as acceptable, and `IS NOT NULL` folding from
statistics is a normal optimization. It is listed here because we could not
find it documented, and because the history-sensitivity is surprising enough
that a user could reasonably report it.

**Recommended framing if filed**: a question about intended semantics ("are
runtime errors guaranteed to fire when the expression is otherwise
eliminable?"), not a bug report.

## 3. Arrow scan vs native table comparison order — RELAYED, NOT VERIFIED

Recorded in `packages/confit/tests/test_duckdb_interpreter.py`: "duckdb
pushes constant filters into registered-arrow scans with IEEE NaN semantics,
which disagrees with its own native-table comparison order (adversarial
probe, 2026-07-26)".

Predates this session and was **not re-measured**. Needs a fresh probe before
it is filed.

## 4. Undrained `.arrow()` reader fed back to its own connection — RELAYED

Recorded in `packages/sql-transform/sql_transform/_duckdb_arrow_test.py` as a
strict-xfail: registering an undrained `.arrow()` reader back onto the
connection that produced it "neither serves nor refuses". The file notes an
earlier version overstated this as a deadlock from a run of three.

Predates this session and was **not re-measured**. TASK-65 already tracks
researching it to report quality.
