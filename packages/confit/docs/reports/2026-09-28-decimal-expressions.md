# Decimal expressions, measured against the goal (2026-09-28)

**What this is.** A reading of the decimal-expressions work against
[the goal](../goal.md). It follows that document's three commitments:
- accepted transforms keep the oracle contract;
- unsupported constructs refuse by name;
- the SQL surface grows without weakening the contract, at in-process latency.

Measured on master `3f8ecf2` (ahrzb/sql-transforms#307) plus the follow-up
branch with the generator and kernel changes. The rules themselves are in
[the spec](../specs/decimal-expressions.md), read from DuckDB v1.5.5's source.

---

## 1. Outcomes

**A class of silently wrong answers is gone.** DuckDB types `2.5` as
`DECIMAL(2,1)` and computes on it exactly; confit computed in DOUBLE. Where a
decimal result became a boolean or an integer, the answers differed:
- `0.1 + 0.2 = 0.3` was `false` in confit and `true` on DuckDB;
- `a * 0.1 = 0.3` differed the same way;
- `CAST(-2.5 AS BIGINT)` was `-2` in confit and `-3` on DuckDB (nightly seed 1119492).

All of these agree now.

**Decimals moved from "unshipped" to "agree", and no agreement was lost.**
Campaign seeds 0..19999, the same seeds on master and on the change, with the
master verdicts as a baseline:

| verdict | master | decimals shipped |
|---|---:|---:|
| AGREE | 12,763 | 12,920 |
| UNSHIPPED | 157 | 0 |
| REFUSED | 6,504 | 6,504 |
| agreements lost vs. master | — | 0 |

The 157 cases the oracle used to set aside as UNSHIPPED now agree, and the refusal
count is unchanged. That UNSHIPPED exemption is deleted, so a DECIMAL-vs-DOUBLE
difference now fails the gate like any other divergence.

**The campaign now reaches decimals often, and they hold.** Bare decimal literals
went from 5% to 30% of the generator's float literals.
- Seeds 0..19999 found one new defect class, four seeds. A decimal operator over a
  constant NULL is DuckDB's untyped NULL, and confit had given it a type. It is fixed
  and pinned.
- Seeds 20000..39999 found no gated verdict.

**Latency stays in-process.** `infer_arrow` p50 on the Cranelift backend, for
`a * 0.5 + 1.25`, `CAST(a * 2.5 AS BIGINT)` and `round(a * 0.125, 2)`, against
the same query spelled in DOUBLE:

| rows | DECIMAL | DOUBLE | projection only |
|---:|---:|---:|---:|
| 1 | ~17 µs | ~15 µs | ~10 µs |
| 1024 | 110 ns/row | 50–56 ns/row | 29 ns/row |

Two kernel changes got DECIMAL here from 141–150 ns/row:
- a compile-time power-of-ten table;
- native i128 `+ - *` where the width cannot overflow.

Decimal work still costs about 55 ns/row over DOUBLE for these three expressions. The
checked casts and rounding go through a helper call per row.

## 2. Required behavior: the contract

- **Parity.** All of the following are read from the DuckDB source and checked by
  `tests/test_decimal_expressions.py` (94 cases, rows, types and traps, both
  backends):
  - literal typing;
  - `+ - * %` with the 18- and 38-digit caps;
  - overflow checks against the storage bound;
  - half-away-from-zero casts;
  - the common type for comparisons, CASE, COALESCE, greatest and IN;
  - the DECIMAL overloads of `abs`, `ceil`, `floor`, `round` and `trunc`.

  `tests/test_decimals.py` covers decimal statics end to end against the live oracle.
- **Traps.** Overflow and cast failures trap with DuckDB's own messages, for example
  `Overflow in addition of DECIMAL(18) (…)`, `Failed to cast decimal value 301 to
  type INT8`, and `Casting value "…" to type DECIMAL(9,1) failed: value is out of
  range!`. Under the oracle's settings a failing constant traps per row, not at
  plan time, and confit does the same.
- **Refusals by name.** These remain:
  - casts into DECIMAL from DOUBLE, VARCHAR or BOOLEAN (DuckDB's double→decimal
    rounding and its string parser are not reproduced);
  - a DECIMAL join key expression against a non-DOUBLE build key;
  - `IN`/`BETWEEN` families whose common type is capped at 38 digits;
  - `round(DECIMAL, n)` with a non-constant `n`, which DuckDB refuses too.

## 3. Scope

The ledger entries `decimal-literal-typing` and `decimal-cast-rounding` are closed.
The last UNSHIPPED feature is gone: no output width is exempt from comparison. What is still open for decimals is listed in PLANS ("Decimal remainders"),
including making DECIMAL a row-column lane.

## 4. Not about decimals

Seed 18995 is a DIVERGE_TRAP that is identical on master: a constant NULL argument
to `repeat()` hides a constant INT64 overflow that DuckDB still evaluates. It is
queued as its own task and is not affected by this work.

## Environment and repro

- Linux x86-64, 4 cores, Python 3.14.7, DuckDB 1.5.5 oracle (optimizer off).
- Campaign: `uv run python -m fuzz.runner --seed 0 --n 20000 --workers 4 --cases
  cases.jsonl [--baseline master_cases.jsonl]` in `packages/confit`.
- Latency: `DuckDBInferFn.infer_arrow` p50 over ≥30 iterations after one warm call,
  release build. Absolute numbers vary about ±10% between runs on this machine; the
  ratios are stable across three runs.
