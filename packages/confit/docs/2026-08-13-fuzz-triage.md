# Fuzz triage — 2026-08-13 campaign (arrow schema API gate)

Campaign: `python -m fuzz.runner --seed 0 --n 20000 --workers 8`, run as the
merge gate for PR #144 (content = master `5e88d38`). Result: **24 raw
findings, 9 classes — zero introduced by the migration.** Every substantive
finding replayed bit-for-bit on pre-migration master `217799d`; the three
that did not are load-flaky timeouts (below). Per-seed replays regenerated
on a quiet machine 21/24 (`--seed N --n 1`).

The engine-bug process applies: each NEW class gets an xfail-strict pin and
a ticket **on AmirHossein's word** — this document is the triage, not the
ticketing.

## Families

### 1. DuckDB traps where the engine serves — 4 seeds

DuckDB's vectorized/fold-time evaluation reaches expressions our row loop
never does (or traps at bind on constants we serve past). The m-8 phase-3
trap slice (TASK-99) is exactly this alignment; TASK-84 covers the INT32
literal-arithmetic member.

| seed | expression | DuckDB | ours |
|---|---|---|---|
| 3730 | `(2147483647 + 0) - unicode(c1)` in WHERE | INT32 overflow trap | serves (rows filter) |
| 7850 | `9223372036854775807 * 100` under `>=` in struct_pack | INT64 overflow at fold | serves |
| 8359 | `ln(-2.0e0)` inside nullif | domain trap at fold | serves NULL |
| 8687 | `9007199254740993 * 9007199254740993` in a filtered `IN (NULL)` | INT64 overflow | serves |

### 2. Engine traps where DuckDB serves (evaluation order) — 3 seeds

**Re-measured 2026-08-16.** 7560 and 11473 now match; 1667 is still live and
became TASK-117. TASK-85 (Done) folds a strict op whose SIBLING is a literal
NULL; in 1667 the trap is the BETWEEN's SUBJECT, which its ACs did not cover.

Our loop evaluates what DuckDB's plan short-circuits away. TASK-85's family
(DuckDB's NULL folding / filter ordering removes trapping subexpressions we
still evaluate). TASK-75 (WHERE AND/OR guards) is Done; these are the
BETWEEN-NULL and filtered-row variants it did not cover.

| seed | expression | our trap |
|---|---|---|
| 1667 | `CAST(c1 AS DOUBLE)` under `BETWEEN 61.591e0 AND NULL` | could not cast VARCHAR to DOUBLE |
| 7560 | `(c2 * c2) <= c2` under `AND (FALSE AND ...)` | INT64 mult overflow |
| 11473 | `(-16 * c0)` vs `length('%_')` in WHERE | INT64 mult overflow |

### 3. VARCHAR-to-integer cast semantics — 2 seeds — TASK-113

DuckDB casts a decimal-looking string to BIGINT by parsing and rounding
(`'1.5' -> 2`); we refuse ("could not cast VARCHAR to BIGINT"). Seeds
12626 (`CAST(CAST(1.5e0 AS VARCHAR) AS BIGINT)`), 13560 (same with
`63.699`). Pre-existing, reproduces on master.

Re-measured 2026-08-15 while ticketing it, and the class is wider than the
two seeds showed. The rounding is **half away from zero** (`'2.5' -> 3`),
which is NOT the half-to-even the DOUBLE path uses (`2.5e0 -> 2`,
TASK-70) — so the two casts need different rounding modes. Parsing is
exact decimal, not an f64 round-trip: `'9223372036854775807.4'` serves
INT64_MAX without overflowing, and `'1.4999999999999999999'` serves 1.
Scientific notation, leading `+`, and a bare `.5` all parse.

And our refusal text is itself wrong about the oracle — it says "DuckDB
errors at plan time; TRY_CAST is the NULL-yielding spelling", while DuckDB
serves 2 and `TRY_CAST('1.5' AS BIGINT)` is also 2. TASK-113 fixes that
message first, as its own commit, independent of the cast work.

### 4. String-builder cap — 1 seed — CLOSED

**Re-measured 2026-08-16: matches.** TASK-88 is Done; the row was stale.

19788: `rtrim(repeat('a', c0))` under a filtering LIKE — we trap "string
builder result exceeds 1 GiB", DuckDB serves the pad. Exactly TASK-88
(To Do), no new information.

### 5. DECIMAL literal through DOUBLE, 1-ulp repr — 4 seeds

982 (`76.853`), 10062 (`-47.907`), 10783 (`8.793`), 18913 (`65.195`):
DuckDB types the literal DECIMAL and its DOUBLE conversion lands 1 ulp off
our direct double parse. No new information about the divergence itself.

**Corrected 2026-08-15 — this class is NOT pinned.** The first draft said
"xfail pins in test_decimals.py; TASK-91 is the statics member", and both
halves were wrong. Phase 5 in the lattice spec is `Dec(p,s)` arithmetic;
TASK-91 is phase **1** (decimal STATIC ingest), a different member, and its
pin is the only xfail in test_decimals.py. Checking all six strict-xfail
pins in the suite: none covers the bare-literal 1-ulp class. The one other
DECIMAL pin (test_integer_widths.py, TASK-103) is the foldable-NULL
collapse.

What actually holds this class is `packages/confit/docs/known-limitations.md` — prose, and
TASK-95 (doc-twin totality) is still To Do, so nothing rings when phase 5
lands. The `decimals` campaign tag in fuzz/oracle.py claimed a strict-xfail
twin for the same reason; corrected there too.

### 6. Negative zero — 3 seeds — CLOSED

**Re-measured 2026-08-16: all three match.** TASK-80 was already Done when
this triage was written (the fix, 2b7866f, predates the campaign base) and
the row below was stale, not the engine.

1675 (`* REPLACE (-0.0 AS c0)`: we keep -0.0, DuckDB serves 0.0),
8168 (`ceil(-0.25)`: we -0.0, DuckDB 0.0), 15634 (`-0.0 / -67.764e0`: we
0.0, DuckDB -0.0). TASK-80's exact family (To Do) — note the sign goes
BOTH directions depending on the operation.

### 7. UDF bind-fold composition gap — 1 seed

8352: `(udf0(NULL, upper('0'), CAST(0.1e0 AS VARCHAR))).f0` — DuckDB folds
the call (args constant AFTER folding `upper`/`CAST`), whole-call None ->
SQLNULL -> INTEGER; we type by declaration (int64) because our bind fold
requires literal args. This is the TASK-103 family-2 composition gap,
already pinned xfail-strict in test_integer_widths.py
(test_bind_fold_composition_gaps). The original TASK-101 bare-literal-args
class is gone from the campaign — its AC held.

### 8. Harness, not engine: nondeterministic static-only SQL — 2 seeds

4038, 8228: `SELECT c1 AS g, max(c0) AS v FROM s0 GROUP BY c1 FETCH FIRST
1 ROWS ONLY` — FETCH without ORDER BY is nondeterministic; the constant
emitter's DuckDB run and the oracle's DuckDB run lawfully pick different
groups. Fuzzer QoL for the TASK-94 rework: either generate an ORDER BY
with every FETCH or teach the classifier that both answers are lawful.

### 9. Timeouts — 4 seeds — the ORACLE is slow, not the engine

Investigated 2026-08-14 (the runner loses the SQL on a timeout, so the
case was recovered from the generator by seed). Seed 4395 is:

```sql
SELECT (lpad(c1, 2147483647, 'NULL') LIKE 'O''Brien') AS o FROM __THIS__
```

Measured both sides: **we refuse in 0.00s at bind** ("lpad count
2147483647 exceeds the 1 GiB string-builder budget"); **DuckDB takes 9.0s**
to actually build the 2 GiB pad and answer `false`. Under 8 workers that
comfortably exceeds the per-case timeout, which is the whole finding — no
engine hang, and no liveness bug. 7422, 12229, 13269 are the same story
without the reproduction.

Two follow-ups, both fuzzer QoL (fold into TASK-94):
- record the SQL BEFORE executing, so a timeout is diagnosable in place
  rather than by seed archaeology;
- either bound generated pad/repeat lengths or classify an oracle-side
  timeout apart from an engine-side one — they mean opposite things.

Note the underlying VALUE divergence here is real but already known: we
refuse where DuckDB serves a giant pad (the TASK-88 family, §4). Refusing
at bind rather than at run is the better half of that behaviour.

## Scoreboard

| family | seeds | status |
|---|---|---|
| duck-traps-we-serve | 4 | TASK-99 / TASK-84 (To Do) |
| we-trap-duck-serves | 3 | TASK-85 **Done**; 1 residual -> TASK-117 |
| VARCHAR->int cast rounding | 2 | TASK-113 (To Do) |
| string-builder cap | 1 | TASK-88 **Done**, re-verified 2026-08-16 |
| decimal-literal 1 ulp | 4 | phase-5 family, documented only — NOT pinned |
| negative zero | 3 | TASK-80 **Done**, re-verified 2026-08-16 |
| fold composition | 1 | TASK-103 family, pinned |
| harness nondeterminism | 2 | fuzzer QoL (TASK-94) |
| timeouts | 4 | oracle-slow, not engine — fuzzer QoL (TASK-94) |

Every family is now mapped to a ticket. §3 was the last unticketed one;
TASK-113 closes the gap.
