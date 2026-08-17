# Fuzz triage — 2026-08-17 campaign (small-divergence sweep gate)

Campaign: `python -m fuzz.runner --seed 0 --n 4000 --workers 8`, run twice —
once against master before the sweep, once after TASK-115/116/117/118/119
landed. It is the gate for the sweep, not a discovery run.

**Headline: zero `confit traps, DuckDB serves` findings.** That is the one
outcome the match-or-refuse contract has no room for, and after the sweep the
class is empty.

## Before vs after

| | before | after |
|---|---|---|
| raw findings | 26 | 25 |
| classes | 6 | 5 |
| confit traps, DuckDB serves | **2** | **0** |
| DuckDB errors, confit serves | 3 | 4 |
| DIVERGE_BUILD (ambiguous reference) | 16 | 16 |
| DIVERGE_VALUE | 5 | 5 |

The two `confit traps` findings did not exist before TASK-118 either: they
were introduced BY the narrow range trap and closed the same day by the
`IS [NOT] NULL` elision, in the same PR. They are listed here because the
campaign is what caught them.

- **seed 1564** → `SELECT -13 FROM __THIS__ WHERE ((c0 * 32) IS NOT NULL)`,
  `c0` int8 = -128.
- **seed 2174** → `SELECT c1 FROM __THIS__ WHERE ((- c0) IS NOT NULL)`,
  `c0` int16 = -32768.

Both shrank to the same shape: a narrow arithmetic overflow under
`IS [NOT] NULL`, which DuckDB answers from the operands' nullness without
evaluating.

## The one new finding, and it is deliberate

**seed 1111** — `WHERE ((9223372036854775807 * c2) IS NOT NULL)`. DuckDB
errors, we serve.

This is the elision above, and the divergence is chosen rather than missed.
Measured while fixing it: DuckDB's elision is driven by the column's null
STATISTICS, so the same query over the same schema with the same overflowing
row serves for `[-128, 7]` and traps for `[-128, NULL]`. Its answer is not a
function of the query at all. A compile-once artifact cannot hold that, and
deciding per batch would make a row's answer depend on its neighbours in the
batch — worse than the divergence, for a serving engine.

The proof and both sides are pinned in
`packages/confit/tests/known_divergences/test_trap_elision.py`.

## The classes that did not move

Unchanged in both runs, all pre-existing and none touched by the sweep:

- **16 × DIVERGE_BUILD, "Ambiguous reference to column"** — the fuzzer
  generates a join whose key name binds on both sides; DuckDB refuses, we
  build. One class, one cause. Not yet ticketed.
- **seed 312, 2805** — `ln(0.0)` / `sqrt(sqrt(-x))` inside `nullif` folded at
  plan time on DuckDB. The §1 family of the 2026-08-13 triage (TASK-99 /
  TASK-84), unchanged.
- **seed 2668** — `Overflow in division of -2147483648 / -1` reached through
  a folded literal. Same family.
- **5 × DIVERGE_VALUE** — including seed 1784's `FETCH FIRST 1 ROWS ONLY`
  without `ORDER BY`, which is the harness nondeterminism §8 of the
  2026-08-13 triage already recorded as fuzzer QoL for TASK-94.

## What to do next with it

The 16-seed ambiguous-reference class is now the largest single thing the
fuzzer sees and it has no ticket. It is a `DIVERGE_BUILD` (we build what
DuckDB refuses), so it is the mild direction, but it is 64% of the findings.
