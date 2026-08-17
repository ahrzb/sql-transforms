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


---

## Addendum, same day: the oracle changed, and so did the engine

`PRAGMA disable_optimizer` DuckDB became the oracle, the engine stopped
reproducing plan rewrites, and the campaign was re-run at the same 4000 seeds.

| | before the sweep | after the sweep | after the oracle change |
|---|---|---|---|
| raw findings | 26 | 25 | 28 |
| confit traps, DuckDB serves | 2 | 0 | 0 |
| DIVERGE_OPT (we match the oracle, opt-on differs) | n/a | 2 | 8 |
| OPT_EMULATED (a pass we still reproduce) | n/a | 6 | 0 |

The headline is the last row. Six optimizer emulations were being exercised by
the campaign; after conforming the engine there are none. The single remaining
`OPT_EMULATED` line is seed 1784's `FETCH FIRST 1 ROWS ONLY` without an
`ORDER BY`, where the two DuckDB reads lawfully pick different groups -- the
harness nondeterminism §8 already records, mislabelled by this category
because the readings disagree with each other.

`DIVERGE_OPT` rising from 2 to 8 is not a regression. It is the same
behaviour, now named: these are the cases where we match the oracle and a user
running DuckDB with the optimizer on would see something else. That price was
always being paid; before the oracle change it was invisible, and before the
conformance work it was partly hidden behind emulations.

Seeds: 312, 812, 1196, 1563, 1564. Two engine bugs found ALONGSIDE this, both
by the oracle rather than by reasoning, and both fixed:

* `substr` with a negative start clamped before applying the length window.
* a filter did not short-circuit on a NULL conjunct, and AND/OR
  short-circuited in a PROJECTION where DuckDB evaluates both operands (that
  second half diverged from BOTH readings, so it was never an oracle artifact).

Still open and unchanged: the 16-seed ambiguous-reference class (57% of
findings now), seed 2668's `-2147483648 / -1` overflow where we serve and both
readings error, and the four `values` seeds.
