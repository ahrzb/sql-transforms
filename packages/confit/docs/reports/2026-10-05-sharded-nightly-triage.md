# The first six sharded nightly runs, triaged (2026-10-05)

**What this is.** The nightly campaign has been red since it moved to four
shards of 100k seeds each (ahrzb/sql-transforms#310). This report covers the
six runs filed on issue ahrzb/sql-transforms#305 from 2026-09-29 to
2026-10-04: 2.4M seeds and 154 gated cases, one shrunk representative per
verdict class per night. All of it is real findings; none of it is runner
trouble.

---

## 1. Outcomes

**Seven defect classes are fixed or refused by name**, covering 108 of the
154 cases (attributed by each class's representative):

| class | cases | was | now |
|---|---:|---|---|
| UDF field over a closed constant argument | 60 | the field kept its declared type | evaluated at bind: a NULL result is DuckDB's SQLNULL (INTEGER) |
| static-only trapping JOIN ON conjunct, read through the key column | 12 | served; the conjunct counted as two-sided | refused by name, like every single-side trapping residual |
| closed NULL operand beside a trapping sibling | 11 | the sibling trapped | the NULL is found at bind and the sibling never runs |
| narrow-width constant overflow | 10 | folded in i64, no trap under `IN` or `repeat` | left to the runtime, which traps |
| DOUBLE join key | 7 | the key projected the probe's `0.0` | the static side's `-0.0`, carried on a shadow lane |
| bare shared name inside a JOIN ON key expression | 7 | bound to the row side | DuckDB's ambiguity error |
| WHERE `BETWEEN` beside a trapping conjunct | 1 | the upper bound short-circuited | checked last, as DuckDB's planner orders it |

**Two classes stay open.**
- **A join on an empty static table** (5 cases). DuckDB's answer is empty,
  but whether it evaluates the row side first depends on its pipeline shape.
  It traps for a projection directly under the join, and not when the key is
  computed in the join or the projection sits under a FILTER. confit always
  evaluates the row side. This is execution-order detail, so the disposition
  is an owner ruling: `docs/decisions/open/empty-static-join-trap-timing.md`,
  which proposes a named exclusion.
- **`nullif(NULL, x)`** (2 cases). DuckDB evaluates `x` and can trap on it.
  confit types the call as an adoptable NULL and drops `x`.

The rest were left alone on purpose. TIMEOUT (21 cases) and OPT_EMULATED
(18 cases) wait on the owner rulings already listed in PLANS (#305: seed
1159605, seed 1102717).

**Nothing was lost.** Campaign seeds 0..19999 against the master verdicts
(3fef800) on the same machine: one change, seed 12730 REFUSED → AGREE. No
agreement was lost, and the gate is green. The new classes are rare (about one
case per 100k seeds), which is why only the 400k-a-night campaign reached
them.

## 2. The rules, from DuckDB's source (v1.5.5)

- **The binder evaluates a foldable operand.** Before binding a call with
  default NULL handling, DuckDB evaluates each foldable argument
  (`TryEvaluateScalar`). A NULL replaces the call; a trap leaves it to run.
  This holds for any closed expression: `'0' LIKE 'a_c'`, `reverse('x')`,
  `CASE WHEN FALSE …`. confit's `fold` finishes only the spellings it was
  taught, so each new spelling was another nightly finding. confit now
  evaluates what `fold` leaves, using the interpreter over one row of no
  columns, so the fold and the runtime cannot disagree. Only a NULL result
  changes the tree.
- **BETWEEN is a conjunction, and the planner reorders WHERE conjuncts.**
  `bind_between_expression.cpp` binds a non-volatile `x BETWEEN l AND u` as
  `(x >= l) AND (x <= u)`. `LogicalFilter::SplitPredicates` runs while the plan
  is built, even with the optimizer off. It keeps the first child of each AND
  in place and appends the rest to the end of the filter list, so the upper
  bound is checked after every later conjunct. A BETWEEN outside a WHERE
  (under CASE, in a projection) keeps the written order.
- **A JOIN ON conjunct that names one side is a filter on that side.**
  DuckDB pushes `(s0.f - s0.v) < s0.k` below the join, where it runs over every
  row of `s0`. confit rebuilds the key column `s0.k` from the probe, so the
  conjunct looked two-sided. Sides are now classified by a second bind that
  keeps key reads on the static side.
- **DOUBLE keys compare `0.0 = -0.0`**, and the projected key is the build
  row's own value.

The first three are in `docs/oracle/02-inherited-quirks.md` with their pins.

## 3. Cost

The bind-time evaluation compiles a small program for each closed constant
that `fold` leaves unfinished and that can be NULL. Prepare time does not
move: a query with ten nested constant math calls prepares in about
6.6 ms on both this branch and master (p50 of 30), and a plain query in about
0.3 ms. That 6.6 ms is already on master and comes from elsewhere. Serving
latency is unaffected, since the evaluation happens at prepare time only.

## 4. Pins

- `tests/test_evaluation_order.py`: each rule on its own, against the oracle
  (19 cases).
- `tests/test_fuzz_smoke.py::test_the_sharded_nightly_findings_stay_fixed`:
  33 representative seeds.
- Rust unit tests updated where they hand-build DOUBLE-keyed statics (the
  shadow lane). The shift test now pins its in-range boundary at BIGINT:
  `1 << 62` at INTEGER is out of range on DuckDB, which the old test had
  pinned as a value.

## Environment and repro

- Linux x86-64, 4 cores, Python 3.14.7, DuckDB 1.5.5 oracle (optimizer off).
- A representative: `uv run python -m fuzz.shrink <seed>` in
  `packages/confit`.
- Campaign: `uv run python -m fuzz.runner --seed 0 --n 20000 --workers 4
  --cases cases.jsonl --baseline master_cases.jsonl`.
