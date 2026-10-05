# The confit loop

Turns [PLANS.md](PLANS.md) into merged PRs on `packages/confit`, one verified
change at a time: implement open work, check it against the DuckDB oracle,
merge it, move on. It also builds what the native catalog loop needs from
confit, ahead of its own query classes.

- [goal.md](goal.md): the goal and the contract that judges the work.
- [PLANS.md](PLANS.md): open work, highest value first.
- [tickets.md](tickets.md): the board, what is in flight and who has it.
- [worker-brief.md](worker-brief.md): this loop's part of a worker's prompt.
- [decisions/](decisions): questions for the owner, and rulings in force.
- [reports/](reports): goal readings and short readings.

The rules both loops share are in [../README.md](../README.md):
[../workers.md](../workers.md) (modes, tickets, the board, launching,
review, merge), [../worker-brief.md](../worker-brief.md) and
[../reporting.md](../reporting.md) (when to report, the KPI block). What the
loop produces for the product stays in the package:
`packages/confit/docs/` (the oracle contract, specs, known limitations,
technical reports) and `packages/confit/docs/reports/corpus-counts.json`.

The agent reading this is assumed to be as capable as the one that wrote it.
These are working rules and their reasons. When a rule and the situation
disagree, follow the reason and record the decision in the PR.

## Running it

Start the loop with `/loop Work the confit loop: follow loops/confit/README.md.`
It runs inline or worker-driven ([../workers.md](../workers.md) §1); the
owner picks. In either mode the **definition of done** is: gate green, lint
clean, a campaign with no gated class, PLANS and the docs updated, the PR
squash-merged.

## One cycle

1. **Sync.** Fetch `master` and start a branch from it
   (`claude/confit-<name>`).
2. **Pick.** The top item by the ticket sources below; put it on the board,
   inline work included.
3. **Measure DuckDB first** (`confit.oracle.Oracle`, optimizer off). Model
   what you measured, cite it in comments and tests; anything confit cannot
   reproduce exactly refuses by name.
4. **Implement**, with parity tests (`fuzz.parity.assert_parity`; never a raw
   DuckDB connection in a test).
5. **Check.** Gate (`uv run --no-sync python scripts/gate.py`), lint (`git add
   -A`, then `uv run --no-sync pre-commit run --all-files`, reading the whole
   output), and a 10–20k-seed campaign on a release build
   (`cd packages/confit && uv run --no-sync python -m fuzz.runner --seed S --n N
   --workers 4`). The tests' native guard rebuilds a debug extension when
   Rust sources are newer, so reinstall release
   (`uv sync --locked --group spark --reinstall-package confit`) before
   timing or campaigning.
6. **Ship.** Open the PR, wait for CI, squash-merge with the expected head
   SHA, reset the branch onto master.
7. **Update** PLANS and, when a limitation changes,
   `packages/confit/docs/known-limitations.md`.

## Standing duties (both modes)

- **Nightly triage.** Read the latest "Nightly campaign findings" comment
  (issue #305), then re-run each gated seed on master. Fix live classes, or
  pin them xfail-strict in `packages/confit/tests/test_open_divergences.py`.
  Re-run the metamorphic seeds too.
- **The native catalog loop.** Reproduce each request it sends, then fix it
  inline or ticket it. Tell that session when a fix lands; when you cannot
  reach it, the PLANS "Delivered" list is where it looks.
- **Reports.** Write one when a trigger in [../reporting.md](../reporting.md)
  fires, in [reports/](reports), opening with that page's KPI block and
  ending with what needs the owner. Title the PR `report: ...`. Never one per
  PR.
- **Owner questions.** OPT_EMULATED classes, exclusions and open decisions go
  to the owner with evidence in `decisions/open/`, never into code.
- **Hygiene.** Commit and push before ending a turn (the stop hook checks
  it); never put model names in repo artifacts; use the trailers in
  [../worker-brief.md](../worker-brief.md); never run `pkill -f python`,
  which kills your own shell.

## Tickets and review

The scheme is [../workers.md](../workers.md). What is confit-specific:

**Where tickets come from,** in order:

1. **Correctness first.** Bugs and live campaign or nightly divergences
   (gated classes in issue #305). Usually small; inline suits them.
2. **Needs from the native catalog loop.** Its PLANS "Needs from confit" and
   its messages. They unblock another loop, so they come ahead of the query
   classes. Reproduce each one before ticketing it.
3. **The ruled query classes,** in the ruled order
   ([decisions/closed/next-query-classes.md](decisions/closed/next-query-classes.md)).
   A class is usually several tickets, in slices that each ship something
   served and tested (the i128 class: unsigned widths on the i64 lane, #341;
   UBIGINT/HUGEINT on a 128-bit lane, #349; exact wide aggregates later).
4. **Performance and engineering items** from PLANS.

**Every confit ticket also states:** what to measure in DuckDB first, and
what must refuse by name.

**Hot files.** Tickets collide most in `frontend/expr.rs`,
`frontend/typing.rs`, `frontend/functions.rs`, `frontend/calls.rs`,
`plan.rs` (`can_trap`, `trap_skeleton`), `lower.rs` and `exec/cranelift.rs`
(all under `packages/confit/src/specializer/`). Two tickets in one hot file
can still run together when they touch different functions; note the
overlap on the board and merge the smaller one first. A new `Lit`, `Ty` or
`SKind` variant breaks every exhaustive match on another open branch: say so
on the board, and check each branch against master after the merge.

**Review checklist,** after scope:

- **Oracle discipline.** DuckDB's behaviour was measured, not assumed; tests
  are parity tests; anything not reproduced exactly refuses by name.
- **Trap semantics.** Where a change moves, shares, drops or reorders a
  computation, every row that trapped before still traps with the same class
  of error, and no new trap appears.
- **Evidence.** Before/after numbers for performance work, the gate, a
  campaign of at least 10k seeds with no gated class, and for generator
  changes a seed-gated run counted by verdict. Re-run anything that looks
  off; CI only runs the gate.
- **On master, not the branch's base.** Merge master into the PR locally,
  build, and run the gate there before merging: a branch that forked before
  another merge can conflict, fail to compile, or undo it (wave 1 hit all
  three).
- **Generator hygiene.** Changes are seed-gated with their own RNG, so no
  other seed moves.
- **Docs.** PLANS and `known-limitations.md` updated; tests pin the new
  boundary.

**After a merge:** when it serves a catalog need, say so in PLANS
"Delivered" and tell the native loop's session; update the board.
