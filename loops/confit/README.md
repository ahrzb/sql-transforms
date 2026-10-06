# The confit loop

This loop turns [PLANS.md](PLANS.md) into merged PRs on `packages/confit`, one verified change at a time. It
implements open work, checks the work against the DuckDB oracle, merges it, and moves to the next item. It
also builds what the native catalog loop needs from confit. It builds these items ahead of its own query
classes.

- [goal.md](goal.md) holds the goal and the contract that judges the work.
- [PLANS.md](PLANS.md) lists open work, with the most valuable item first.
- [tickets.md](tickets.md) is the board. It shows what is in progress and who has each ticket.
- [worker-brief.md](worker-brief.md) is this loop's part of a worker's prompt.
- [decisions/](decisions) holds questions for the owner and the rulings in force.
- [reports/](reports) holds goal readings and short readings.

Both loops share some rules. [../README.md](../README.md) states them.

- [../workers.md](../workers.md) covers modes, tickets, the board, launching, review and merge.
- [../worker-brief.md](../worker-brief.md) is the shared part of a worker's prompt.
- [../reporting.md](../reporting.md) says when to report and defines the KPI block.

The product output of this loop stays in the package. It is in `packages/confit/docs/`. That folder holds the
oracle contract, the specs, the known limitations and the technical reports. The corpus counts are in
`packages/confit/docs/reports/corpus-counts.json`.

The agent that reads this file is as capable as the agent that wrote it. The sections below give working
rules and their reasons. If a rule and the situation disagree, follow the reason. Then record the decision in
the PR.

## Running it

To start the loop, run `/loop Work the confit loop: follow loops/confit/README.md.`

The loop runs in one of two modes. In inline mode, the loop session does each ticket itself. In
worker-driven mode, the loop session starts one worker session for each ticket
([../workers.md](../workers.md) §1). The owner picks the mode.

In both modes, the work is done when all of these are true:

1. The gate is green.
2. Lint is clean.
3. A campaign shows no gated class.
4. PLANS and the docs are updated.
5. The PR is squash-merged.

## One cycle

1. **Sync.** Fetch `master`. Start a branch from it, named `claude/confit-<name>`.
2. **Pick.** Take the top item from the ticket sources below. Put it on the board. Do this also for inline
   work.
3. **Measure DuckDB first** with `confit.oracle.Oracle`, optimizer off. Model what you measured. Cite the
   measurement in comments and tests. If confit cannot reproduce a behavior exactly, confit refuses it by
   name.
4. **Implement** the change with parity tests (`fuzz.parity.assert_parity`). Never use a raw DuckDB connection
   in a test.
5. **Check.** Run these three checks:
   - the gate (`uv run --no-sync python scripts/gate.py`);
   - lint. Run `git add -A`, then `uv run --no-sync pre-commit run --all-files`. Read the whole output;
   - a campaign of 10 000 to 20 000 seeds on a release build
     (`cd packages/confit && uv run --no-sync python -m fuzz.runner --seed S --n N --workers 4`).

   If Rust sources are newer than the debug extension, the native guard in the tests rebuilds a debug
   extension. Before you time anything or run a campaign, reinstall the release build with
   `uv sync --locked --group spark --reinstall-package confit`.
6. **Ship.** Open the PR and wait for CI. Squash-merge it with the expected head SHA. Then reset the branch
   onto master.
7. **Update** PLANS. If a limitation changes, also update `packages/confit/docs/known-limitations.md`.

## Standing duties (both modes)

- **Nightly triage.** Read the latest report on the open issue titled "Nightly campaign findings". Then re-run each
  gated seed on master. Fix each live class. Alternatively, pin it as an expected failure that must keep
  failing (xfail-strict) in `packages/confit/tests/test_open_divergences.py`. Re-run the metamorphic seeds
  too.
- **The native catalog loop.** Reproduce each request that it sends. Then fix the request inline or make a
  ticket for it. When a fix lands, tell that session. If you cannot reach that session, it reads the
  "Delivered" list in PLANS.
- **Reports.** When a trigger in [../reporting.md](../reporting.md) fires, write a report in
  [reports/](reports). Open the report with the KPI block from that page. End it with what needs the owner.
  Title the PR `report: ...`. Do not write a report for each PR.
- **Owner questions.** If a question concerns an OPT_EMULATED class, an exclusion or an open decision, write
  it in `decisions/open/` with the evidence. Do not settle it in code. (OPT_EMULATED is the verdict for a case
  where confit matches the optimizer-on reading and the optimizer-off reading differs.)
- **Hygiene.** Commit and push before you end a turn, because the stop hook checks it. Never put model names
  in repo artifacts. Use the trailers in [../worker-brief.md](../worker-brief.md). Never run
  `pkill -f python`, because it kills your own shell.

## Tickets and review

[../workers.md](../workers.md) gives the scheme. This section gives what is specific to confit.

**Where tickets come from,** in this order:

1. **Correctness first.** These are bugs and live divergences from a campaign or the nightly run (gated
   classes in issue #305). They are usually small, so inline mode suits them.
2. **Needs from the native catalog loop.** These are the items in its PLANS section "Needs from confit" and
   its messages. They unblock another loop, so they come before the query classes. Reproduce each need before
   you make a ticket for it.
3. **The ruled query classes,** in the ruled order
   ([decisions/closed/next-query-classes.md](decisions/closed/next-query-classes.md)). A class usually has
   several tickets. Each ticket is a slice that ships something served and tested. For example, the i128
   class (128-bit integers) has these slices:
   - unsigned widths on the i64 lane (#341);
   - UBIGINT and HUGEINT on a 128-bit lane (#349);
   - exact wide aggregates, later.
4. **Performance and engineering items** from PLANS.

**Every confit ticket also states** what to measure in DuckDB first and what must refuse by name.

**Hot files.** Tickets collide most in these files, all under `packages/confit/src/specializer/`:
`frontend/expr.rs`, `frontend/typing.rs`, `frontend/functions.rs`, `frontend/calls.rs`, `plan.rs` (the
functions `can_trap` and `trap_skeleton`), `lower.rs` and `exec/cranelift.rs`.

Two tickets in one hot file can run together if they touch different functions. In that case, note the
overlap on the board and merge the smaller ticket first.

A new `Lit`, `Ty` or `SKind` variant breaks every exhaustive match on another open branch. If you add one,
say so on the board. After the merge, check each open branch against master.

**Review checklist,** after scope:

- **Oracle discipline.** The change measured DuckDB's behavior and did not assume it. The tests are parity
  tests. Confit refuses by name each behavior that it does not reproduce exactly.
- **Trap semantics.** A change can move, share, drop or reorder a computation. If it does, every row that
  trapped before must still trap with the same class of error. No new trap may appear.
- **Evidence.** For performance work, give before and after numbers. Give the gate result. Give a campaign
  of at least 10 000 seeds with no gated class. For generator changes, also give a seed-gated run counted by
  verdict. Re-run anything that looks wrong, because CI only runs the gate.
- **On master, not on the branch's base.** Before you merge, merge master into the PR locally. Build, and run
  the gate there. A branch that forked before another merge can conflict, fail to compile, or undo that
  merge. All three happened in the first wave of tickets (see [tickets.md](tickets.md)).
- **Generator hygiene.** Generator changes are seed-gated and use their own random number generator (RNG), so
  no other seed changes.
- **Docs.** PLANS and `known-limitations.md` are updated. Tests pin the new boundary.

**After a merge:** If the merge serves a catalog need, say so in the PLANS "Delivered" list. Also tell the
native loop's session. Then update the board.
