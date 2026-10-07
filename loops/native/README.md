# The native-transform loop

This loop grows the native catalog (`sql_transform.native`). The catalog
turns fitted sklearn transformers into functions that confit serves. The
loop adds one family in each cycle. It continues until every transformer in
scope in [coverage.md](../../packages/sql-transform/spec/native/coverage.md) is native.

- [goal.md](goal.md) says what done means. The [catalog contract](../../packages/sql-transform/spec/native/catalog-contract.md) says what a catalog entry must equal, and the scope.
- [PLANS.md](PLANS.md) lists open work, with the most valuable item first.
- [tickets.md](tickets.md) is the board. It shows what is in progress and
  who has each ticket.
- [worker-brief.md](worker-brief.md) is this loop's part of a worker's
  prompt.
- [decisions/](decisions/README.md) holds the questions for the owner and
  the rulings in force.
- [coverage.md](../../packages/sql-transform/spec/native/coverage.md) is the scoreboard. A script generates it.
- [report-format.md](report-format.md) gives the shape of a milestone
  report. [reports/](reports/) holds the reports.

Both loops share some rules. [../README.md](../README.md) states them.

- [../workers.md](../workers.md) covers modes, tickets, the board,
  launching, review and merge.
- [../worker-brief.md](../worker-brief.md) is the shared part of a worker's
  prompt.
- [../reporting.md](../reporting.md) says when to report and defines the
  KPI block.
- [../../AGENTS.md](../../AGENTS.md) gives the rules for text that the
  owner reads, the glossary, and questions for the owner.

The agent that reads this file is as capable as the agent that wrote it.
The sections below give working rules and their reasons. If a rule and the
situation disagree, follow the reason. Then record the decision in the PR.

## Running it

To start the loop, run `/loop Work the native-transform loop: follow
loops/native/README.md.`

Each iteration is one cycle below. The loop stops when all the work in
PLANS needs the owner.

The loop runs in one of two modes. In inline mode, the loop session does
each family itself. In worker-driven mode, the loop session is a supervisor.
It gives one family to each worker session, and it reviews and merges their
PRs ([../workers.md](../workers.md)). The owner picks one of the two
modes. "Tickets and review" below gives what is specific to this loop.

## One cycle

1. **Sync.** Fetch `master`. Start a branch from it, named
   `claude/native-<name>`. The confit loop works on its own branches.
2. **Pick.** Take the top item of PLANS "Next". If "Next" is empty, take
   the easiest "not yet" row of coverage.md, and record it in PLANS first.
3. **Read the twin.** Find the transformer's `transform` method in the
   installed sklearn (`.venv/lib/python3.*/site-packages/sklearn/`). Write
   down its exact sequence of operations and its dtype. Include what it
   does with NaN, and every constructor parameter that changes `transform`.
4. **Translate.** Write the translator in the family module (`scalers.py`,
   `impute.py`, `encode.py`, and so on). A new family gets a new module,
   which `__init__.py` imports. The translator maps the fitted attributes
   onto the features in the twin's order. It uses the SQL builder of confit
   (`confit.sql`) and the shared helpers of the catalog (`_helpers.py`). If
   the entry does not serve a configuration, it raises `NotNative` and
   names the configuration.
5. **Fixtures.** Add the class to `FIXTURES` in `catalog_test.py`, with one
   factory for each configuration that changes `transform`. If the family
   needs inputs that the row generator does not make, extend the generator
   (`_fit_matrix` and `_rows`). Do not add a special case for one entry.
6. **Bound.** Register the entry as bit-exact (`ulps=0`). If the entry
   cannot follow the twin's order, measure the largest distance in ulps
   (units in the last place) over at least 200 seeds. Register that ulp
   bound, and cite the measurement in the translator's docstring: the
   seeds, the maximum and the date. Never raise a bound to make a failure
   pass. First find the twin's order, and follow it.
7. **Done.** Do these steps:
   1. Regenerate coverage with
      `uv run --no-sync python -m sql_transform.native.coverage --write`.
   2. Run `git add` on the new files, because pre-commit skips untracked
      files.
   3. Run `uv run --no-sync pre-commit run --all-files`.
   4. Run the gate: `uv run --no-sync python scripts/gate.py`.
   5. Open a PR and wait for CI. Squash-merge it with the expected head SHA.
8. **Update PLANS.** Remove the item. Add what the cycle found: a need from
   confit, a configuration that stays `NotNative`, or a follow-up.
9. **Report.** If a trigger in [../reporting.md](../reporting.md) happens,
   write the report. [report-format.md](report-format.md) gives its shape.

## When confit does not have a capability

The catalog belongs to sql-transform, and the machinery belongs to confit.
Confit has its own loop. That loop builds the needs of this loop before its
own work ([../README.md](../README.md), "Between the loops").

So this loop does not change confit. It sends each need to the confit loop's
session, with a reproduction. It records the need under PLANS "Needs from
confit", with the entries that wait on it. Then it moves to the next item.
An entry that waits on confit stays `NotNative`, with a reason that names
the capability.

## Rules

- The twin is the reference. Never change the twin to make an entry pass.
- One PR holds one family, and one module holds one family.
- Register exact classes only, because a subclass may override `transform`.
- Write a decision record in `decisions/open/` only for a question about
  the contract or the scope in the
  [catalog contract](../../packages/sql-transform/spec/native/catalog-contract.md).
  The loop makes the implementation choices.
- Text that the owner reads follows the `simple-english` skill and uses the
  terms in [GLOSSARY.md](../../GLOSSARY.md). This is a report, a decision
  record, this README, goal.md, report-format.md or the native pages of the
  package spec.

## Tickets and review

[../workers.md](../workers.md) gives the scheme. This section gives what is
specific to the native loop.

**A ticket is one family.** It has one module and one PR. It covers the
cycle above, from "Read the twin" to "Done". The families live in separate
modules, so their tickets are almost independent.

**Where tickets come from,** in this order:

1. PLANS "Next".
2. The easiest "not yet" rows of [coverage.md](../../packages/sql-transform/spec/native/coverage.md).
3. The configurations that a merged family left `NotNative`, after confit
   delivers what they wait on.

A family that waits on a confit capability is not a ticket yet. It goes
under PLANS "Needs from confit".

**Every native ticket also names these items:**

- the transformer classes and their sklearn module;
- the configurations to cover;
- the known traps: dtype or NaN handling, a kernel that depends on the
  platform, or an order of reduction with an ulp bound;
- each confit capability that the family needs, merged or not.

**Files that every ticket changes,** and how to resolve their conflicts:

- `catalog_test.py` (`FIXTURES`): each ticket adds its rows in its own
  block. A conflict there is mechanical. Keep both sides.
- `__init__.py`: a new family adds one import line.
- [coverage.md](../../packages/sql-transform/spec/native/coverage.md): a script generates it. Never resolve a
  conflict in it by hand. Regenerate it after you merge master.
- [PLANS.md](PLANS.md): a worker changes only its own family's item, and
  adds to "Needs from confit".
- `_registry.py`, `_helpers.py` and `_check.py` are shared machinery. If a
  ticket must change them, the ticket says so first. If two such tickets
  would run at the same time, the change to the machinery is its own ticket
  and goes first.

**The review checklist.** It adds to [../workers.md](../workers.md) §5.
Give each review on the PR and also to the worker directly. Mark each item
blocking or optional.

- [ ] The translator follows the twin's order of operations, and the PR
      states that order. Check one configuration against the source of the
      installed sklearn, and of scipy or numpy where the twin calls them.
- [ ] Each configuration that changes `transform` has a fixture, or the
      entry raises `NotNative` and names it.
- [ ] Each bound is 0, or the PR measured it over at least 200 seeds and
      cites the measurement. A bound that is not small is a decision record
      in `decisions/open/`, not code.
- [ ] Nothing changed outside the family module, its fixtures,
      `__init__.py`, PLANS and coverage, unless the ticket said so.
- [ ] The gate is green. The PR regenerated [coverage.md](../../packages/sql-transform/spec/native/coverage.md) and
      did not edit it by hand.
- [ ] Each new item under "Needs from confit" has a reproduction.

**After each merge,** do these steps:

1. Merge master into the other open branches, and regenerate
   [coverage.md](../../packages/sql-transform/spec/native/coverage.md) in each.
2. Update the board ([tickets.md](tickets.md)).
3. Send each new need to the confit loop's session.
4. At the end of a wave, write the milestone report yourself. A worker never
   writes one. Measure every number again on master.
