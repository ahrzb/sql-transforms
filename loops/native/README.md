# The native-transform loop

Grows `sql_transform.native`, the catalog that turns fitted sklearn
transformers into confit functions, one family per cycle, until every
in-scope transformer in [coverage.md](coverage.md) is native.

- [goal.md](goal.md): what a native entry must equal, and the scope.
- [PLANS.md](PLANS.md): the working list, highest value first.
- [tickets.md](tickets.md): the board, what is in flight and who has it.
- [worker-brief.md](worker-brief.md): this loop's part of a worker's prompt.
- [decisions/](decisions/): questions for the owner, and rulings in force.
- [coverage.md](coverage.md): the scoreboard (generated).
- [report-format.md](report-format.md): the shape of a milestone report;
  [reports/](reports/) holds them.

The rules both loops share are in [../README.md](../README.md):
[../workers.md](../workers.md) (modes, tickets, the board, launching,
review, merge), [../worker-brief.md](../worker-brief.md) and
[../reporting.md](../reporting.md) (when to report, the KPI block).

## Running it

Start a session on this repository and run:

> `/loop` Work the native-transform loop: follow loops/native/README.md.

Each iteration is one cycle below. The loop stops when PLANS has nothing it
can do without the owner.

The loop runs inline (one session, one family per cycle) or, when the owner
asks for parallelism, worker-driven: a supervisor gives one family per
worker session and reviews and merges their PRs
([../workers.md](../workers.md); what is native-specific is under
"Tickets and review" below).

## One cycle

1. **Sync.** Fetch `master` and start the working branch from it. Use a
   branch of this loop's own; another loop works confit on its own branch.
2. **Pick.** The top item of PLANS "Next". When "Next" is empty, the
   easiest "not yet" row of coverage.md; record it in PLANS first.
3. **Read the twin.** Find the transformer's `transform` in the installed
   sklearn (`.venv/lib/python3.*/site-packages/sklearn/`) and write down its
   exact operation sequence and dtype, including what it does with NaN, and
   every constructor parameter that changes `transform`.
4. **Translate.** In the family module (`scalers.py`, `impute.py`,
   `encode.py`, `linear.py`, ...; a new family gets a new module imported by
   `__init__.py`), write the translator from the fitted attributes onto the
   features, with `confit.sql` and `_helpers.py`, in the twin's order.
   A configuration the entry does not serve raises `NotNative` naming it.
5. **Fixtures.** Add the class to `FIXTURES` in `catalog_test.py`, with one
   factory per configuration that changes `transform`. If the family needs
   inputs the generic generator does not make (strings, categories, a
   narrower domain), extend `_fit_matrix` / `_rows` rather than special-
   casing the entry.
6. **Bound.** Register bit-exact (`ulps=0`). If the twin's order cannot be
   reproduced (BLAS sums, NumPy pairwise reductions), measure the largest
   ulp distance over at least 200 seeds, register that, and cite the
   measurement (seeds, max, date) in the translator's docstring. Never
   raise a bound to make a failure pass: find the order first.
7. **Done.** Regenerate coverage
   (`uv run --no-sync python -m sql_transform.native.coverage --write`),
   `git add` the new files (pre-commit skips untracked ones), then
   `uv run --no-sync pre-commit run --all-files` and
   `uv run --no-sync python scripts/gate.py` (`--no-sync` keeps the
   environment the install step built, as the shared brief says). Open a
   PR, wait for CI, squash-merge with the expected head SHA.
8. **Update PLANS.** Remove the item; add what the cycle found (a gap in
   confit, a configuration left as `NotNative`, a follow-up).
9. **Milestone?** When a trigger in [../reporting.md](../reporting.md)
   fires (a batch of families, a scoreboard mark, a confit capability
   landing, a regression, the loop stopping, or 24 h or 15 merged PRs
   without a report), write the report: [report-format.md](report-format.md)
   gives its shape, `reports/` holds it.

## When confit is missing something

The catalog is sql_transform's; the machinery is confit's, and confit has
its own loop ([../README.md](../README.md), "Between the loops"), which
builds what this one needs first. So this loop does not change confit: it
sends the need to the confit loop's session with a reproduction, records it
under PLANS "Needs from confit" (what, and which entries wait on it), and
moves to the next item. An entry that waits on confit stays `NotNative`
with a reason naming the capability.

## Rules

- The twin is the reference: `PythonTransform` calling the fitted
  estimator. Never change it to make an entry pass.
- One family per PR; a family is one module.
- Register exact classes only. A subclass may override `transform`.
- Stop and write a record in `decisions/open/` only for a question of
  contract or scope (goal.md). Implementation choices are the loop's.

## Tickets and review

The scheme is [../workers.md](../workers.md). What is native-specific:

**A ticket is one family:** one module, one PR, covering the cycle above
from "Read the twin" through "Done". Families live in separate modules, so
tickets are nearly independent by construction.

**Where tickets come from,** in order: PLANS "Next"; the easiest "not yet"
rows of [coverage.md](coverage.md); configurations a merged family left
`NotNative`, once confit has delivered what they waited on. A family that
waits on a confit capability is not a ticket yet: it goes under PLANS
"Needs from confit".

**Every native ticket also names:** the transformer classes and their
sklearn module; the configurations to cover; known traps (dtype or NaN
handling, a platform-dependent kernel, a reduction order that needs an ulp
bound); and any confit capability the family needs, merged or not yet.

**Files every ticket shares,** and how their conflicts resolve:
- `catalog_test.py` (`FIXTURES`): each ticket appends its rows in its own
  block; conflicts there are mechanical, keep both sides.
- `__init__.py`: a new family adds one import line.
- [coverage.md](coverage.md): generated. Never resolve a conflict in it by
  hand; regenerate it after merging master
  (`uv run --no-sync python -m sql_transform.native.coverage --write`).
- [PLANS.md](PLANS.md): a worker edits only its own family's item and
  appends to "Needs from confit".
- `_registry.py`, `_helpers.py`, `_check.py`: shared machinery. A ticket
  that has to change them says so up front; when two such tickets would run
  at once, the machinery change goes first as its own ticket.

**Review checklist** (on top of [../workers.md](../workers.md) §5; deliver
each review on the PR and to the worker directly, marked blocking or
optional):
- [ ] The translator follows the twin's operation order, which the PR
      states. Spot-check one configuration against the installed sklearn
      (and scipy or numpy where the twin calls them) source.
- [ ] Every configuration that changes `transform` is either fixtured or
      refused with `NotNative` naming it.
- [ ] Bounds are 0, or measured over at least 200 seeds and cited; a bound
      that is not small is a record in `decisions/open/`, not code.
- [ ] Nothing outside the family module, its fixtures, `__init__.py`,
      PLANS and coverage changed, or the change was announced in the
      ticket.
- [ ] The gate is green; [coverage.md](coverage.md) was regenerated, not
      hand-edited.
- [ ] Every new "Needs from confit" entry has a reproduction.

**After each merge:** merge master into the other open branches and
regenerate [coverage.md](coverage.md) in each; update the board
([tickets.md](tickets.md)); send any new need to the confit loop's session;
and, at the end of a wave, write the milestone report yourself (a worker
never writes one; every number is re-measured on master).
