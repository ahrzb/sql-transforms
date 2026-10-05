# The native-transform loop

Grows `sql_transform.native`, the catalog that turns fitted sklearn
transformers into confit functions, one family per cycle, until every
in-scope transformer in [coverage.md](coverage.md) is native.

- [goal.md](goal.md): what a native entry must equal, and the scope.
- [PLANS.md](PLANS.md): the working list, highest value first.
- [decisions/](decisions/): questions for the owner, and rulings in force.
- [coverage.md](coverage.md): the scoreboard (generated).
- [subagents.md](subagents.md): running the loop with worker sessions.
- [reports.md](reports.md): the report written after each milestone.

## Running it

Start a session on this repository and run:

> `/loop` Work the native-transform loop: follow
> packages/sql-transform/docs/native/README.md.

Each iteration is one cycle below. The loop stops when PLANS has nothing it
can do without the owner.

The loop runs inline (one session, one family per cycle) or, when the owner
asks for parallelism, subagent-driven: a supervisor gives one family per
worker session and reviews and merges their PRs. See
[subagents.md](subagents.md).

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
   (`uv run python -m sql_transform.native.coverage --write`), `git add`
   the new files (pre-commit skips untracked ones), then
   `uv run pre-commit run --all-files` and `uv run python scripts/gate.py`.
   Open a PR, wait for CI, squash-merge with the expected head SHA.
8. **Update PLANS.** Remove the item; add what the cycle found (a gap in
   confit, a configuration left as `NotNative`, a follow-up).
9. **Milestone?** When this cycle completes a milestone (a batch of
   families, a scoreboard mark, a confit capability landing, or the loop
   stopping), write the report described in [reports.md](reports.md).

## When confit is missing something

The catalog is sql_transform's; the machinery is confit's, and confit has
its own loop (`packages/confit/PLANS.md`), which builds what this one needs
first. So this loop does not change confit: it records the need under PLANS
"Needs from confit" (what, and which entries wait on it) and moves to the
next item. An entry that waits on confit stays `NotNative` with a reason
naming the capability.

## Rules

- The twin is the reference: `PythonTransform` calling the fitted
  estimator. Never change it to make an entry pass.
- One family per PR; a family is one module.
- Register exact classes only. A subclass may override `transform`.
- Stop and write a record in `decisions/open/` only for a question of
  contract or scope (goal.md). Implementation choices are the loop's.
