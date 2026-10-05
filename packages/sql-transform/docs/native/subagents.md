# Running the native loop with workers

This loop can run in two modes. The owner picks one, usually by budget:

- **Inline.** One session works [README.md](README.md)'s cycle, one family
  per PR. This is the default, and the cheapest.
- **Subagent-driven.** A supervisor session splits the work into tickets,
  starts one worker session per ticket, reviews each worker's PR, and merges
  it.

Both modes share the same definition of done: README's cycle steps 6–8, with
the gate green, coverage regenerated, PLANS updated, and the PR
squash-merged.

The confit loop runs the same scheme. Its general policy is in
[`packages/confit/docs/loop/README.md`](../../../confit/docs/loop/README.md),
covering modes, ticket splitting, launching workers, review and merging, and
it applies here too. This page covers only what is specific to the catalog.

The owner's rulings for both loops:

- one branch per ticket;
- as much parallelism as the tickets allow;
- the supervisor reviews, gives feedback, and merges;
- the cost is acceptable for now.

## Switching modes

All state lives in GitHub (open PRs) and in the docs ([PLANS.md](PLANS.md),
[coverage.md](coverage.md)), so you can switch modes at any point.

- **Going inline.** Stop launching workers. Let running workers finish their
  PRs, or take the branches over yourself.
- **Going subagent-driven.** Write the next tickets as described below and
  launch workers for them.

When the account warns that a usage limit is close, tell the owner and drop
to fewer workers or to inline mode.

## Tickets

A ticket is **one family**: one module, one PR. It covers the cycle in
[README.md](README.md) from "read the twin" through "done". Families live in
separate modules (`scalers.py`, `impute.py`, `select.py`, ...), so tickets
are almost independent by construction.

### Where tickets come from

Take them in this order:

1. PLANS "Next".
2. The easiest "not yet" rows of [coverage.md](coverage.md).
3. Configurations a merged family left as `NotNative`, once confit has
   delivered what they waited on.

### Files every ticket shares

Several files are touched by every ticket. Expect conflicts there, and
resolve them at merge time:

- `catalog_test.py` (`FIXTURES`): every ticket adds rows. Workers append
  their rows in their own block. Merge conflicts in it are mechanical.
- `__init__.py`: a new family adds one import line.
- `coverage.md`: generated. Never resolve a conflict in it by hand.
  Regenerate it after merging master
  (`uv run python -m sql_transform.native.coverage --write`).
- `PLANS.md`: each worker edits only its own family's item and appends to
  "Needs from confit".
- `_helpers.py` and `_registry.py`: shared machinery. A ticket that has to
  change them says so up front. If two such tickets would run at once, run
  the helper change first as its own small ticket.

### How many workers at once

The number of independent families is rarely the limit. The limits are:

- **Review.** Each PR must be checked twin by twin.
- **Waiting on confit.** A family that waits on a confit capability is not a
  ticket yet. It goes under PLANS "Needs from confit".

Start with 2–3 workers. Add more when merges flow without rework.

## Launching a worker

Use the Claude Code Remote tool `create_session` with these arguments:

- `source_url`: `https://github.com/ahrzb/sql-transforms`
- `outcome_branch`: `claude/native-<family>`
- `title`: `native: <family>`
- `prompt`: the worker brief below, followed by the ticket.

The ticket names:

- the transformer classes;
- their sklearn module;
- the configurations to cover;
- known traps, such as dtype or NaN handling, or a reduction order that
  needs an ulp bound;
- any confit capability the family needs (merged, or not yet).

Record the session id and branch in PLANS next to the item.

Workers cannot message back, so follow them some other way:

- subscribe to their PRs once they exist;
- `get_session` (`status_bucket` reads `failed` when a turn errored);
- the session's events.

Correct a worker with `send_message`, stop it with `interrupt_session`, or
take the ticket back inline.

## Worker brief

The block below is sent verbatim, before the ticket.

```
You are a worker on the native-transform catalog in ahrzb/sql-transforms
(package packages/sql-transform, module sql_transform.native: fitted sklearn
transformers translated into confit SQL functions, checked against the
fitted estimator itself, the "twin"). A supervisor session assigned you ONE
family (below). It reviews your PR, may leave review comments, and merges
it. You never merge.

Read packages/sql-transform/docs/native/README.md (the cycle you will
follow, steps 3-7), goal.md, and the existing family modules (scalers.py,
impute.py, select.py) before writing code.

Ground rules
- Work only on your ticket's branch; create it from origin/master; push only
  there.
- Do not change packages/confit/. If the family needs a confit capability
  that is missing, record it under PLANS "Needs from confit" (what, and
  which configurations wait on it), leave those configurations NotNative
  with a reason naming the capability, and ship the rest.
- Never put a model name or model identifier in commits, PR text, or code.
- Commit trailer, exactly these two lines at the end:
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  Claude-Session: <your own session URL>
- PR body ends with "🤖 Generated with [Claude Code](https://claude.com/claude-code)"
  and then your session URL.
- After opening the PR, subscribe to its activity (subscribe_pr_activity);
  answer every review comment with a push or a reasoned reply; keep CI
  green; stop only when the PR is merged or closed.

Commands (repo root)
- Install: uv sync --locked --group spark --reinstall-package confit
  (never drop --group spark).
- Gate: uv run python scripts/gate.py   (must end green)
- Lint: git add -A first, then uv run pre-commit run --all-files; read the
  full output (ruff check AND ruff format).
- Coverage: uv run python -m sql_transform.native.coverage --write
- Never run pkill -f python style commands (they kill your own shell).

Discipline
- The twin is the reference; never change it to make an entry pass.
- Bit-exact (ulps=0) unless the twin's order cannot be reproduced; then
  measure the max ulp distance over >= 200 seeds, register that, and cite
  seeds/max/date in the translator's docstring. Never raise a bound to make
  a failure pass.
- Register exact classes only. A configuration you do not serve raises
  NotNative naming it.
- Record build and serve times for the widest configuration you test in
  the PR (the catalog's width limits are confit's business, and the
  supervisor forwards them).

PR: one family, one PR. Describe the twin's operation sequence, the
configurations served and refused, the bounds and how they were measured,
the timings, the gate, and the regenerated coverage.
```

## Review checklist (supervisor)

- [ ] The translator follows the twin's operation order, which the PR
      states. Spot-check one configuration against the installed sklearn
      source.
- [ ] Every configuration that changes `transform` is either fixtured or
      refused with `NotNative`.
- [ ] Bounds are 0, or measured and cited.
- [ ] Nothing outside the family module, its fixtures, `__init__.py`, PLANS
      and coverage changed. A change to a shared helper was announced in
      the ticket.
- [ ] The gate is green. `coverage.md` was regenerated, not hand-edited.
- [ ] Every new "Needs from confit" entry has a reproduction.

Merge one PR at a time, squash with the expected head SHA. Then merge
master into the other open branches and regenerate `coverage.md` in each.
Tell the confit loop about new needs. Its session id is in that loop's
messages; when you cannot reach it, rely on PLANS "Needs from confit", which
the confit loop reads.
