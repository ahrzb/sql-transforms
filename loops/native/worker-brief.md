# Native worker brief

The second part of every native worker's opening prompt: after
[`../worker-brief.md`](../worker-brief.md), before the ticket's section of
[`tickets.md`](tickets.md) ([`../workers.md`](../workers.md) §4). It holds
only what the shared brief does not say. Send it verbatim.

---

You work on the native-transform catalog: package `packages/sql-transform`,
module `sql_transform.native`, which translates fitted sklearn transformers
into confit SQL functions, checked against the fitted estimator itself (the
"twin"). Your ticket is ONE family: one module, one PR.

Read `loops/native/README.md` (the cycle you follow, steps 3-7, and
"Tickets and review"), `loops/native/goal.md`, and the existing family
modules (`scalers.py`, `impute.py`, `select.py`, `encode.py`) before writing
code.

## What you must not touch
- `packages/confit/` and `loops/confit/`. If the family needs a confit
  capability that is missing, record it under PLANS "Needs from confit"
  (what, and which configurations wait on it, with a reproduction), leave
  those configurations `NotNative` with a reason naming the capability, and
  ship the rest.
- `_registry.py`, `_helpers.py`, `_check.py`: shared machinery. If your
  family needs a change there, keep it minimal and name it first in the PR.

## Files other workers change at the same time
- `catalog_test.py`: append your `FIXTURES` rows as your own block.
- `__init__.py`: one import line for your module.
- `loops/native/coverage.md`: regenerate it
  (`uv run --no-sync python -m sql_transform.native.coverage --write`),
  never edit it by hand.
- `loops/native/PLANS.md`: edit only your family's item (remove it from
  "Next" when it lands, add your "Left Python" lines) and append to "Needs
  from confit".

## Discipline
- The twin is the reference; never change it to make an entry pass.
- Bit-exact (`ulps=0`) unless the twin's order cannot be reproduced; then
  measure the max ulp distance over >= 200 seeds (`NATIVE_SEEDS=200` runs
  the catalog's parity test that many seeds), register that, and cite
  seeds, max and date in the translator's docstring. Never raise a bound to
  make a failure pass. A bound that is not small is a question for the owner
  (`loops/native/decisions/open/`), not code.
- Register exact classes only. A configuration you do not serve raises
  `NotNative` naming it.
- Record build and serve times for the widest configuration you test; the
  supervisor forwards width limits to the confit loop.

## The PR description shows
The twin's operation sequence, the configurations served and refused, the
bounds and how they were measured, the timings, the gate, and the
regenerated coverage.
