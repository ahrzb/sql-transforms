# Loops

A loop is an agent session that turns a goal into merged PRs on its own, one
verified change at a time, and stops only when what is left needs the owner.
This directory holds what drives the loops: their goals, plans, ticket boards,
the owner's decision inbox, and their reports. What a loop *produces* for the
product (contracts, specs, pins, technical write-ups, and any file code or
tests read) stays with its package.

## The loops

| loop | works on | folder | start it with |
|---|---|---|---|
| `confit` | the confit SQL specializer (`packages/confit`) | [`confit/`](confit/) | `/loop` Work the confit loop: follow loops/confit/README.md. |
| `native` | the native sklearn catalog (`packages/sql-transform`, `sql_transform.native`) | [`native/`](native/) | `/loop` Work the native-transform loop: follow loops/native/README.md. |

The folder name is the `loop:` key of the KPI block in
[`reporting.md`](reporting.md), so a report maps to its folder without a
lookup table.

> **Migration in progress (2026-10-05).** Each loop moves its own files here
> from its old place (`packages/confit/docs/loop/`, `packages/confit/PLANS.md`,
> `packages/sql-transform/docs/native/`). Until a loop's folder exists, its old
> README is still the entry point.

## Shared rules

- [`reporting.md`](reporting.md): when a loop reports, the KPI block, and how
  a report reaches the owner.
- [`workers.md`](workers.md): inline and worker modes, tickets and the board,
  launching workers, review and merge.
- [`worker-brief.md`](worker-brief.md): the part of every worker's opening
  prompt that both loops share.

## The shape of a loop folder

Every loop folder has the same files, so an agent or the owner who knows one
knows both:

```
<loop>/
  README.md        how the loop runs: its cycle, then "Tickets and review"
  goal.md          what done means, and the scope
  PLANS.md         open work, highest value first
  tickets.md       the board: what is in flight (workers.md, "The board")
  worker-brief.md  this loop's part of a worker's prompt
  decisions/       open/ questions for the owner; closed/ and postponed/ rulings
  reports/         YYYY-MM-DD-<slug>.md, per reporting.md
```

A loop adds files it alone needs (the native loop's generated `coverage.md`,
its `report-format.md`).

## Between the loops

Each loop owns its folder and its package. A loop never edits the other's
folder or package; it asks. Needs go to the other loop's session by message
and are recorded in the asking loop's PLANS ("Needs from confit" in native;
confit takes them as its second ticket source). When a need is served, the
serving loop tells the asking loop.
