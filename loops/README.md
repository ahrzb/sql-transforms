# Loops

A loop is an agent session that turns a goal into merged PRs on its own, one
verified change at a time. It stops only when the remaining work needs the
owner.

This directory holds the files that drive the loops. These are their goals,
their plans, their ticket boards, the owner's decision records and their
reports.

A loop also produces files for the product. These are contracts, specs,
pins, technical write-ups, and any file that code or tests read. Those files
stay with the package of the loop.

## The loops

| loop | works on | folder | start it with |
|---|---|---|---|
| `confit` | the confit serving engine (`packages/confit`) | [`confit/`](confit/) | `/loop` Work the confit loop: follow loops/confit/README.md. |
| `native` | the native catalog of sklearn transformers (`packages/sql-transform`, `sql_transform.native`) | [`native/`](native/) | `/loop` Work the native-transform loop: follow loops/native/README.md. |

The folder name is the `loop:` key of the KPI block in
[`reporting.md`](reporting.md). The KPI block is the front matter of a
report, and it holds fixed keys that the owner's tracker reads. With this
key, a report maps to its folder without a lookup table.

> **Migration (2026-10-05).** Each loop moved its own files here from its old
> place. The native loop has moved. The confit loop has moved. The file
> `packages/confit/docs/loop/README.md` now keeps only a pointer to
> [`confit/`](confit/README.md).

## Shared rules

- [`../AGENTS.md`](../AGENTS.md) gives the rules for every agent. Text for
  the owner follows the `simple-english` skill. The terms of the project are
  in [`../GLOSSARY.md`](../GLOSSARY.md).
- [`reporting.md`](reporting.md) says when a loop writes a report, what the
  KPI block holds, and how a report reaches the owner.
- [`workers.md`](workers.md) describes the inline mode and the
  worker-driven mode. It also describes tickets, the board, how to start
  workers, review and merge.
- [`worker-brief.md`](worker-brief.md) is the part of the opening prompt of
  every worker that both loops share.

## The files of a loop folder

Every loop folder has the same files. An agent or the owner who knows one
folder then knows both.

```
<loop>/
  README.md        how the loop runs: its cycle, then "Tickets and review"
  goal.md          what done means, and the scope or a link to it
  PLANS.md         open work, highest value first
  tickets.md       the board: what is in flight (workers.md, "The board")
  worker-brief.md  this loop's part of a worker's prompt
  decisions/       open/ questions for the owner; closed/ and postponed/ rulings
  reports/         YYYY-MM-DD-<slug>.md, per reporting.md
```

A loop adds the files that only it needs. The native loop adds
`report-format.md`. Its scoreboard is a product file, so it is in the package
spec: `packages/sql-transform/spec/native/coverage.md`.

## Between the loops

Each loop owns its folder and its package. A loop never edits the folder or
the package of the other loop. It asks the other loop instead.

A need is a capability that one loop asks the other loop to make. The asking
loop sends each need to the session of the other loop by message. The asking
loop also records the need in its own PLANS.

- The native loop records its needs under "Needs from confit".
- The confit loop takes the needs as its second source of tickets.

When the serving loop serves a need, it tells the asking loop.
