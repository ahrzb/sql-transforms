---
id: decision-8
title: 'What makes a frozen fold trustworthy: enumerate impure shapes vs pin the fold configuration — OPEN'
date: '2026-09-10'
status: proposed
---
## Context

A static-tables-only query is evaluated once at build by DuckDB and frozen; serving
replays the frozen rows. The goal's exclusion `whole-relation-shapes` says a
whole-relation construct may be frozen **only when what it selects is a function of the
query text and the statics**, and gives the reason: two builds of the same function must
not disagree.

Today the `refuse-static-tie-order` branch enforces that by reading DuckDB's parse and
catalogue for shapes whose answer depends on something else, and refusing them by name.
Six review rounds have found eighteen such shapes; each round closed everything the last
one found and the next round found more, and the newest ones carry no name a reading can
see — an opaque `SHOW_REF` node, a file path standing where a table name stands, a time
zone applied by a plain cast, a zoned type chosen at bind time from a string argument.

The question underneath is **which property we are buying**. A fold's answer can depend on
three things beyond the query and the statics:

| class | examples | what fixes it |
|---|---|---|
| execution order | tie order, `first`, `list`, `avg` over doubles, `LIMIT` without a total order, `ROWS` frames, `ASOF` and `POSITIONAL` joins, `DISTINCT ON` | a pinned configuration: one thread, insertion order preserved, the same wheel |
| session settings | `TimeZone`, `Calendar`, the optimizer | set explicitly on the fold connection |
| reads of the run | clocks, `random`, files, the catalogue, `version()`, `SUMMARIZE` | cannot be pinned; must be refused, but DuckDB flags most of them and two switches close the rest (external access off, a `FROM` allow-list) |

Enumeration treats all three classes the same way, by naming shapes. Pinning removes the
first two classes by construction and leaves a delimited third class, which is the class
the row path also refuses.

## Decision framework

Three questions, in order, because each one's answer decides whether the next is worth
asking.

1. **Is a pinned fold actually deterministic?** Measurable in an hour: rerun the batteries
   that produced 5–15 distinct answers per shape under a pinned connection, across twenty
   fresh processes, and count answers per shape. If any shape still moves, the fork closes
   and enumeration stays. Expectation: one answer each — DuckDB has no randomized hash seed
   and a single thread has no scheduling.
2. **Does "a function of the query, the statics and the pinned fold" satisfy the goal?**
   The exclusion's stated *reason* is build-to-build agreement, which pinning delivers. Its
   *letter* says "a function of the query", which a stable-but-arbitrary tie order is not.
   This is the ratification question, and it is the owner's: if the reason is what matters,
   ties, positional selection and order-dependent aggregates can serve again,
   deterministically. If the letter is what matters, pinning still closes the nameless
   class but the semantic refusals stay.
3. **Which reading does the fold use?** Production folds with the optimizer on; every gate
   compares against an optimizer-off oracle; the oracle spec's `ask: engine-fold-reading`
   has been open since the spec landed. Pinning through the `Oracle` class settles it as
   optimizer-off, one door for every DuckDB read, and answers
   `ask: threads-and-value-order` in the same stroke (threads joins the oracle constant;
   order inside a value is then deterministic and compared ordered).

## Options

### A. Keep enumerating

Round 6 is the latest; rounds continue until a review comes back empty.

- **Pros:** no change to the goal's wording; every rule is measured; the spec stays as
  written.
- **Cons:** six rounds have not converged and the last findings were nameless, so there is
  no evidence of a floor; the accepted surface shrinks each round (64 of 88 aggregate
  names, every table function outside five, collations, zoned types, every base table
  outside statics/CTEs); the reading is now the largest function in the engine and grows a
  list per round; a fail-open serves a wrong constant **silently**, the worst failure mode
  the contract has, and enumeration cannot bound it.

### B. Pin the fold through the Oracle

The fold connects through `confit.oracle` with optimizer off, one thread, insertion order
preserved, `TimeZone` and `Calendar` set, external access disabled. The exclusion is
reworded to "a function of the query, the statics **and the pinned fold**". Refusals stay
only for reads of the run: the stability flags, the four run-state names and one-argument
`age`, the `FROM` and table-function allow-lists, `SHOW_REF`. The tie probe and the
position, order-dependent-aggregate, frame and join refusals retire.

- **Pros:** build-to-build agreement by construction, not by a list; the enumeration ends
  at a class DuckDB mostly flags itself; most of the acceptance price comes back; the
  oracle's one-door claim becomes true and two open asks close; the 4–5× build cost of the
  probes goes away.
- **Cons:** the goal's wording changes, which is a ratification, not a fix; frozen answers
  become a function of the wheel version, so a DuckDB bump can change constants (a rebuild
  already implies that, and `VERSION` is recorded, but it becomes visible); the
  optimizer-off fold may trap where the optimizer used to fold a trap away, which turns
  some served constants into named refusals and needs one gated run to price; the premise
  in question 1 must be measured before anything is built.

### C. Pin for safety, keep the semantic refusals

Same configuration as B, but the exclusion's letter stays and the by-name refusals for
ties, position and order-dependent aggregates remain.

- **Pros:** closes the nameless value class (zoned rendering, float accumulation order)
  without touching the goal's wording.
- **Cons:** no acceptance comes back; enumeration continues for the semantic shapes, which
  is where the rounds have been; two mechanisms guard one property.

### D. Fold twice under two configurations and refuse on disagreement

**Considered and rejected:** it is a probability, not a proof, it doubles the fold, and it
cannot see reads of the run at all.

## Recommendation

**B, gated by the measurement in question 1, with C as the fallback if question 2 is
answered with the letter.** The measurement is one agent and one hour and it decides
whether there is a fork at all. If it holds, B is the only option that ends the
enumeration rather than continuing it, and it is the one that makes the spec's one-door
claim true.

What the owner must supply is the answer to **question 2**, since it rewrites a sentence in
`goal.md`. Questions 1 and 3 can be settled by measurement and reported.

## Decision

**OPEN — the owner's call.** This record captures the fork and the framework, not a
ruling.

## The four goal asks, and why two of them wait

Written in `goal.md` with their options; recommendations with the dependency stated:

- **`ask: acceptance-target`** — option (b), ratchet without a target: the dialect floors
  already run that rule, and a target on a grammar rate optimizes the generator as readily
  as the engine. Answer together with the oracle spec's `ask: match-count-ratchet`, as the
  doc says.
- **`ask: exclusion-ratification`** — ratify the six rows, with the `whole-relation` row
  reworded if B is chosen. Answering before this RFC would ratify wording the RFC may
  change, so decide the fork first.
- **`ask: kpi-set-change`** and **`ask: next-query-classes`** — both wait on the fork: the
  accepted surface and the refusal vocabulary are different under B than under A, and the
  first KPI worth adopting (`named-refusal-share`) needs the refusal registry the report
  already queues.
