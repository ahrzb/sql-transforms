# Loop reporting policy

This policy applies to both loops: the confit loop
([`confit/`](confit/README.md)) and the native-transform loop
([`native/`](native/README.md)). It says when a loop writes a report. It says
which key performance indicators (KPIs) every report carries. It also says
how a report reaches the owner.

Each loop keeps the shape of its own full report:

- The native loop uses [`report-format.md`](native/report-format.md).
- The confit loop uses its goal readings. One example is
  [`2026-09-26-goal-reading-n3.md`](confit/reports/2026-09-26-goal-reading-n3.md).

The owner reads the loops through a tracker. The tracker refreshes every 3
hours from master, from the PRs and from these reports. It reads the KPI
block below, so keep the keys exact.

## 1. When to report

A loop writes a report when any of these triggers happens. A loop never
writes one report for each PR.

| trigger | confit loop | native loop |
|---|---|---|
| batch | a set of tickets merges, or a wave of workers merges | about five families merge, or a wave of workers ends |
| KPI crosses a mark | a `kpi:` in [`specs/success-measures.md`](../packages/confit/docs/specs/success-measures.md) changes state (opens, closes or regresses) | coverage crosses a quarter (17, 34, 51 or 68 of 68), or a whole sklearn module goes native |
| cross-loop delivery | a catalog need lands. Say what it unblocked. | a confit capability lands. Count the `NotNative` configurations that it turned native. |
| regression | a gated campaign class appears again, or the nightly finds a new class | a registered ulp bound grows, or an entry goes back to `NotNative` |
| stop | PLANS has no work that the loop can do without the owner | same |
| staleness | 24 h of activity, or 15 merged PRs, since the last report. Write a short reading (§3) even when no other trigger happened. | same |

## 2. The KPI block (required, machine-readable)

Every report starts with a front-matter block that the tracker reads. The
block has the same keys every time, so the tracker can always compute a
delta.

```yaml
---
loop: native            # or confit
date: 2026-10-05
master: c3edefb
previous: none          # or the path of the previous report
kind: milestone         # milestone | short
trigger: batch          # which row of the trigger table above
kpis:                   # fixed per loop, see below; `null` if not measured this time, never omitted
  coverage_native: 20
  coverage_partial: 0
  coverage_notnative: 0
  coverage_not_yet: 48
  nonzero_ulp_bounds: 0
  families_merged_since_last: 5
  widest_build_s: null
  widest_serve_us_per_64: null
  twin_us_per_64: null
  open_needs_from_confit: 1
needs_owner:            # decisions/open records this report asks the owner to rule on
  - loops/native/decisions/open/matvec-parity-bound.md
needs_from_other_loop: [cse-within-call]
---
```

**The fixed KPIs:**

- **Native:**
  - `coverage_native`, `coverage_partial`, `coverage_notnative` and
    `coverage_not_yet`. These come from the regenerated `coverage.md`.
  - `nonzero_ulp_bounds`.
  - `families_merged_since_last`.
  - `widest_build_s`, `widest_serve_us_per_64` and `twin_us_per_64`. These
    three keys describe the slowest family.
  - `open_needs_from_confit`.
- **Confit:**
  - One key for each `kpi:` in `specs/success-measures.md`. The keys are
    `training_round_trip`, `engine_parity`, `binding_parity`,
    `transformer_parity`, `no_third_mode`, `coverage_ladder` and
    `serving_latency`. Each key has the value `pass`, `fail` or `open`, with
    a value of one line.
  - Then `corpus_match`, `corpus_total` and `corpus_fail`.
  - Then `gated_campaign_classes`, `catalog_needs_open` and
    `catalog_needs_served_since_last`.

## 3. Two forms

- **Milestone report:** Keep the existing shape. For the native loop, this is
  sections 0 to 8 of `report-format.md`. For the confit loop, this is the
  shape of its goal reading. In a milestone report:
  - Write each claim about an outcome in bold.
  - Give deltas, not only levels.
  - Measure again each number on the SHA that the report names.
  - Report bad news as plainly as good news.
- **Short reading:** Use it for staleness. It has the KPI block and a maximum
  of 40 lines. The lines have three sections: "What moved" (a maximum of
  three paragraphs), "Needs the owner" and "Next". Do not measure speed
  again, unless a family or a fix changed the speed.

Both forms follow the writing rules in [`AGENTS.md`](../AGENTS.md) §1 and
use the terms in [`GLOSSARY.md`](../GLOSSARY.md). Before you merge a report, use the
`fresh-reader` skill (`AGENTS.md` §3).

Both forms end with a section called **Needs the owner**. It lists every open
decision record. For each record, it gives the one line of evidence that the
owner needs to rule. If nothing needs the owner, write "Nothing." in the
section.

## 4. Where it goes, and the hand-off

- Put the file in `loops/confit/reports/` or in `loops/native/reports/`.
  Name it `YYYY-MM-DD-<slug>.md`.
- Never edit an older report. A new reading names the report that it follows.
- Commit the report in the PR that completes the milestone, or in its own
  small PR. The loop merges it under its usual self-merge rule.
- Start the PR title with `report:`. Then the tracker finds the report
  without a scan of the prose.
- Do not message the owner about a report. The tracker picks the report up
  within 3 hours. It shows anything under "Needs the owner".
