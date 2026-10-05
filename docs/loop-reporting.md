# Loop reporting policy

Applies to both loops: the confit loop
([`packages/confit/docs/loop/`](../packages/confit/docs/loop/README.md)) and the
native-transform loop
([`packages/sql-transform/docs/native/`](../packages/sql-transform/docs/native/README.md)).
It says when a loop reports, which KPIs every report carries, and how a report
reaches the owner. The shape of a full report stays each loop's own: the native
loop's [`reports.md`](../packages/sql-transform/docs/native/reports.md), and the
confit loop's goal readings (for example
[`2026-09-26-goal-reading-n3.md`](../packages/confit/docs/reports/2026-09-26-goal-reading-n3.md)).

The owner reads the loops through a tracker that refreshes every 3 hours from
master, the PRs and these reports. It reads the KPI block below, so keep its
keys exact.

## 1. When to report

A loop writes a report when any of these happens, and never once per PR:

| trigger | confit loop | native loop |
|---|---|---|
| batch | a ticket batch or a wave of workers merges | about five families merge, or a wave of workers ends |
| KPI crosses a mark | a `kpi:` in [`specs/success-measures.md`](../packages/confit/docs/specs/success-measures.md) changes state (opens, closes, regresses) | coverage crosses a quarter (17, 34, 51, 68 of 68), or a whole sklearn module goes native |
| cross-loop delivery | a catalog need lands; say what it unblocked | a confit capability lands; count the `NotNative` configurations it turned native |
| regression | a gated campaign class reappears, or the nightly finds a new class | a registered ulp bound grows, or an entry goes back to `NotNative` |
| stop | PLANS has nothing left that it can do without the owner | same |
| staleness | 24 h of activity, or 15 merged PRs, since the last report: write a short reading (section 3) even when no trigger fired | same |

## 2. The KPI block (required, machine-readable)

Every report starts with a front-matter block that the tracker reads. It has the
same keys every time, so a delta is always computable:

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
  - packages/sql-transform/docs/native/decisions/open/matvec-parity-bound.md
needs_from_other_loop: [cse-within-call]
---
```

**The fixed KPIs:**

- **Native:** `coverage_native`, `coverage_partial`, `coverage_notnative`,
  `coverage_not_yet` (from the regenerated coverage.md); `nonzero_ulp_bounds`;
  `families_merged_since_last`; `widest_build_s`, `widest_serve_us_per_64`,
  `twin_us_per_64` for the slowest family; `open_needs_from_confit`.
- **Confit:** one key per `kpi:` in `specs/success-measures.md` (`training_round_trip`,
  `engine_parity`, `binding_parity`, `transformer_parity`, `no_third_mode`,
  `coverage_ladder`, `serving_latency`), each `pass`, `fail` or `open` with a
  one-line value. Then `corpus_match`/`corpus_total`/`corpus_fail`,
  `gated_campaign_classes`, `catalog_needs_open` and `catalog_needs_served_since_last`.

## 3. Two forms

- **Milestone report:** the existing shape, unchanged (native `reports.md`
  sections 0–8; confit's goal-reading shape). Bold outcome claims, deltas over
  levels, every number re-measured on the SHA named, bad news as plainly as good.
- **Short reading:** used for staleness. The KPI block plus at most 40 lines:
  "What moved" (three paragraphs at most), "Needs the owner" and "Next". No
  re-measuring of speed unless a family or a fix changed it.

Both forms end with **Needs the owner**: every open decision record, with the one
line of evidence the owner needs to rule. When nothing needs the owner, write
"Nothing." explicitly.

## 4. Where it goes, and the hand-off

- File: `packages/confit/docs/reports/` or `packages/sql-transform/docs/native/reports/`,
  named `YYYY-MM-DD-<slug>.md`. Never edit an older report: a new reading names
  the one it follows.
- Commit it in the PR that completes the milestone, or in its own small PR.
  The loop merges it under its usual self-merge rule.
- The PR title starts with `report:`, so the tracker finds it without
  scanning prose.
- The loop does not message the owner about a report. The tracker picks it up
  within 3 hours and surfaces anything under "Needs the owner".
