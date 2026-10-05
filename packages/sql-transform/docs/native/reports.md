# Milestone reports

After each milestone, the loop writes one report. A report tells the owner
what the catalog can do now, what changed since the last report, and what
remains open. The confit loop already reports this way; see
`packages/confit/docs/reports/2026-09-26-goal-reading-n3.md` for a full
example. This page explains how the native loop does it.

## When to report

The triggers, the KPI block every report opens with, and the hand-off are
shared with the confit loop: see
[`docs/loop-reporting.md`](../../../../docs/loop-reporting.md). This page keeps
the native loop's own parts: where the file goes and the shape of a full
report.

Do not write a report per PR. The PR descriptions already cover single
changes.

## Where to put it

**The file.** Save it as `docs/native/reports/YYYY-MM-DD-<slug>.md`. Never
edit an older report. A new reading is a new file, and it names the report
it follows.

**Committing it.** Commit the report in its own small PR, or in the PR that
completes the milestone. Open the file with the KPI block from
[`docs/loop-reporting.md`](../../../../docs/loop-reporting.md#2-the-kpi-block-required-machine-readable),
and start the PR title with `report:`.

**Publishing it.** If the session can publish a document for the owner (for
example a Claude Doc or an Artifact), publish the same text and link both in
the hand-off message.

## The shape

The sections below are in order. Number them, and give each an anchor
(`{#what-moved}` and so on).

0. **What this is.** One paragraph:
   - the master SHA and the date;
   - the previous report, with its SHA;
   - what was measured, with the same commands as last time so that the
     deltas mean something.

1. **What moved.** Four to six short paragraphs, the most important first.
   Each one opens with a bold claim that is an outcome, then gives the
   evidence with numbers. For example: **"Every scaler is native and
   bit-exact."** followed by the evidence.
   - Lead with outcomes: what a user gets now that they did not before.
     Work done is not an outcome.
   - Name the downstream effect when there is one. For example, a pipeline
     that now serves end to end, or a transformer cost that dropped from
     about 108 µs to a few µs per row.

2. **Scoreboard.** One table per sklearn module, plus one totals row:

   | module | native | partly native | NotNative | not yet | delta since last |
   |---|---|---|---|---|---|

   Take the counts from the regenerated coverage, never from memory.

3. **Speed.** For each family that merged since the last report, measure
   the widest configuration tested:
   - the build time;
   - the serve time per 64-row call;
   - the twin's time for the same call;
   - the speedup.

   Mark anything still capped by a confit limit, and name the limit.

4. **Exactness.** List every registered bound that is not 0. For each one
   give the family, the configuration, the ulp bound, the seeds and date
   measured, and the reason. The reason is usually a reduction order that
   cannot be reproduced. Say "all bit-exact" when that is true.

5. **Findings.** Number the bugs found and fixed, in the catalog or in
   confit. For each one: what it was, how it was found, and the PR that
   fixed it. Give each finding a slug such as `finding: minmax-clip-nan`,
   and keep the slug unchanged across reports, so the next report can say
   it is closed.

6. **Needs from confit.** The state of each need:
   - delivered, with the PR and the configurations it unblocked;
   - open, with what still waits on it and the numbers.

7. **What this leaves open.** The next families, ordered by value. Then the
   questions for the owner, each with the evidence the owner needs to rule.
   Those questions also have records in `decisions/open/`.

8. **Environment and reproduction.** A table that lists:
   - the SHA;
   - the versions of DuckDB, sklearn, numpy and Python;
   - the machine;
   - the exact command behind every number in the report.

   Someone else should be able to re-take the reading from this table
   alone.

## Style

- **Facts, not adjectives.** "Bit-exact over 200 seeds" says something;
  "robust" does not. Every number carries its unit and its sample size.
- **Deltas over levels.** Compare against the previous report at every turn.
  A number without its previous value is half a finding.
- **Report bad news as plainly as good news.** That covers a bound that
  grew, a family that regressed, and a limit that still blocks. A report
  that hides a regression costs more than the regression.
- **Measure, then write.**
  - Run every command in section 8 on the SHA in section 0 before writing.
  - Never copy a number forward from a PR description without re-measuring
    it.
  - Use release builds for timings.
- **Short paragraphs.** Each opens with its claim in bold. Use a table
  whenever there are more than three comparable numbers.
- **No model names or model identifiers** in the report.

## In the subagent-driven mode

The supervisor writes the report itself, at the end of a wave. Writing it is
part of reviewing the wave's work. A worker never writes one. The merged
PRs' descriptions are inputs, and every number is re-measured on master.
