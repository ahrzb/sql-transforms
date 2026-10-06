# Milestone reports

After each milestone, the native loop writes one report. The report tells
the owner what the catalog can do now, what changed since the last report,
and what stays open. The confit loop reports in the same way. Its
[goal reading of 2026-09-26](../confit/reports/2026-09-26-goal-reading-n3.md)
is a full example. This page says how the native loop writes its reports.

## When to report

[`../reporting.md`](../reporting.md) gives the rules that both loops share:
the triggers, the KPI block that opens each report, and the hand-off. This
page gives the parts that only the native loop has: where the file goes,
and the shape of a full report.

Do not write a report for each PR. The descriptions of the PRs already
cover single changes.

## Where to put it

**The file.** Save the report as
`loops/native/reports/YYYY-MM-DD-<slug>.md`. Never edit an older report. A
new report is a new file, and it names the report that it follows.

**The PR.** Commit the report in its own small PR, or in the PR that
completes the milestone. Open the file with the KPI block from
[`../reporting.md`](../reporting.md#2-the-kpi-block-required-machine-readable).
Start the title of the PR with `report:`.

**The text.** The report follows the `simple-english` skill and uses the
terms in [GLOSSARY.md](../../GLOSSARY.md). Before you merge it, run the
checker and the fresh-reader test of that skill.

**Publishing it.** If the session can publish a document for the owner,
such as a Claude Doc or an Artifact, publish the same text. Link both in
the hand-off message.

## The shape

The sections below come in this order. Number them, and give each one an
anchor (`{#what-moved}` and so on).

0. **What this is.** Write one paragraph that gives:
   - the master SHA and the date;
   - the previous report, with its SHA;
   - what the report measured. Use the same commands as the previous
     report. Then the differences from that report (the deltas) are
     measurements, not noise.

1. **What moved.** Write four to six short paragraphs, the most important
   first. Each one starts with a claim about an outcome, in bold. The
   evidence follows, with numbers. For example: **"Every scaler is native
   and bit-exact."**, then the evidence.
   - Start with outcomes: what a user gets now that the user did not get
     before. Work that was done is not an outcome.
   - Name the effect downstream when there is one. One example is a
     pipeline that now serves from end to end. Another is a cost that fell
     from about 108 µs to a few µs for each row.

2. **Scoreboard.** Give one table for each sklearn module, and one row of
   totals:

   | module | native | partly native | NotNative | not yet | delta since last |
   |---|---|---|---|---|---|

   Take the counts from the regenerated coverage. Never take them from
   memory.

3. **Speed.** For each family that merged since the last report, measure
   the widest configuration that the tests cover:
   - the build time;
   - the time to serve a call of 64 rows;
   - the twin's time for the same call;
   - the speedup.

   If a confit limit still stops a configuration, mark the configuration
   and name the limit.

4. **Exactness.** List every registered ulp bound that is not 0. For each
   bound, give the family, the configuration, the bound, the seeds and date
   of the measurement, and the reason. The reason is usually an order of
   reduction that the entry cannot follow. If every entry is bit-exact,
   say so.
   - A class can declare its bound for each configuration
     (`translates(cls, ulps=ceiling, bound=...)`). List each configuration
     with a bound above 0 on its own line, for example
     `FunctionTransformer(np.log10)`, 2 ulps.
   - The KPI `nonzero_ulp_bounds` counts the catalog classes that serve a
     configuration within a bound above 0. A class with a bound for each
     configuration counts once, for any number of bounded configurations.
     `python -m sql_transform.native.coverage --bounds` prints the count.
   - Run the parity test on 200 seeds of every fixture configuration
     (`NATIVE_SEEDS=200`). Give the number of steps and of configurations
     that it checked, and its failures and skips.

5. **Findings.** Number the findings that the loop found and fixed, in the
   catalog or in confit. For each finding, give what it was, how the loop
   found it, and the PR that fixed it. Give each finding a slug, such as
   `finding: minmax-clip-nan`. Keep the slug the same in later reports, so
   that the next report can say that the finding is closed.

6. **Needs from confit.** Give the state of each need:
   - delivered, with the PR and the configurations that it turned native;
   - open, with what still waits on it, and the numbers.

7. **What this leaves open.** List the next families, the most valuable
   first.

8. **Environment and reproduction.** Give a table with these items:
   - the SHA;
   - the versions of DuckDB, sklearn, numpy and Python;
   - the machine;
   - the exact command behind every number in the report.

   Another person must be able to take the measurements again from this
   table alone.

The report ends with the section **Needs the owner** from
[`../reporting.md`](../reporting.md). It lists every open decision record,
with the one line of evidence that the owner needs to rule.

## Style

- **Give facts, not adjectives.** "Bit-exact over 200 seeds" says
  something. "Robust" does not. Every number has its unit and the size of
  its sample.
- **Give deltas, not only levels.** Compare with the previous report each
  time. A number without its previous value tells half of the story.
- **Report bad news as plainly as good news.** This includes a bound that
  grew, a family that regressed, and a limit that still blocks work. A
  report that hides a regression costs more than the regression.
- **Measure, then write.**
  - Run every command of section 8 on the SHA of section 0 before you
    write.
  - Never copy a number from a PR description without a new measurement.
  - Use release builds for timings.
- **Use short paragraphs.** Each paragraph starts with its claim in bold.
  Use a table when there are more than three numbers to compare.
- **Never write a model name or a model identifier** in a report.

## In the worker-driven mode

The supervisor writes the report itself, at the end of a wave. The report
is part of the review of the wave's work. A worker never writes one. The
descriptions of the merged PRs are inputs, and the supervisor measures
every number again on master.
