# Confit engine

**Kind:** standing mandate. **Status:** paused.
**Limits:** claimed tasks: 4.

## Objective

Widen the SQL that confit serves and lower its serving latency, without weakening the [oracle contract](../../packages/confit/docs/oracle/README.md) or the [serving contract](../../packages/confit/docs/specs/serving-contract.md).
Confit specializes SQL feature transforms into functions that answer one request at a time, with no SQL engine per request.

**Measure:**

- the `kpi:` entries of the [success measures](../../packages/confit/docs/specs/success-measures.md);
- the corpus counts in `packages/confit/docs/reports/corpus-counts.json`;
- the gated classes of the campaigns and of the nightly campaign.

**Baseline:** the [baseline report](reports/2026-09-02-goal-baseline.md) of 2026-09-02.

**Better:** more measures pass, more corpus queries match, no gated class appears, and serving latency falls.
Every accepted transform still gives the oracle's answer, and every unsupported construct still refuses by name at build.

## Boundary

**May, without asking:**

- Change `packages/confit/` (Rust sources, Python package, `fuzz/`, tests, benchmarks) to serve a query class, fix a divergence, refuse a construct by name, or make serving or building faster. Except:
  - changing `packages/confit/docs/oracle/` or `packages/confit/docs/specs/serving-contract.md`, other than editorial changes and **Not built yet.** markings;
  - adding or widening an exclusion in `fuzz/exclusions.py` that no [closed proposal](../../proposals/README.md) admits;
  - serving a query class out of the order that [next query classes](../../proposals/closed/next-query-classes.md) gives.
- Correct `packages/confit/docs/known-limitations.md`, `packages/confit/docs/specs/success-measures.md` measurements and `packages/confit/docs/reports/` to match merged behaviour. Except: changing a success measure's definition.
- Pin a live divergence as a strict expected failure in `packages/confit/tests/test_open_divergences.py`. Except: pinning a gated class that a fix in reach would remove.
- Run campaigns, and triage the issue titled "Nightly campaign findings" by commenting on it and closing it when every class is fixed, pinned or excused. Except: changing `.github/workflows/`.
- Write this mandate's brief (its **Traps**, and standing instructions that apply a closed proposal or an answer), tracker, task descriptions, audit log and reports. Except: changing this README.
- Add needs to the tracker of [native catalog](../native/README.md), including notices that a merge changed the catalog's build or serving costs.
- Write research projects under `research/` for findings that outlive a task.
- Push branches named `claude/confit-<name>`, open pull requests from them, and squash-merge them with the expected head SHA once CI is green. Except: force-pushing or rebasing another session's branch; merging a change that fails a required check.
- Start read-only helpers to investigate DuckDB's behaviour, review a change, or measure. Except: none.
- Start runbook helpers following the brief's helper procedure and their task description, writing only within their task's branch and the files its description names. Except: a helper never merges, never writes a report, and never edits `packages/sql-transform/`.

**Independent review:** only what AGENTS.md requires.

## Reporting contract

The default, with these additions.

**Added triggers:**

- A `kpi:` of the success measures changes state (opens, closes or regresses).
- A need of the native catalog lands. Say what it unblocked.
- A gated campaign class appears again, or the nightly finds a new class.
- 15 merged pull requests since the last report.

**Added detail:**

- Start each report with this front-matter block. The maintainer's tracker reads it every 3 hours from master, so keep the keys exact; write `null` for a value not measured, never omit a key. A measure's value is `pass`, `fail` or `open`, with one line.

  ```yaml
  ---
  loop: confit
  date: <YYYY-MM-DD>
  master: <short sha>
  previous: <path of the previous report, or none>
  kind: <milestone | short>
  trigger: <the trigger>
  kpis:
    training_round_trip: <value>
    engine_parity: <value>
    binding_parity: <value>
    transformer_parity: <value>
    no_third_mode: <value>
    coverage_ladder: <value>
    serving_latency: <value>
    corpus_match: <n>
    corpus_total: <n>
    corpus_fail: <n>
    gated_campaign_classes: <n>
    catalog_needs_open: <n>
    catalog_needs_served_since_last: <n>
  needs_owner: [<paths of the open questions>]
  needs_from_other_loop: [<need names>]
  ---
  ```

- Start the pull request title with `report:`.
- Under **Where it stands**: each measure against the baseline, the latest campaign's seed range and gated classes, and the corpus counts.
- Every number cites a script in the repository at the measured commit; data may stay outside.

## Files

- [Brief](brief.md)
- [Tracker](tracker.md)
- [Audit log](audit.md)
- [Reports](reports)
