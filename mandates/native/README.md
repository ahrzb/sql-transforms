# Native catalog

**Kind:** mission. **Status:** running.
**Limits:** claimed tasks: 4.

## Objective

Grow the native catalog (`sql_transform.native`) until every scikit-learn transformer in scope is native in the [coverage table](../../packages/sql-transform/spec/native/coverage.md).
`to_native(step)` turns a fitted `PythonTransform` into a confit `SqlFunction` that serves the same calls with no Python in the work for each row.
Through sklearn, a fitted transformer costs about 118 µs for each row.
The same query without the transformer costs about 1.4 µs (`benchmarks/bench_transforms.py`, 2026-09-26).

The [native catalog contract](../../packages/sql-transform/spec/native/catalog-contract.md) judges each entry: what it must equal, where the twin raises, and the scope.
The mission is done when the coverage table has no "not yet" row in scope.

## Boundary

**May, without asking:**

- Add or change catalog entries, their family modules, `__init__.py` imports, fixtures and tests under `packages/sql-transform/sql_transform/native/`. Except:
  - changing the twin;
  - registering a parity bound above 0 that no [closed proposal](../../proposals/README.md) covers;
  - widening a registered bound (an ulp ceiling, K, S, τ or a comparison map), unless the same change adds a closed proposal that covers it;
  - changing the rules or the scope of the contract.
- Change the shared catalog machinery `_registry.py`, `_helpers.py`, `_check.py` and `compose.py`. Except: when another claimed task changes the same file, the machinery change is its own task and merges first.
- Regenerate `packages/sql-transform/spec/native/coverage.md` and `bounds.md` with `uv run --no-sync python -m sql_transform.native.coverage --write`. Except: editing them by hand.
- Correct `packages/sql-transform/spec/native/` in place to match merged behaviour, and mark agreed but unbuilt parts **Not built yet.** Except: adding, removing or changing a rule's meaning (that needs a proposal).
- Add or change `benchmarks/bench_native.py`, `benchmarks/native_estimates.py` and the build-estimate coefficients they produce. Except: refitting an estimate to absorb an unexplained slowdown.
- Write this mandate's brief (its **Traps**, and standing instructions that apply a closed proposal or an answer), tracker, task descriptions, audit log and reports. Except: changing this README.
- Add needs to the tracker of [confit](../confit/README.md).
- Write research projects under `research/` for findings that outlive a task.
- Push branches named `claude/native-<name>`, open pull requests from them, and squash-merge them with the expected head SHA once CI is green. Except: force-pushing or rebasing another session's branch; merging a change that fails a required check.
- Start read-only helpers to investigate a twin, review a change, or measure. Except: none.
- Start runbook helpers following the brief's helper procedure and their task description, writing only within their task's branch and the files its description names. Except: a helper never merges, never writes a report, and never edits `packages/confit/`.

**Independent review:** only what AGENTS.md requires.

## Reporting contract

The default, with these additions.

**Added triggers:**

- Coverage crosses a quarter (17, 34, 51 or 68 of 68), or a whole sklearn module goes native.
- A confit capability lands. Count the `NotNative` configurations that it turned native.
- A registered bound grows, or an entry goes back to `NotNative`.
- 15 merged pull requests since the last report.

**Added detail:**

- Start each report with this front-matter block. The maintainer's tracker reads it every 3 hours from master, so keep the keys exact; write `null` for a value not measured, never omit a key.

  ```yaml
  ---
  loop: native
  date: <YYYY-MM-DD>
  master: <short sha>
  previous: <path of the previous report, or none>
  kind: <milestone | short>
  trigger: <the trigger>
  kpis:
    coverage_native: <n>
    coverage_partial: <n>
    coverage_notnative: <n>
    coverage_not_yet: <n>
    nonzero_ulp_bounds: <n>
    families_merged_since_last: <n>
    widest_build_s: <s>
    widest_serve_us_per_64: <µs>
    twin_us_per_64: <µs>
    open_needs_from_confit: <n>
  needs_owner: [<paths of the open questions>]
  needs_from_other_loop: [<need names>]
  ---
  ```

- Start the pull request title with `report:`.
- Under **Where it stands**: coverage from the regenerated table; for each changed family or machinery, entry/twin serving time and measured build/estimate, each measured in one process on a release build; the full catalog parity sweep with `NATIVE_SEEDS=200` and its `NATIVE_SEED_BASE`, with configurations, refusals, failures, skips and answered rows with a non-NULL instance id. A checker overflow is inconclusive, never a pass.
- New and changed findings in one table: number and stable slug, what differed, how found, fixing pull request or open question. Continue the numbers from the previous report.
- Every number cites a script in the repository at the measured commit; data may stay outside.
- Keep the body within 120 lines.

## Files

- [Brief](brief.md)
- [Tracker](tracker.md)
- [Audit log](audit.md)
- [Reports](reports)
