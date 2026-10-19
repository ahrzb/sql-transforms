# Native-transform loop: timeline, process, bugs and rework

> Written by a cheap read-only scout from `local://native-git-log.txt` (full `git log` with PR bodies, 165 commits), loop docs, reports and PR pages.
> **Known limits of this report (added by the supervisor):** the scout paged through only part of the 242 KB log and leaned on the milestone reports; it did not trace rule-file diffs (`local://native-rule-diffs.txt`) to incidents. Treat it as a map, verify claims in the sources. The supervisor-computed metrics in the Appendix are from direct `git`/`gh` runs.

## 1. Timeline

### 2026-10-05: catalog launch, inline start, worker waves and migration

- **The catalog began as code and an oracle-checking loop, before it had a shared `loops/` home.** The source history starts at `972134c8` (#323), "the catalog, swap-the-entry check, and its loop." Confit was split off to supply catalog needs in #324 (`f85ea291`); early confit work added trapping CASE results, struct-field semantics and list outputs (#325–#328).
- **The first implementation batch grew coverage from one to 24 of 68.** #329 (`daff625d`) five scalers; #332 (`b8512980`) `SimpleImputer`/`MissingIndicator`; #340 (`a8399eff`) feature selectors; #343 (`c3edefbd`) `PolynomialFeatures`; #347 (`51f75eea`) encoders; #351 (`5934e12a`) `KBinsDiscretizer`; #356 (`3de6289c`) and #357 (`5a852c86`) Box-Cox `PowerTransformer` and `QuantileTransformer`. Reading 1 records 24/68 on `5513891` and six initial families, 13,933 compared steps and no parity breach (#360). Early work uncovered helper ordering issues, fixture errors and confit bottlenecks (#329, #332, #336–#353; reading 1, §5).
- **Worker-driven operation became explicit before the directory move.** #345 (`9a28e45d`) introduced subagent-driven mode and milestone reports. #352 (`a60ee5ef`) a shared reporting policy. #361 (`a117b99b`) created top-level `loops/` with shared reporting, worker and brief rules. The board then recorded wave 2: T1 `Pipeline`, T2 `FunctionTransformer` (#365, `ae16ebdb`). #366 (`49acad5d`) moved the loop from `packages/sql-transform/docs/native/` to `mandates/native/README.md`, splitting the old `subagents.md` between the shared worker guide and the native README; no behaviour change.
- **The first two readings documented worker-wave outcomes and a regression.** T1/#368 `Pipeline`; T2/#367 exact `FunctionTransformer` functions. Reading 2 on `89e99fc` reports 27/68. It identified confit #363's regression: eager evaluation of shared expressions from CASE arms made a three-instance `QuantileTransformer` only 1.4× faster than its twin (from 25×); confit #377 fixed it, native #378 (`f2ef1848`) adopted confit #374/#375/#377; quantile serving returned to 28×.
- **Wave 3 added compositions and preprocessing.** T6 `SplineTransformer` #384, T8 `ColumnTransformer`/`FeatureUnion` #382, T9 `IsotonicRegression` #381, T10 `FeatureAgglomeration` #380; 30/68 at `87636ef` (reading 3). Confit #387 (`7f4f3be9`) changed shared-value evaluation timing, making spline builds ~4× faster; T13/#396 refitted spline estimates.

### 2026-10-06: rule-setting, edge-case discoveries, waves 4–6

- **Owner rulings made nonzero parity bounds opt-in.** #404 (`5634d6ab`) records rulings on five native decisions, with research: bit-exact by default; nonzero bounds need `to_native(step, allow_bound=True)`. T19/#414 implemented that; T24/#421 added error-scale bounds to `native.check`. Later used by T37/#458 and T41/#494.
- **Wave tickets expanded coverage and test inputs.** T11/#388 per-configuration bounds + transcendentals; T12/#391 quantile from two trees to one; T14/#394 `cbrt`; T15/#399 boolean fixture features; T16/#403 dense `RandomTreesEmbedding`; T17/#402 float32 `KBinsDiscretizer`; T18/#406 periodic spline at ±inf. Reading 4 at `627abf0`: 31/68.
- **A missing fixture edge produced a real regression.** Periodic spline twin returned NaN at ±inf while native returned 0 in some fields. The generator omitted infinity; research for #404 found it, #406 fixed it, T20/#415 added infinities to the generator. Reading 4: 22 of 398 probed configurations differed before #406, none after.
- **Simple-English and scope rules introduced then revised.** #401 (`e2720646`) simple-English skill, glossary, prose checker; #405 (`c02990cc`) removed the prose checker (owner did not want an automatic text check). #408 rewrote the native docs in simple English and opened a worker-environment record. #410 (`627abf0c`) corrected readings 1–3: reports had repeated decision proposals/questions because report-format instructed that content and later reports copied earlier ones. #410 moved it back to decision records, breaking "never edit an old report"; no measurements changed.

### 2026-10-14: loop resumed; guards and cost estimates

- **Owner pause** 2026-10-06 16:11 UTC → 2026-10-14 10:07 UTC (reading 5, #445). T19–T29 landed in 11 PRs: `allow_bound` (#414), ±inf fixtures (#415), spline binding (#416), forest spelling (#420), shared output fields (#418), error scales (#421), input guards (#423), guard-aware estimates (#427), encoder guard (#429), post-#430 estimate refit (#437), probe draws (#439).
- **Input guards and confit interactions entered the contract.** T25/#423 entries trap where twins raise; T26/#427, T28/#437 refit estimates around guard and struct-field costs. Confit #430 avoided rebuilding a trapping struct field at every read. 8,965/140,240 rows where the twin raised, entry trapped on each; guard added median 8% to a 64-row call.
- **Efficiency tickets.** T30/#446 one CASE per tree; T31/#447 batched encoder probes; T33/#450 integer category ranges; T34/#452 estimated-build caps for quantile/isotonic; T35/#454 compact guards; T36/#455 sparse outputs as dense (owner ruling); T37/#458 four linear projections under an error-scale bound. Reading 6 at `98033e8`: 35/68.

### 2026-10-15–17: projections, cluster/kernel families, recurring refits

- T38/#462 zero signs; T39/#463 refit projection estimates after confit #459/#461; T40/#466 refit forest/spline/quantile/isotonic; T41/#494 six centered projections → 41/68 (reading 7).
- T42/#495 bounded outputs in compositions; T43/#497 zero sign + deterministic ARPACK fixture; T44/#498 `x - 0.0` reads; T45/#499 bisection for ordinal bins; T46/#500 refit four estimates, guard reads; T47/#501 four cluster-distance classes; T48/#502 recursion refusal + projection estimate n² term; T49/#503 bit-exact `AdditiveChi2Sampler(sample_steps=1)`; T50/#504 bounded cos/sin; T51/#505 `RBFSampler`; T52/#506 `SkewedChi2Sampler`. Reading 8: 48/68 on `612831d`.
- T53/#508 AdditiveChi2 build cap; T54/#509 `Nystroem`; T55/#510, T56/#511 `PowerTransformer` within owner-ruled scales; T57/#512 bit-exact sparse random projection; T58/#513 `FactorAnalysis` under a cross-host bound. 51/68 (`tickets.md`).

## 2. How work gets done

**Ticket selection and implementation.** Family-per-ticket cycle: top PLANS item, then easiest uncovered class, then configurations left `NotNative` once confit serves the capability. Ticket names classes/configurations, dtype/NaN/platform/order traps and confit dependencies. Worker reads the installed twin, translates op-by-op, adds fixtures per transform-changing configuration, registers parity evidence, updates PLANS (`mandates/native/README.md`). #345 introduced subagent mode; #365 the board.

**Supervisor and review.** `AGENTS.md#helpers-in-cloud-sessions`: inline default, worker-driven supervisor mode; branch + PR per ticket; supervisor reviews, gives feedback, merges; workers never merge. Reviews on GitHub and direct to worker, blocking/optional; serial squash merges after green CI; board/plan/report duties. Native README adds a checklist: twin operation order, fixtures, measured bounds, scope, green gate, generated coverage, confit reproductions.

**Gate and merge.** Regenerate coverage, stage untracked files, pre-commit, `scripts/gate.py`, PR, CI, squash-merge. PR bodies carry extensive parity campaigns (e.g. #513: 14,157 gate passes, 3,059 native tests); 200-seed runs; cross-host experiments for host-sensitive bounds (#513).

**Reports and owner decisions.** Milestone readings, not per PR (#345, #352, #361; `mandates/README.md` triggers and KPI block). Eight readings + one short reading. Decision records for contract/scope; #404 rulings 2026-10-06; #482 (`21b4c149`) closed bounded-composition and worker-environment records 2026-10-15. #410 repaired reports.

**Confit interaction.** Native asks the confit loop with a reproduction; PLANS tracks blocked capabilities. Not one-way: #363 regressed quantile serving, #387 improved spline builds, #430 changed guarded struct reads, #456 caused quadratic field-by-field builds, #459/#461 changed costs again. Native T13, T26/T28, T39/T40, T46, T53 refit estimates or changed behaviour after these.

**Practice vs written rules.** (1) Reports 1–3 duplicated decision content; #410 says report guidance caused it and the fresh-reader test missed it. (2) #410 edited old reports against the rule. (3) T46's estimates measured trial build alone while T47/T48/T51/T52 measured whole warm `to_native`; #508 made docstrings say which. (4) Fixtures missed ±inf and an ARPACK fixture lacked `random_state`; fixes #415/#497.

## 3. Bugs, regressions and rework

| Ticket / PR | What went wrong; stated root cause | Detection | PRs and time to fix |
|---|---|---|---|
| #329 | DuckDB NaN ordering made `x > hi` true where numpy is false; numpy `row_norms` uses einsum order, not left-to-right. | DuckDB-vs-numpy helper tests; wrong order failed 40–50% of rows. | In #329; same day. |
| #332 | Fixture rounded integers after scrubbing the missing marker, so rounding brought the marker back. | While building imputer fixtures. | #332; same day. |
| #347 | All-missing string feature had `[None]` categories, treated as numeric; reused data shapes hid it. | Targeted review test, then investigation of why fixtures missed it. | #347 fixed types + per-configuration seeds; same PR. |
| #353 (confit) | JIT builds leaked two memory maps (~8 KB each); process stalled at OS map limit; no standalone repro. | First catalog sweep failed at 61%. | #353; same day. |
| #363 → #377 (confit) | Eager shared-subexpression evaluation inside untaken CASE arms; quantile 25× → 1.4×. | Remeasurement after #363. | #377; ~1–2 h. |
| #387 → #396 | Confit computed shared values before first reader; spline ~4× faster, old cap too strict. | A/B timings. | T13/#396; same morning. |
| #406 / #415 | Periodic spline 0 instead of NaN at ±inf; fixtures drew no infinities. | Research probe and sweep. | #406 code, #415 fixtures; Oct 6–14. |
| #367 | Function cap used pre-#362 timings: limited linear ops, missed sin/cos slowdown. | Supervisor review remeasured at merged head. | Same PR. |
| #367 | Kernel probe used DuckDB test oracle, which raised except on DuckDB 1.5.5 though newer versions allowed; `to_native` could raise instead of declining. | Review. | Same PR: probe via confit, decline on failure. |
| #410 | Reports repeated owner proposals/questions (report-format asked; each copied the last); fresh-reader checked words, not placement. | Owner/project-thread feedback. | One PR; breaks "never edit old report". |
| T25/#423 → T35/#454 | Guard as per-feature tests joined by OR drove slow builds; per-kind tests cut build 2.4–2.7 s → 0.9–1.2 s, serve ~6.4 → 5.4 ms. | Build/serve bench, guard ablation. | #423, #454 (+#429, #437). |
| #430 → #427/#437 | Struct field reads rebuilt guard tests; estimates fitted before were stale. | Re-measuring builds after #430. | T26/#427, T28/#437; Oct 6 and 14. |
| #452 | Count cap (8,000 quantiles) did not predict cost: 10 × 800 features passed but built in 11.4 s. | Build measurement. | T34/#452: 7 s estimated-build cap. |
| #456 → #461, T39/T40 | Confit made field-by-field struct reads quadratic; forest 3.7 s → 18.1–18.3 s. | Native reproduced with confit alone, bisected commits; reading 6. | Confit #461; native #463, #466; ~8 h. |
| T38/#462 | Balanced sums gave -0.0 vs twin +0.0 on 403 fields; numeric error-scale bound treats them equal. | 200-seed parity. | One PR. |
| T43/#497 | 1×1 `np.dot` kept -0.0 while entry added +0.0; ARPACK fixture used global RNG. | 200-seed parity, cross-process fixture compare. | One PR. |
| T31/#447 | Encoder probing called transform once per category: 4,098 calls, 40.8 s. | Translation timing. | One PR: 130 calls / 1.3 s. |
| T33/#450 | Integer IN list one comparison per category; build 4.6 s. | Encoder timing. | One PR: ranges, 0.74 s. |
| T39, T40, T46 | Estimates stale after confit cost changes; caps rejected work that fit or admitted slow builds. | Timed builds on changed confit. | ≥3 refit PRs, Oct 15–16. |
| T48/#502 | Normalizer l2 with 2,048 features raised `RecursionError` at 1,024 nested additions, not `NotNative`; projection estimate omitted n² list cost. | Width/build experiment. | One PR. |
| T54/#509 | First Nystroem error scale failed far from zero: rounding of squared distance could shift the kernel argument by ≥1. | Research, failures at 11.1 M fields. | Same ticket. |
| T58/#513 | Same-process `cov_z` bound failed on another host's LAPACK: 8 BLAS setups, 2,966/4,480 pairs differ; same-host bound failed 46/80 alternate twins. | Host/BLAS matrix. | Same ticket: independent inverse error bound + conditional refusal. |

**Recurring cause groups** (incidents above; overlapping): numeric/oracle edge semantics 6; fixture/test coverage or determinism gaps 4; engine performance / estimate drift 8; probe/measurement/reporting mismatch 4; bound/owner-contract constraints 3.

## 4. Metrics (scout)

165 path-filtered commits, `972134c8` (#323, Oct 5) → `40f6ab17` (#513, Oct 17). 58 ticket numbers, not all coverage. Coverage per reading: 24, 27, 30, 31, 31, 35, 41, 48 → 51/68 now. Report PRs: #360, #376, #395, #411, #445, #460, #496, #507 (+ short #467). Seven closed native decision records now.

## 5. Observed friction points

- Fixture generators initially missed ±inf, booleans, all-missing strings, deterministic RNG (#347, #399, #415, #497).
- Build estimates repeatedly changed as confit changed lowering, binding, guards, struct reads (#396, #427, #437, #463, #466, #500, #508).
- Struct vs list return shape changed build time a lot (#456, #461).
- Owner rulings constrained approximation (#404, #414, #482, #495).
- Report policy and practice diverged (#410).
- Bench/parity measurements missed output forms or environment variation; later corrected (reading 6 findings 74–77; #513).
- Some refusals came after expensive work or lacked an estimate (Normalizer refusals up to 47 s; AdditiveChi2 had no cap until #508).
- Coordination cost: separate loops, boards, direct feedback, serial merges; owner questions outside the ticket queue (#361, #366; `AGENTS.md#helpers-in-cloud-sessions`).

## Appendix: supervisor-computed metrics (direct `git log` / `gh pr list`)

Commits by category per day (path-filtered, 165 total; categories by subject prefix):

| date | native ticket | native other | confit | report | decision | board | docs/other |
|---|---|---|---|---|---|---|---|
| 10-05 | 2 | 18 | 4 | 3 | 0 | 1 | 7 |
| 10-06 | 20 | 2 | 13 | 4 | 3 | 5 | 5 |
| 10-07 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |
| 10-14 | 9 | 0 | 7 | 2 | 1 | 0 | 1 |
| 10-15 | 4 | 0 | 21 | 4 | 1 | 0 | 0 |
| 10-16 | 12 | 0 | 7 | 1 | 0 | 0 | 0 |
| 10-17 | 6 | 0 | 0 | 1 | 0 | 0 | 0 |

- Of 53 ticket-tagged native commits, 15 subjects are refits of build estimates, caps, guard reshaping or numeric fixes (T13, T26–T30, T34, T38–T40, T43, T44, T46, T48, T53), about 28%.
- 94 native PRs on GitHub; open→merge median **12 min**, p90 50 min, max 135 min. One stale open PR: #432 "native spline: the gate holds a 'zeros' periodic fixture at every degree". Per-PR data: `local://native-pr-stats.txt`.
- 11 commits changed rule files (`mandates/native/README.md`, `AGENTS.md#helpers-in-cloud-sessions`, `mandates/README.md`, `mandates/README.md`, worker briefs, report-format), including #361, #366, #370, #388, #401, #408, #410, `b4b8d033`, #458, #460, #501.
