# Reviews of the sql-transforms native loop

Twelve independent reviews, verbatim, of how the native-transform loop (`sql-transforms/loops/native/`) worked from 2026-10-05 to 2026-10-17, on `sql-transforms` master `40f6ab17`. Four categories, each reviewed by three models: Astra (`review-astra`, openai-codex/gpt-6-astra), Opus (`review-opus`, anthropic/claude-opus-5-5), Muse (`review-muse`, muse-code/muse-wonder-full-1.0). These are opinions with their own citations, not facts checked by this project; see [README](README.md) for what was spot-checked and which reviewer claims were corrected. Paths like `local://row_census.py` refer to session artifacts; the ones that survived are in [scripts/](scripts/).


---

# A correctness — AstraCorrectness

## Verdict

**Keep the twin/oracle design, but strengthen the checks that justify a green result.** The main weakness is not too few random seeds. It is that the loop sometimes treats sampled agreement, excluded inputs, or platform-dependent arithmetic as proof of the contract.

I reproduced three current weaknesses: overflow can make the checker accept unequal answers; the kernel probe ignores zero signs; and a parity run can succeed without comparing a transformed value.

I made no repository changes. I ran targeted reproductions, not the full gate. Below, `native/` means `packages/sql-transform/sql_transform/native/`. Cost and expected-effect estimates are **[INFERENCE]**.

## Top findings

### 1. The checker can accept grossly wrong answers on a platform where `longdouble` is only a double

The approved rule says:

> “check computes S in a way that is safe from overflow.”

That is not true on every supported numerical environment.

**Evidence.** `native/_check.py:62–82` casts values to `np.longdouble`, applies the comparison map, and accepts `ga == gb`. `native/_registry.py:104–114` also computes the bound in `longdouble`. On this Windows environment, I measured `np.finfo(np.longdouble).maxexp == 1024`.

Two direct reproductions:

- `near(1.4e154, 1.5e154, 1.0, np.square)` returned **True**. Both squares overflowed to infinity. An independent `Decimal` calculation gave a difference of approximately **2.9e307**, not ≤1.
- Computing `S = 1e308 + 1e308 + 1e308` in `longdouble` produced infinity. The resulting bound accepted `1e308` against infinity. The mathematical bound for `K=4` is approximately **2.6645e293**, which does not justify that acceptance.

The relevant overflow regression is explicitly skipped when `longdouble` lacks extra range: `native/check_test.py:180–190`. CI has one `ubuntu-latest` job, not a platform matrix: `.github/workflows/ci.yml:13–15`.

**Process cause.** The loop validates the translator’s numerical environment more carefully than the checker’s. Skipping the test records a limitation but leaves the checker issuing successful verdicts.

**Concrete change.**

> “A checker arithmetic overflow is an inconclusive check, never a parity pass. Numerical checker tests must run on platforms with and without extended-range `longdouble`.”

Implement overflow-safe bound and mapped-difference arithmetic in `_check.py` and `ErrorScale.at`. Use scaled arithmetic or a high-precision checking path. Until available, fail explicitly rather than certify affected comparisons. Add the two cases above to `check_test.py`, without platform skips. Run this small checker suite on Linux, Windows, and ARM64 macOS.

**Cost/effect.** About 1–2 engineering days. This fixes a demonstrated false-pass mechanism across all error-scale families, rather than one transformer.

### 2. T58’s measurements do not establish the unconditional contract that the PR claims

The FactorAnalysis PR says:

> “The parity bound of the step holds when the twin runs on another host.”

But it also reports:

> “369,221 fields where one side is not finite and the two differ… the bound does not cover such a row.”

The implementation states the missing assumption directly:

> “the bound assumes that no partial sum overflows.”

**Evidence.** These statements appear in [#513](https://github.com/ahrzb/sql-transforms/pull/513), SHA `40f6ab17`, and `native/project.py:974–1029`. The public catalog contract states the bound without this partial-sum restriction: `packages/sql-transform/spec/native/catalog-contract.md:19–50`. The owner’s ruling explicitly discusses overflow comparisons, rather than granting a blanket exclusion: `proposals/closed/matvec-parity-bound.md:320–327`.

There is a second proof boundary. The FactorAnalysis derivation assumes identical LU pivots across hosts. A fixed margin, `1 + 2^-8`, controls refusal (`project.py:797–802, 884–889, 1010–1015`). The text reports matching pivots in sampled fits. It does not derive why this margin guarantees matching pivots for every admitted matrix.

The inverse-mutation test is useful but does not close that gap. It constructs an alternate inverse **inside its own proposed D**, then verifies that D is needed: `native/project_test.py:650–671`. That tests propagation of the allowance, not whether real alternate inverses always lie inside it.

**Process cause.** The same work defines the bound, identifies exceptions, and reports success after excluding those exceptions. A finite sample can test a derivation; it cannot supply a missing premise.

**Concrete change.**

> “Every bound must list its assumptions and show where each assumption is enforced. An excluded mismatch requires an owner-approved domain restriction. Otherwise it remains a parity failure.”

For FactorAnalysis, require an independent review of the pivot-margin implication and the substitutions in `_factor_prior`. Add adversarial matrices near pivot ties and the conditioning threshold. Preserve every overflow mismatch as a replayable case. Reconcile those cases with the approved contract before describing the entry as bounded for all admitted calls.

Do not “fix” this by trapping on inputs where the twin returns an answer. That would violate a different part of the contract.

**Cost/effect.** Roughly 1–3 days for a numerical review and reduced counterexample corpus; any domain change needs the owner. Expected effect: prevent a large successful campaign from concealing a narrower guarantee.

**Uncertainty.** I have not proved the final FactorAnalysis bound false within its stated assumptions. I found that the published contract is broader, and that the pivot guarantee remains unestablished by the evidence I could inspect.

### 3. Required input classes should be deterministic; random fixture growth repeatedly arrives after the bug

The current instruction says:

> “If the family needs inputs that the row generator does not make, extend the generator.”

That makes the implementer responsible for anticipating the very cases the checker should independently challenge.

**Evidence and the single pre-merge check for each incident:**

- **Periodic spline ±inf:** [#406](https://github.com/ahrzb/sql-transforms/pull/406) documents 0 instead of NaN under periodic extrapolation with `handle_missing="zeros"`. A mandatory infinity row for every applicable configuration would have caught it. Adding more seeds to the old generator could not.
- **Boolean dtype semantics:** SHA `a52bd17e`, #399, records fixes or refusals for encoders, functions, and compositions after boolean fixtures were introduced. A deterministic all-boolean / mixed-boolean-and-double / boolean-with-NULL matrix would have exposed those paths.
- **All-missing strings:** SHA `51f75eea`, #347, records `[None]` categories misclassified as numeric. One all-missing string fit, followed by missing and unseen serving values, would have caught it.
- **Missing-marker reintroduction:** Reading 1, finding 3, records integer rounding restoring a scrubbed marker. A generator invariant—“a `hole=None` numeric column contains no marker after all conversions”—would have caught it without invoking a translator.
- **Reduction order:** SHA `daff625d`, #329, records the einsum ordering issue and helper tests that detected wrong orders. Keep those width-sensitive tests; random normal rows alone are not the specification.
- **Recursion refusal:** SHA `5edd450e`, #502, records `Normalizer(l2)` raising `RecursionError` at 2,048 features. The general fixture generator cannot reach it: `_draw` selects at most 32 features (`catalog_test.py:1139–1144`). A structural-width test would have caught it before merge.

Current `_rows` randomly chooses 40 rows and regimes (`catalog_test.py:1239–1277`). Its `EDGES` includes infinities and zero signs, but not an explicit floating NaN (`:970–973`). NULL often reaches the twin as NaN, but that is not a substitute for testing both representations through the serving boundary.

**Process cause.** “Configuration coverage” and “input-class coverage” are conflated. Counts of seeds obscure which classes never ran.

**Concrete change.**

> “Each fixture configuration must run a deterministic applicable-input corpus before random rows. Record exercised classes and named refusals.”

The corpus should include ±inf, floating NaN, NULL, ±0, subnormals, finite extremes, constants, all-missing fits, and the three boolean layouts above. For branch boundaries, generate the fitted threshold and its two `nextafter` neighbours.

Separately, add a build/refusal suite at reduction-block boundaries and representative large widths, including 2,048. Assert either successful translation or named `NotNative`; never silently skip an unexpected exception. Do not send every huge expression through the expensive DuckDB oracle.

**Cost/effect.** About 2–3 days for the shared corpus and invariants. It directly addresses several observed incident classes and gives future families an independent challenge.

### 4. The kernel probe’s definition of “zero difference” is weaker than the exactness contract

The function translator says:

> “A bounded function whose kernel probe reads 0 on this platform is bit-exact here.”

There are two problems: the probe ignores zero signs, and sampled equality is promoted to universal equality.

**Evidence.** `native/function.py:192–196, 231–241` maps `-0.0` and `+0.0` to the same ordered position. `_bound` then promotes a measured zero distance to an exact configuration (`:265–274`).

I reproduced:

```text
doubles_apart(array([-0.0]), array([0.0])) -> 0
_same(-0.0, 0.0, 0)                     -> False
```

Thus the probe can classify a kernel as exact under a comparison that the actual exact checker rejects.

The finite-sampling issue has a historical warning in this same loop. Reading 4, finding 54, records that `exp(uniform)` probe inputs made accurate logarithm kernels agree and concealed differences. The current `_random_doubles` fixes that sampling defect (`function.py:199–208`), but does not turn a finite sample into proof.

**Process cause.** A platform diagnostic has become a certification mechanism. Two implementations of “difference” also encode different zero semantics.

**Concrete change.**

> “A probe used to authorize exact serving must compare zero signs. A zero sampled maximum is evidence of agreement, not proof of kernel identity.”

Use one shared comparison policy for probes and `check`. Add a mutant that changes only `-0.0` to `+0.0`; an exactness probe must reject it. Add a mutant that changes one input outside the fixed probe set, to demonstrate the certification limitation.

For default exact serving, require established operation/kernel equivalence. Where only empirical evidence exists, retain the approved nonzero bound or refuse exact serving. This would change the documented probe-based exactness policy, so it needs owner approval.

For T38/#462 and T43/#497, retain the dedicated zero-sign regressions in `project_test.py:100–110, 293–349`. Those were quality defects within an approved nonzero bound, not failures that the ordinary bounded comparator was supposed to reject. Track zero-sign differences separately rather than silently strengthening that contract.

**Cost/effect.** Half a day for consistent zero handling and mutations. Kernel-equivalence policy requires additional review. This prevents demonstrated false exactness classification.

### 5. A green parity campaign needs evidence that it exercised the computation, and an independent twin

The checker promises:

> “Where both trap on every row, nothing is compared and `check` raises.”

A NULL instance ID bypasses this protection.

**Evidence.** `_check.py:304–330` increments `compared` for a successful NULL-ID response. `_rows` deliberately generates NULL IDs (`catalog_test.py:1241–1242`).

I exercised the real `check` with a fitted StandardScaler:

- One NULL-ID row returned **1 compared row**.
- One NULL-ID row plus one known-ID infinity row, rejected by both sides, also returned **1**.

Neither run compared a transformed numeric answer. These runs check useful NULL/trap semantics, but do not establish translation parity.

There is also no generic independent-twin snapshot in the main catalog test. It calls `to_native(step)` and then checks that same `step` (`catalog_test.py:1291–1303`). The rule “Never change the twin” is therefore primarily a review rule. **[INFERENCE]** An accidental in-place translator change could contaminate both the reference and the produced expression.

**Process cause.** The campaign counts query outcomes rather than distinguishing numeric evidence, NULL propagation, traps, and refusals. The reference remains reachable by the implementation under test.

**Concrete change.**

> “A catalog parity case must compare a non-NULL-ID answer for every served instance. Count NULL answers and matching traps separately.”

Keep valid NULL-only tests, but do not count them as numerical parity evidence. Add separate counters rather than changing every caller’s meaning of `check` silently.

In the catalog harness, preserve an independent estimator copy before translation. Check against that copy and assert that translation leaves fitted state unchanged. Add a translator mutant that modifies a coefficient in place. It must fail.

Keep and extend the existing negative controls: reordered StandardScaler (`catalog_test.py:1835–1858`), dropped terms (`check_test.py:139–147`), and guard mutations (`guard_test.py:442–450`). They already test whether the checker can detect known errors.

**Cost/effect.** About one day. This closes a demonstrated empty-evidence path and protects against a future shared-reference error.

### 6. Cross-host and 200-seed evidence should be reproducible checks, not external PR experiments

T58’s investigation is valuable. Its operational form is too fragile to protect the next family.

**Evidence.** [#513](https://github.com/ahrzb/sql-transforms/pull/513) used eight configurations of **one machine**, varying OpenBLAS kernels, thread counts, and NumPy dispatch. Its scripts and data are cited under `/mnt/project-files/transforms-loop/t58/`, which was unavailable here.

I reproduced the denominator arithmetic: `80 × 8 × 7 = 4,480` ordered comparisons. These represent 2,240 unordered pairs, not 4,480 independent host comparisons. The reported 2,966 differences are 66.2% of ordered comparisons. I could not reproduce that numerator, the 46 failed fits, or the reported maxima without the underlying artifacts.

The main CI has one Linux environment. The nightly workflow runs confit’s campaign, not the native fixture campaign. The catalog campaign always uses the prefix `range(SEEDS)` (`catalog_test.py:1283–1288`), with eight seeds by default. The same 200-seed prefix is repeatedly reused.

The ARPACK incident was a separate reproducibility problem, now addressed well: SHA `9f74bc0d`, #497, fixed `random_state` and added the all-configuration test under two global RNG states (`catalog_test.py:1306–1337`).

**Process cause.** Exploratory campaigns discover important failure dimensions, but those dimensions are not consistently transferred into retained, repeatable gates.

**Concrete change.**

> “Cross-host evidence must replay identical fitted state and rows. Re-fitting independently on each host does not test the same twin.”

Retain a compact trusted corpus and a runner in the repository. Record the commit, dependency versions, loaded BLAS library/kernel, thread count, array layout, and checker precision. Compare the row-local twin as the contract reference. Report batch differences separately.

Use a small PR matrix and a larger scheduled matrix. Include real OS/architecture variation, not only OpenBLAS dispatch overrides. Verify the requested dispatch actually loaded. Keep the standard regression seeds, and add a recorded rotating seed window for discovery.

For #367’s version-dependent probe failure, retain `test_a_failing_kernel_probe_leaves_the_function_python` (`catalog_test.py:2030–2044`). Add a public-runtime smoke lane with an allowed newer DuckDB version, separate from the pinned test oracle.

**Cost/effect.** About 2–4 days for the runner and CI wiring; recurring cost depends on corpus size. It would make T58-style failures and future dependency surprises discoverable without another bespoke investigation.

### 7. “Never raise a bound” is not enforceable as written—and freezing K alone would enforce the wrong thing

The README says:

> “Never raise a bound to make a failure pass.”

The worker brief still says to measure a maximum over 200 seeds and “register that.” It does not include the README’s later distinction between measured evidence and a derived error-scale K (`mandates/native/brief.md:39–46`; `README.md:70–84`).

**Evidence.** `_registry.Entry.bound` checks a per-estimator value against its current class ceiling (`native/_registry.py:198–213`). It does not compare that ceiling with the previously approved ceiling.

More importantly, the allowed error is a function of **K, S, τ, the comparison map, and the admitted domain**. [#509](https://github.com/ahrzb/sql-transforms/pull/509) corrected Nystroem’s S while retaining K. The PR reports that the earlier scale failed on 11.1 million fields. This was a reasoned mathematical correction, not evidence that a K-only lock would protect the contract.

**Process cause.** The rule attempts to constrain intent—“to make a failure pass”—without requiring a reviewable record of what changed in the acceptance predicate.

**Concrete change.**

> “Any widening of the acceptance predicate requires an explicit bound-change review. Include K, S, τ, comparison maps, exactness classification, and domain exclusions.”

Add an approved-bound manifest with decision references and named numerical implementations. CI should flag changes to those implementations and show old/new effective tolerances on a fixed adversarial corpus. Require independent review for a widening and owner approval when it changes the contract. Retain the original failing case.

Update the worker brief to distinguish empirical ulp evidence from derived error-scale bounds.

This is enforceable as a **change-control rule**. No generic check can prove that arbitrary Python formulas never widen a bound, or determine the author’s motive. Sampled tolerance comparisons are useful alarms, not proof of semantic equivalence.

**Cost/effect.** About 1–2 days. It makes legitimate corrections visible while preventing quiet relaxation through S, exclusions, or configuration classification.

## What is working and should be kept

- **Two comparisons separate translation errors from engine errors.** `check` compares entry against twin, and confit’s entry against its DuckDB definition (`_check.py:184–224, 290–303`).
- **Review #367 challenged the implementation against the actual merged runtime.** It found the test-only Oracle dependency and stale assumptions, then required a failure-to-`NotNative` regression. [Review evidence](https://github.com/ahrzb/sql-transforms/pull/367#issuecomment-6000587862).
- **Exact-by-default serving and restricted bounded composition are sound safeguards.** They avoid pretending a small numeric error cannot change a downstream branch.
- **The loop already uses negative controls and reusable repairs.** The reordered-operation test, dropped-term test, and generic RNG determinism test are better than one-off successful examples.
- **T58 tested the twin against alternate twins.** That is the right experiment for host-sensitive constants. Preserve its method in a repeatable harness.

## Disagreements with the report

1. **“Another host’s LAPACK” overstates T58’s experiment.** The PR describes alternate OpenBLAS setups on one machine. This supports sensitivity to dispatch and threads, not independent OS/architecture coverage.

2. **“Zero parity failures” needs denominators and exclusions.** T58 reports 369,221 unequal nonfinite fields outside its bound’s assumptions. Those cannot disappear from the contract assessment.

3. **The signed-zero incidents were not ordinary bounded-parity breaches.** T38 explicitly says its differences were “within the bound.” T43’s zero correction likewise improved agreement beyond the numeric bound. The report should distinguish contract failures from permitted but undesirable differences.

4. **The blanket “200-seed” description hides different experiments.** T58 reports 200 seeds for three catalog fixtures, but 44 seeds for its larger numerical campaign. T54 reports 16 seeds for its large campaign. Those are not interchangeable coverage claims.

5. **Some historical numbers remain claims, not reproduced measurements.** I checked their PR/source provenance and T58’s denominator arithmetic. I did not reproduce the large campaign numerators or maxima. The integer-marker incident is supported by Reading 1, but I found no independent PR explanation beyond that report.

## Out-of-category observation

Performance: #367 shows why measurements must identify the merged engine revision; a green pre-merge benchmark can become stale without any family-code change.

---

# A correctness — OpusCorrectness

## Category A verdict: correctness and verification

The twin method is sound: the step is compared against a one-row sklearn twin, and the native SQL is also checked against its own DuckDB definition. Most incidents were bugs the parity check could have caught, but its comparator, its row generator or its fixed seeds hid them. Since 6 October no second agent has checked any merged numerical claim. The rule "never raise a bound" depends only on discipline, and its wording does not cover the error scale S.

---

### 1. Measured: the gate passes many steps on NULL-id rows alone, so the T20 "compared no row" guard almost never fires

**Problem.** `_rows` draws each row's instance id from `[None, *instances]` (`catalog_test.py:1242`). With 1–3 instances, about 25–50% of rows have a NULL id. `check` counts those rows as compared (`_check.py:303`, `:321`), even though both sides just answer NULL. The guard added in T20/#415 ("where the twin raises on every row, the rows proved nothing") therefore only fires if no row has a NULL id, which is rare.

The generator also draws serving values with no regard to the fitted domain: uniform(-1e3, 1e3), and strings from `VOCAB + UNSEEN` (3 of the 9 strings are unseen). It never replays the instance's own fit rows. So configurations whose twin raises outside the domain hardly ever get a row the twin answers.

**Evidence.** I measured the gate's seeds 0–7 with the twin on this host, using `local://row_census.py`, `local://dead_guard.py` and `local://per_config.py`:
- 85,440 rows in total. 34,111 (40%) have a NULL id and 6,128 raise.
- **37 steps get through `to_native`, and `check` passes them while comparing only NULL-id rows.** By class: OneHotEncoder 13, Pipeline 9, OrdinalEncoder 5, Spline `extrapolation="error"` 5, Isotonic `out_of_bounds="raise"` 4, MissingIndicator 1.
- Rows with an id that the twin answers, out of 320 per configuration across all 8 seeds:

| Configuration | Answered rows |
|---|---:|
| default `OrdinalEncoder()` | 18 |
| default (sparse) `OneHotEncoder()` | 21 |
| `OneHotEncoder(sparse_output=False)` | 29 |
| `OneHotEncoder(drop="if_binary")` | 12 |
| `SplineTransformer(extrapolation="error")` | 19 |

These are the defaults of the most-used classes.

**Root cause in the process.** The generator was grown by adding input classes (booleans in #399, ±inf in #415). No one ever measured what each configuration actually compares. #415's own numbers showed the dilution: "1,661 of 56,320 rows hold ±inf, and the twin answers 179 of them". The same arithmetic applies to finite edges: an "edges" row avoids inf with probability 0.925^n, which is 0.29 at 16 features and 0.08 at 32. [INFERENCE] For families that reject inf, almost every wide edges row raises, so -0.0, 1e±300 and 5e-324 rarely reach the comparison.

**Change.**
- In `check`, count only rows that have an id and that the twin answers. Return those counts.
- In `test_an_entry_matches_its_twin`, aggregate across seeds and assert a floor, e.g. "≥ 64 answered id rows per configuration over the gate's seeds".
- In `_rows`:
  - set the NULL-id rate to a fixed 5%;
  - split `edges` into `finite-edges` and `nonfinite` regimes;
  - add a `replay` regime: the instance's own fit rows, each paired with its own id, and each value nudged by `np.nextafter`.
- Add to the README, step 5: "A fixture configuration must compare at least N answered rows over the gate's seeds; the parity test asserts it."

**Cost and effect.** About 1 ticket. Every class's draws change, as with T15, so re-run the 200 seeds once. The default encoders then get real coverage from the 8-seed gate instead of only at milestone runs. This also stops a future configuration from passing on NULL rows alone.

### 2. The comparator ignores the sign of zero for every bounded entry, while the "twin's sign" policy is checked only by off-tree scripts

**Problem.**
- `ulp_distance` puts -0.0 and 0.0 at the same position (`_check.py:36`), and `near` returns True on `ga == gb` (`_check.py:75`). `check_test.py:207` pins this behaviour with the comment "Signed zeros are one value".
- So for every entry with ulps > 0 or an `ErrorScale`, a sign flip is invisible to the gate and to each PR's 200-seed measurement.
- Yet the loop's policy is "each entry gives the twin's sign where it can" (reading 7:738–740, reading 8:1072–1074). Only `signed_zero.py`, `signed_zero_r7.py` and `signed_zero_r8.py` enforce it. These live in `/mnt/project-files/transforms-loop/` and are not in git. `git ls-files` finds only two research scripts.

**Incidents.**
- T37/#458 claimed 200 seeds and merged after 7 minutes. Reading 6's off-tree script then found 403 fields with the wrong sign (finding 77), which cost ticket T38.
- T41/#494's docstring said the entry always answers a zero with the twin's sign. Reading 7 found 54 FastICA fields that do not (finding 81), which cost ticket T43.
- The report's "Detection: 200-seed parity" for T38 and T43 is wrong. The parity test passed both times, by construction.

**Change.**
- In `check`, before the bound comparison, add: if `x == 0.0 == y` and `math.copysign(1, x) != math.copysign(1, y)`, raise `ParityError`. The exception is an entry that declares `signed_zeros="either"` with a cited reason.
- Change `check_test.py:207` to expect this.
- Add to the README, step 6: "A bound covers magnitude only; the sign of a zero is always exact unless the entry declares otherwise."

**Cost and effect.** About 20 lines. Reading 8 found no sign-only difference across 17 bounded classes and 200 seeds, so the false-positive cost is about zero. [INFERENCE] At 8 seeds, T37's rate (403 per 200 seeds) gives about 16 expected failures, so T37 would have failed its own gate. Two tickets and three throwaway scripts would have been saved. The same promotion rule should cover T58's `measure.py` and the fixture digest: "a script that produces a finding is committed as a test or under `native/tools/` before the fix merges".

### 3. Since 6 October no second agent has checked a bound derivation, although every past check of that kind found errors

**Problem.** Since #410, ~50 merged native PRs show up under one branch name, `claude/project-thread-2pg6vk`. In the subset I pulled with `gh` (#406–#513), every inline PR has 0 comments and 0 reviews. Bounded PRs merged 7–25 minutes after opening:

| PR | Ticket | Minutes to merge |
|---|---|---:|
| #458 | T37 | 7 |
| #494 | T41 | 12 |
| #501 | T47 | 8 |
| #509 | T54 | 13 |
| #510 | T55 | 9 |
| #511 | T56 | 8 |
| #513 | T58 (409 new lines of Higham-based bound) | 12 |

Independent checks did find things when they ran:
- The one worker-mode review I read (#367) caught two blocking bugs. Its kernel probe used the oracle, so `to_native` would raise rather than refuse on DuckDB versions newer than 1.5.5. Its caps were based on stale timings.
- The adversarial verifier for #404 refuted measured constants: SkewedChi2 2.77 against a claimed ≤ 2.5, and Nystroem 1.58 against 0.83 (`decisions/research/2026-10-06/distances.md:246–247`).
- The supervisor's adversarial probe of #382 established finding 40.

T54 then replaced the ruled Nystroem S with a new S (`kernel.py:782–787`: "the research's S fails"). It edited no decision record and had no reviewer.

**Root cause.** `workers.md` §1 makes inline mode the default and attaches review only to the worker-driven mode. The native checklist ("Check one configuration against the source…") has nobody to run it in inline mode.

**Change.** Add to the native README under "Tickets and review":

> In inline mode, a PR that adds or changes a bound above 0 (ulps, K or S), `_check.py`, `_registry.py`'s guard, or the fixture generator gets a fresh-context verifier subagent. It is given only the diff, the installed twin source and the decision record, and is asked to (a) re-derive K/S, (b) list input classes the PR's measurement did not draw, and (c) run the PR's measurement on new seeds. Its verdict is posted on the PR, and the merge waits for "no blocking".

**Cost and effect.** About 15 of 58 tickets would qualify, so about 15 subagent runs. [INFERENCE] Such a verifier would likely have flagged T41's false claim and T54's change to a contract term.

### 4. "Never raise a bound" is not enforceable as written

**Problem.**
- Bounds are code literals, and the tests that pin them are edited in the same PR (`power_test.py:330,333`, `catalog_test.py:2184`).
- The rule covers ulps and K but not S, which is the term T54 changed.
- Step 6 still says to measure the largest ulp distance and "Register that ulp bound". The bound then *is* the sample maximum: `function.py:158–170` (cbrt 3, log10 2) and #356 ("Bound 4: the largest distance … measured").
- The #404 verifier showed that measured maxima get breached on independent data. A measured-max bound is therefore the setup most likely to need raising. The README already forbids this for K ("Do not use the measured value as K") but still prescribes it for ulps.

**Change.**
- Extend `coverage.py --write` to emit a committed `bounds.md` manifest. Per fixture configuration at seed 0, record: the ulps, `K(est)`, and the maximum and a hash of the per-field `K·eps·S + τ` over fixed probe rows.
- Add a CI step that fails when any manifest value grows compared with the base, unless the PR also changes a `decisions/closed/*.md` that names the class.
- Rewrite step 6 so that an ulp bound above 0 needs a derivation, or a runtime probe that *is* the bound (as `kernel_distance` already is). Measurements count only as evidence.

**Cost.** About 100 lines plus one CI step. This makes the rule mechanical and catches S changes like T54's.

### 5. The gate cannot see the host, width or thread claims, and those bugs were found only by off-tree experiments

**Evidence.**
- CI runs on `ubuntu-latest` only (`ci.yml`).
- `_one_blas_thread` pins BLAS to 1 thread for every test (`catalog_test.py:114`).
- Widths are capped at 32 features (`_draw`). Nystroem's tests "keep to few components" because `check`'s definition grows as m²·n (#509).
- The owner ruled "the approved forms hold on another host" (`matvec-parity-bound.md:413`). The same record says that same-host is "all that `check` can test" (`:378–379`).
- T58's cross-host failure (46 of 80 transformers, up to 2.9e8×) was found by an 8-setup `OPENBLAS_CORETYPE` matrix *on one machine*. glibc version and OS were never varied.
- T48's `RecursionError` at 2,048 features was found by a width experiment. The fix catches `RecursionError` only around `SqlFunction(...)` (`_registry.py:742`).
- On this Windows host, `np.longdouble` is 64-bit (checked). So the "no overflow in long doubles" premise of `near` and S fails here. The `_registry` docstring names only macOS arm64.

**Change.**
- A nightly native workflow, modelled on `nightly-campaign.yml`, which today is confit-only. It runs the bounded families' parity under:
  - default settings;
  - `OPENBLAS_CORETYPE=Sandybridge`;
  - `NPY_DISABLE_CPU_FEATURES=AVX512F,AVX512_SKX` with 4 BLAS threads (no `threadpool_limits`);
  - plus one cross-process job: translate under setup A, then run the twin and `check` under setup B.
- A width-ladder contract test: one fixture per class at 64, 256, 1,024 and 2,048 features, asserting that `to_native` either returns or raises `NotNative`, with no other exception.

**Cost and effect.** One workflow and one test. T48 would have been caught by the ladder, and T58's first bound by the cross-process job.

### 6. The seeds never change, so evidence and validation reuse the same sample

**Problem.** `SEEDS = int(os.environ.get("NATIVE_SEEDS", "8"))` with `range(SEEDS)`. The gate always runs seeds 0–7, and every "200-seed parity run" re-runs seeds 0–199. Bound evidence is often measured on those same seeds: `function.py:160–162` cites "NATIVE_SEEDS=200 of the catalog's fixtures … cbrt reaches 3", and 3 is the bound. The milestone run that "confirms" it is the same sample. New random cases appear only when the generator changes (T15, T43). Confit's nightly uses fresh seed ranges (#484: 31000000–31049999).

**Change.** Add a `NATIVE_SEED_BASE` setting and run the nightly in item 5 from a date-derived base. Step 6 should require that bound evidence comes from seeds outside 0–199, or from a dedicated stress generator.

**Cost.** 2 lines plus the nightly.

---

### What is working and should be kept
- **The twin and the self-check.** The twin is the one-row step (`_udf.py:317`, `est.transform([vals])`), which avoids T58's batch-size-dependent zero sign. `check` also verifies confit against DuckDB's own run of the native definition (`_definition`).
- **Bit-exact comparison uses `repr`.** -0.0 and NaN handling are correct at ulps=0 (`_check.py:57`).
- **Derived K, not measured K.** The research, the verifier and T58's Higham-based bound with a conditional refusal are the strongest work in the history.
- **Runtime platform probes.** `kernel_distance`, `bspline_is_scipys` and `interp_is_numpys` refuse instead of guessing.
- **Promoted determinism test.** `test_a_seed_draws_the_same_step_under_any_global_random_state` is the right pattern: a reading-time finding (82) became a test.
- **Self-tests of `check`.** `test_a_reordered_translation_is_caught` and #367's mutation check are good starts. Make mutation general: for each family, add `+0.0`, scale one field by (1+eps), or drop the NaN arm, and require the gate to fail. That would have exposed findings 1 and 2.

### Disagreements with the report
- **T38 and T43 "Detected by 200-seed parity": wrong.** The parity test passes sign flips by design. The off-tree `signed_zero*.py` scripts found them.
- **The report omits that review stopped.** §2 describes supervisor review, but no inline native PR I sampled (#406–#513) has a single comment or review.
- **T20/#415 "check fails on no row" is presented as a fix.** It is effectively dead: 37 steps pass on NULL-id rows alone.
- **T54 "Detection: Research".** It was T54's own measurement (11.1 M fields past K with the research's S). The fix changed a ruled contract term without the owner, and the report does not say so.
- **T58 "failed on another host's LAPACK".** The evidence is 8 setups *on one machine* (`OPENBLAS_CORETYPE`, numpy without AVX-512), not a second host. glibc and OS were not varied.
- **"Fixture/test coverage gaps 4" undercounts.** It misses the generator drawing values without regard to the fitted domain (item 1) and the fixed seed set (item 6).

### Out-of-category, one line each
- From #410 on, inline mode reused one branch name, `claude/project-thread-2pg6vk`, for ~50 PRs, against the owner ruling "Use one branch for each ticket" (`workers.md:39`).
- The repeated build-estimate refits (T13, T26, T28, T39, T40, T46, T53) are a confit-coupling cost, not a correctness one.
- Stale open PR #432.

The measurement scripts are in `local://row_census.py`, `local://dead_guard.py` and `local://per_config.py`. They ran on this Windows host. [INFERENCE] Whether a row raises depends on category membership and validation, not on host numerics, so the row classification should match Linux.

---

# A correctness — MuseCorrectness

# Category A — Correctness and verification

**Verdict:** The oracle machinery (`check`: swap-the-entry + DuckDB definition cross-check + trap-iff-raises) is genuinely strong. Almost every escaped bug was *outside what the generated rows, the comparator, or the host could see* — and each fix patched the instance (one more edge value, one more fixture, one more bespoke test) rather than the blindness. The loop is one generic generator upgrade, one comparator fix, and one CI job away from catching most of its historical bug classes pre-merge. I verified the load-bearing claims against code, commits, and PRs; uncertainties are marked.

---

## 1. Top findings

### 1. The comparator is blind to the sign of zero under any nonzero bound — two shipped bugs it structurally cannot see

**Evidence.** `_check._ordered` maps `-0.0` and `0.0` to one position ("-0.0 and 0.0 share a position", `packages/sql-transform/sql_transform/native/_check.py:34-38`), so `ulp_distance(-0.0, 0.0) == 0` at any ulp bound > 0. `near()` compares absolute differences, and `check_test.py:207` pins this as intended: `(0.0, -0.0, 0.0, None, True)` — "Signed zeros are one value." Only the bit-exact path (`repr(a) == repr(b)`, `_check.py:57`) distinguishes them, matching the contract's "-0.0 and 0.0 are different answers" *only* "with a bound of 0" (`catalog-contract.md`, claim: entry-same-answer).

Consequences shipped twice. T38/#462 (`bea1372e`): "Where every product of an output field was -0.0, the balanced sum … answered -0.0 and the twin 0.0: 403 output fields over seeds 0 to 199 … **within the bound**." T43/#497 (`9f74bc0d`): the same class again, 54 FastICA fields, opposite mechanism (1×1 `np.dot` preserves -0.0, entry added 0.0). Both were found by an offline sign-only scan of a 200-seed run's outputs — the parity run itself *passed*. Sign-of-zero is not cosmetic for a catalog whose owner ruling (2026-10-06) exists because bounded entries "can change a prediction": `1/-0.0 = -inf`, and any downstream comparison/branch on sign flips.

**Process root cause.** When error scales were introduced (T24), "within bound" silently replaced "same answer" for zeros too; nobody re-decided signed zero for bounded families, so `near`'s absolute-difference form decided it by accident. The fixes then added bespoke sign tests in `project_test.py` only (lines 102, 296, 341) — `kernel.py`, `distance.py`, `power.py` bounded families have no generic equivalent.

**Concrete change.** In `_check.check`, in both the ulp>0 and error-scale branches, add: if `x == 0.0 and y == 0.0 and math.copysign(1, x) != math.copysign(1, y)` and the exact-zero case applies (for scales: the S-part of the bound is 0, i.e. all products exactly zero), raise `ParityError`. The exact-zero restriction matters: T38's commit shows the *underflow*-to-zero sign is BLAS-kernel-dependent ("-0.0 in 213 shapes"), so a blanket sign assertion would be cross-host flaky; the exact-product case is deterministic (numpy answered `0.0` in 855/855 shapes). Add the sentence to `mandates/native/README.md` step 6: *"A bound above 0 does not license the other signed zero where every product is exactly zero; `check` asserts the sign there."*

**Cost/effect.** ~30 lines + one test; retrospectively catches T38 and T43 in the gate at 8 seeds (both had dozens of fields), and covers every future bounded family for free.

### 2. The generators are fit-blind and magnitude-narrow; every edge class was added after it bit

**Evidence.** Serving rows (`_rows`, `catalog_test.py:1239-1273`) are drawn independent of the fitted estimator: plain values are `uniform(-1e3, 1e3)`; regimes are plain/edges/nulls/small/ints. `EDGES` (line 971) is eight values: `[0.0, -0.0, 1e-300, -1e300, 1e300, 5e-324, inf, -inf]` — no NaN as a served double (NULL→NaN is the only NaN path), no `DBL_MAX`, no `-5e-324`, no 2^52-boundary values, and nothing derived from the fit (knots, bin edges, tree cutpoints, centres, `mean_`). Fit data (`_fit_matrix`, lines 1003-1078) is similarly tame: normal means in ±100, sd ≤ 50, integers ±1000, `n` ∈ [5, 60], features ≤ 32 (line 1126) and ≤ 4 in 70% of draws.

The escape history maps one-to-one onto these blind spots:

- **±inf**: EDGES had no infinities; periodic spline answered 0.0 where the twin answers NaN. Found not by the gate but by #404 decision research; fixed in #406 (`36d4eab4`: 22 of 398 configurations parted), generator fixed 8 days later in #415 (`a7cd5aa8`).
- **All-missing string feature**: #347 (`51f75eea`) — `[None]` categories read as numeric; "A targeted test caught this"; also exposed that all configurations shared the same 8 data shapes (fixed by per-variant seeds in the same PR).
- **Booleans**: the generator drew none until T15/#399, *after* the encoders had to decline boolean features for lack of fixtures.
- **Far-from-zero fits/rows**: T54/#509 — the research-derived Nystroem error scale fails "where the rows and the fit sit far from 0": 11.1M fields past K, up to `9.5e286·eps·S` (`kernel.py:804`). It was caught by a bespoke 354M-field campaign using means of 1e8–1e9 (`kernel.py:792-793`) — magnitudes `_fit_matrix` (±100) and `_rows` (±1e3 plain, ±1e300 edges which trivially overflow the kernel to 0/inf) cannot produce. The dangerous band is the middle: offsets ~1e4–1e8 where `eps·x² ≈ 1` moves the `exp` argument by ≥1.
- **Exact-zero products** (finding 1): need `x == mean_` exactly — probability ≈ 0 under continuous draws; surfaced only as a statistical residue over 200 seeds.
- **Domain boundaries for `positive` fixtures**: `_rows(..., positive=True)` applies `abs()` to every serving value including EDGES (line 1263), so catalog-level Box-Cox/AdditiveChi2/SkewedChi2 fixtures never serve a negative row; trap-where-twin-raises for `x <= 0` rests solely on bespoke `guard_test.py`/`power_test.py` cases.

Meanwhile the *proven* antidote exists only as per-family handiwork the next worker must remember to repeat: `_knot_rows` for splines, `test_trees_at_the_cutpoints` ("Rows at and beside the cutpoints are in…" `catalog_test.py:786`), `_wide_rows` with `10**uniform(-300,300)` in `project_test.py:448-459`.

**Process root cause.** README step 5 says "extend the generator (*Do not add a special case for one entry*)" — good rule, but in practice each family added its adversarial rows to its own `*_test.py`, and the shared generator only grew when a bug escaped into a reading. There is no checklist item asking "what are this family's discontinuities, and does a served row sit on ±1 ulp of each?"

**Concrete change.** Two generic additions to `catalog_test.py`, applied to every fixture step in `test_an_entry_matches_its_twin`:
(a) **Fit-derived rows**: collect all floats from fitted attributes (`vars(est)` names ending `_`, flattened, per instance, recursively for compositions via the existing `_runs`); for each numeric feature serve rows cycling these values, their `nextafter` ±1 ulp, negations, and ±0.0. This generically places rows on knots/bin-edges/cutpoints/centres/means without per-family code.
(b) **Magnitude/offset ladder**: port `_wide_rows` into `_rows` as a sixth regime plus offset rows `c + plain` for `c ∈ {1e4, 1e8, 1e12, 1e16}` (same offset in fit via one `_fit_matrix` kind: `normal(1e8, 1)`). Add `math.nan`, `sys.float_info.max`, `-5e-324`, `2.0**52 ± 0.5` to `EDGES`. For `positive` fixtures, append one row per feature with `-1.0` and `-0.0` (check already asserts trap-iff-raise).
Checklist line for `mandates/native/README.md` review section: *"☐ The PR names the family's discontinuities (thresholds, knots, domain edges, exact-zero cases) and each is hit by the fit-derived rows or a named test."*

**Cost/effect.** ~150 lines once; [INFERENCE] roughly doubles catalog-test rows (40 → ~80/step) and gate time for that file. Retrospectively catches #406 (fit-derived knot rows at ±inf serve the periodic arm — ±inf already in EDGES now, but the *next* family's unlisted edge is the point), T54's scale failure (offset regime), T38/T43 (x == mean_ rows at seed 0, given finding 1), and the #347 class (fit-derived categories include the all-missing case when combined with existing holes).

### 3. "Holds on another host" is a contract requirement with zero automated evidence; on Windows/macOS-ARM the bound arithmetic silently becomes vacuous

**Evidence.** CI is a single `ubuntu-latest` job (`.github/workflows/ci.yml`); there is no OS/BLAS/thread matrix anywhere, and the nightly workflow runs confit's fuzz campaign, not native seeds. The gate actively suppresses one variance axis: `catalog_test.py:114-118` pins every test to one BLAS thread (`threadpool_limits(limits=1)`). Twin, fit, and entry are always compared within one process on one BLAS.

T58/#513 (`40f6ab17`) shows what that hides: the first FactorAnalysis bound, read off the translating host's `cov_z`, "held on each setup for the twin of the same setup" but "in 46 of the 80 fitted transformers, it failed for the twin of another setup, by up to 2.9e8 times"; 2,966 of 4,480 cross-setup `cov_z` comparisons differ in bits. The shipped fix is exemplary numerical work (host-independent Cholesky + Higham bounds), but note what the evidence actually is: 8 setups **on one machine** via `OPENBLAS_CORETYPE` and thread counts — no second host, and no CI job that would notice if a future family's bound is same-host-only. The same exposure exists, unmeasured, for every BLAS-order family already merged under MATVEC/CENTERED scales (T37, T41): their derivations claim host-independence, but nothing re-runs them under another kernel.

Separately, the overflow-safe bound arithmetic depends on 80-bit `longdouble` (x86-64 Linux). `_registry.py`'s ErrorScale docstring concedes: where a long double is a double (named: macOS arm64; **also Windows** — verified on this machine: `np.finfo(np.longdouble).maxexp == 1024`), "an S past DBL_MAX bounds nothing." Worse than "bounds nothing": `ErrorScale.at` then returns `inf`, and `near()` returns `abs(ga-gb) <= inf == True` for any finite pair — `check` silently passes arbitrarily wrong answers exactly on the huge-magnitude rows where bounds matter most. The one test guarding this, `test_an_error_scale_past_dbl_max_still_bounds`, is `skipif`'d off on precisely those platforms (`check_test.py:181-183`).

**Process root cause.** The 2026-10-06 ruling required bounds that "hold on another host," but the loop operationalized it as *derivation style* (prose in docstrings) rather than as a *recurring measurement*. T58's author did the measurement by hand because FactorAnalysis recomputes `cov_z` per call made the risk obvious; families whose host-dependence is less visible got no such campaign.

**Concrete change.** (a) One nightly CI job (extend `nightly-campaign.yml` or a new `native-hosts.yml`) running only the bounded families' tests (`project_test.py kernel_test.py distance_test.py power_test.py check_test.py` + `catalog_test.py -k "PCA or FactorAnalysis or Nystroem or …"`) under a 4-cell env matrix on the same runner: `OPENBLAS_CORETYPE=HASWELL`, `=SANDYBRIDGE`, `OMP_NUM_THREADS=4`, `NPY_DISABLE_CPU_FEATURES=AVX512F` — the exact axes T58 varied manually, no new hardware. (b) In `_check.check`'s scale branch, fail loud: if `not np.isfinite(bounds[j])` and `repr(x) != repr(y)` and neither side is infinite, raise `ParityError("…bound overflowed on this platform (long double is a double)")`. Add a `check_test` case monkeypatching `EPS`/S to overflow so the failure is pinned on Linux too.

**Cost/effect.** CI YAML ~1 hour; check fix ~10 lines. Retrospectively, (a) catches the first T58 bound pre-merge and continuously audits T37/T41/T54/T55 derivations; (b) converts a silent vacuous pass into a named failure for any Windows/macOS user running `native.check`.

### 4. "Never raise a bound" is unenforceable as written — and for ulp bounds the prescribed procedure *is* raising it

**Evidence.** Bounds live as bare literals and callables in family modules: `function._BOUNDS = {exp:1, log:1, …, cbrt:3}` with `ulps=max(_BOUNDS.values())`, and `ErrorScale(k=_k, …)` functions per family. `translates()` (`_registry.py:259-298`) validates only that a per-estimator bound sits under its own ceiling — the ceiling itself is whatever the PR says. There is no baseline artifact (contrast `coverage.md`, which is generated and diff-checked by `coverage_test`), so a diff changing `cbrt: 3 → 5` or `K = n+5 → 2n+5` is indistinguishable from any other numeric edit, and the review checklist item ("measured over at least 200 seeds and cites the measurement") is satisfied *by* the raise.

Worse, README step 6 for ulp bounds instructs: "measure the largest distance in ulps over at least 200 seeds. Register that ulp bound." That is bound-by-measurement: any failure mechanically produces a larger measurement and a passing registration. The "Never raise a bound to make a failure pass; first find the order" sentence that follows is pure exhortation — nothing in the gate can tell a found-order 0 from a measured-up 3. Error scales have the right discipline on paper (derived K, measurement only as evidence), but the derivation is prose: the *research* derivation for Nystroem's S was wrong by up to `9.5e286·eps·S` (T54) and was only caught because that worker ran an unusually adversarial campaign (finding 2). No test asserts *margin* (measured max ratio vs K): a derivation that is wrong-but-lucky passes identically to a right one. T54's PR helpfully reports "least room 1.99 against K = 11" — as prose, once, by hand.

**Process root cause.** Bounds are the catalog's correctness contract with the owner, but they're stored with less change-control than the generated coverage table.

**Concrete change.** (a) Generate `packages/sql-transform/spec/native/bounds.md` (or extend `coverage.py`) from `catalog()`: per class — bit-exact / ulp ceiling + per-function table / scale K-formula evaluated on a canonical estimator + the decision-record link. Pin it with a `coverage_test`-style test ("the bounds table is current"). Add to the README review checklist: *"☐ A diff in the bounds table is blocking unless it cites a closed decision record or a derivation change in the module docstring."* That makes "never raise" a visible, reviewable event instead of a vibe. (b) Each bounded family's `*_test.py` gains one margin assertion over its existing adversarial rows: observed `max |entry−twin|/(eps·S) ≤ K/2` (families pick the factor once, from their campaign's least-room figure). A future derivation error that eats the margin fails before it breaches.

**Cost/effect.** (a) ~half a day including the generator; (b) ~15 lines per bounded family (5 families). Effect: bound raises become auditable owner-visible events; margin tests would have flagged the research S (ratio ≫ K) in the gate had the rows existed (depends on finding 2's offset rows).

### 5. `to_native`/`check` raising the *wrong exception* was caught twice by human review, never by a test — and one instance is still live

**Evidence.** The contract is best-effort: untranslatable ⇒ return unchanged / `NotNative` under `strict`. Two escapes of the "raised something else" class:

- **#367** (verified in the PR's supervisor review): the sin/cos kernel probe went through `confit.oracle.Oracle`, which raises `RuntimeError` unless DuckDB is exactly 1.5.5, while sql-transform allows `duckdb>=1.5.5` — "on any newer DuckDB, `to_native` on `np.sin` raises instead of returning `NotNative`." Caught by the supervisor reading the code, fixed by probing confit and declining on any exception, plus a monkeypatch test — for that one probe.
- **T48/#502** (`5edd450e`): `Normalizer(l2)` over 2,048 features raised `RecursionError`, not `NotNative` (1,024-deep addition chain, confit emits SQL recursively). Found by a width experiment. The generator never draws it: features ≤ 32, outputs ≤ 1,000 (`catalog_test.py:1126, 979`).

The fixes are local try/excepts (`_registry.py:742-753` catches `RecursionError` around `SqlFunction`; `_builds` catches `ValueError|RuntimeError`). The class remains open: translator-internal recursion (deep `Pipeline` nesting through `compose.py`, balanced-tree builders), other exception types from confit, and — still live today — **`native.check` itself**: `_check._definition` opens `Oracle()` (`_check.py:195`), which raises `RuntimeError` on any `duckdb != 1.5.5` (`packages/confit/confit/oracle.py:84-95`), while `packages/sql-transform/pyproject.toml` permits `duckdb>=1.5.5`. The exact pattern review removed from the probe survives in the verification entry point the contract advertises as evidence. It is masked because the repo's dev pin is `duckdb==1.5.5` (root `pyproject.toml:16`).

**Process root cause.** "Raises only NotNative" is a stated contract with no property test; shape/width space in fixtures is capped for build-time reasons (legitimately — but then the refusal path at uncapped shapes is untested by construction).

**Concrete change.** One parametrized test in `catalog_test.py`, seed 0, no serving rows needed: for every catalog class, fit the narrowest fixture configuration, then call `to_native(strict=True, allow_bound=True)` on synthetic steps of widths {1, 33, 256, 2048, 8192} features (fit on tiny random data directly, bypassing `_draw`'s caps) and `Pipeline`s nested {10, 200} deep; assert the outcome is a function or `NotNative` — any other exception (explicitly including `RecursionError`, `MemoryError` excluded) fails. Plus fault injection generalizing #367's monkeypatch test: patch `DuckDBInferFn` to raise, assert every probed family (function kernels, encoder probes, guard `_probed`) declines with `NotNative`. Separately, decide the Oracle question in one line in `catalog-contract.md`: either `native.check` requires the pinned test DuckDB (say so, and make sql-transform's `duckdb>=1.5.5` a `==` for check users) or `_definition` should probe through confit like `kernel_distance` now does.

**Cost/effect.** ~60 lines, seconds of runtime (translation only). Retrospectively catches both #367 and T48 before review; prospectively it's the generic net for "environment/shape makes us raise wrong."

### 6. The gate runs 8 seeds; the 200-seed evidence base is post-merge, manual, and partly not in the repo

**Evidence.** `SEEDS = int(os.environ.get("NATIVE_SEEDS", "8"))` (`catalog_test.py:1000`); `scripts/gate.py` runs plain `pytest`. The 200-seed sweep is a *milestone report* duty (`mandates/native/README.md#reporting-contract:108-109`), i.e. after merges, and per-PR 200-seed campaigns are worker-run and self-reported in PR bodies (e.g. #513's "14,157 gate passes" — counts no reviewer can reproduce). T54's board entry states where its bound campaign lives: "**Scripts and data are in `/mnt/project-files/transforms-loop/t54/`**" — outside the repo. Findings 77 (T38), 81/82 (T43, ARPACK) all surfaced at 200 seeds in readings, meaning the discovery mechanism for residual classes is "the next milestone report," with the originating worker's context gone. Compounding thinness: `_step` silently redraws up to 20 times discarding unfittable/refused shapes (`catalog_test.py:1117-1122`), so the 8 gate seeds are a biased subset whose rejection rate nobody reports.

**Process root cause.** Seed count was tuned for gate wall-time (comments at lines 334-335, 974-978 are explicit cost accounting — reasonable), but the compensating control (a scheduled deep run) was assigned to humans writing reports instead of CI, unlike confit, which has `nightly-campaign.yml` filing issues automatically.

**Concrete change.** Add a nightly job mirroring confit's pattern: `NATIVE_SEEDS=200 pytest packages/sql-transform/sql_transform/native/catalog_test.py` on `ubuntu-latest`, red night ⇒ issue. Require bound-campaign scripts to land in-repo (e.g. `packages/sql-transform/sql_transform/native/campaigns/<family>.py`, excluded from the gate) — one README line in step 6: *"The measurement's script is checked in; a bound whose evidence cannot be rerun from the repo is not registered."* Also have `_step` count redraws and have the milestone report print the skip/redraw rate per class (data already computed; currently discarded).

**Cost/effect.** CI minutes nightly [INFERENCE: 25× the catalog file's gate time, tens of minutes]; converts findings-77/81/82-class discoveries from milestone archaeology into next-morning failures, and makes every registered bound's evidence reproducible by a reviewer.

### 7. Oracle sensitivity is proven for exactly one of 51 classes

**Evidence.** The sole mutation test is `test_a_reordered_translation_is_caught` (`catalog_test.py:1835`): swap `x/s` for `x·(1/s)` in StandardScaler, assert the bit-exact check notices — over 8 seeds, requiring only that *some* seed catches it (`assert caught`). For bounded families, sensitivity evidence is a synthetic `_Dot` dropped-term test in `check_test.py:139`. Nothing demonstrates that the generated fits+rows would distinguish, say, two swapped fitted coefficients, an off-by-one bin edge, or the wrong centre in a real family — the failure mode behind #347 (partition of categories wrong; the encoder translator *probes the twin itself*, so a wrong class-predicate is exactly what rows must expose) and behind every "reused data shapes hid it" incident. Relatedly, `check` serves the twin through the same confit query path as the entry (`_serve(sql, rows, step)` vs native), so a common-mode marshaling error (e.g. bool→double, NULL handling in the serving layer) is invisible to parity; `_definition` cross-checks confit-vs-DuckDB for the *entry* only. (`_answers` calls `step(...)` directly, but only for the ARPACK determinism test.)

**Process root cause.** "The check would catch it" is argued per-family in PR prose ("with the arm removed, every fixture parts" — the unmerged #432 did this properly) instead of being a standing generic test.

**Concrete change.** A nightly (or seed-0 gate) generic mutation test: for each fixture configuration, deep-copy the fitted step, perturb one fitted float attribute (first float in each `*_` array: multiply by `1+2^-40` and, separately, swap two entries where width ≥ 2), translate the *mutated* step, and `check` it against the *original* twin on seed-0 rows (+ fit-derived rows from finding 2) — must raise `ParityError`, with a named allowlist for provably-no-op attributes (unused statistics). Add to `check` an optional direct-call cross-check on the first row batch: served-twin answers vs `step(iid, *features)` — kills the common-mode concern for ~20 lines.

**Cost/effect.** ~80 lines + allowlist churn per new family (bounded, informative). This is the one generic check that catches the *next unknown* insensitivity: if rows can't see a coefficient, they can't see its mistranslation.

---

## 2. What is working and should be kept

- **Swap-the-entry + definition check**: serving the identical query through confit-twice plus DuckDB-runs-the-definition (`_check.py:185-230, 276-390`) is a real two-engine oracle; `check` raising when it compares **no** row (#415) closed a vacuous-pass hole most projects never notice.
- **Trap-iff-twin-raises as a first-class parity axis** in `check`, with the guard probe (`_probed`, `_registry.py:523`) deriving rejection sets empirically per estimator/container rather than restating sklearn validation rules.
- **ARPACK determinism test** (`test_a_seed_draws_the_same_step_under_any_global_random_state`, T43): the model response to a bug class — a generic test over *all* fixtures that makes the whole category (implicit global RNG) unshippable. Findings 5–7 above ask for the same move applied to widths, hosts, and sensitivity.
- **Kernel probe done right**: probing the serving engine (post-#367), declining on any probe failure, and `_random_doubles`' random-significand draws after the chi² research showed `exp(uniform)` draws make two accurate `log` kernels agree vacuously (`function.py:197-206`).
- **Derived-K discipline for error scales** (measurement as evidence, never as K) and T58's execution of it; `allow_bound` opt-in keeping bit-exact the default.
- **Per-variant seed streams** (#347) and per-configuration fixtures enforced by `test_every_entry_has_fixtures`.

## 3. Disagreements with `local://native-loop-report.md`

1. **§2 "Gate and merge … 200-seed runs" is unsupported.** The gate (`scripts/gate.py`, default `NATIVE_SEEDS=8`) never runs 200 seeds; 200 is milestone-report duty and manual PR campaigns (finding 6). This matters because the report's own bug table repeatedly lists "200-seed parity" as the *detection* mechanism — a mechanism that does not run pre-merge.
2. **§3, T38/T43 "Detection: 200-seed parity" misattributes the catch.** Both commit messages state the answers were *within the bound*; parity passed. Detection was an ad-hoc sign-only scan of run outputs. Crediting "parity" hides the comparator blindness (finding 1) that is the actual lesson.
3. **§3, T58 "failed on another host's LAPACK / Host/BLAS matrix" overstates the evidence.** #513's own body: 8 setups on "this machine" via `OPENBLAS_CORETYPE`/threads/numpy feature flags. No second host was involved, and none is in CI now. "A bound that holds on another host" is a derivation claim with single-host emulated-kernel evidence — good evidence, but not what the table says.
4. **§3 treats #367's Oracle-probe defect as closed.** The same pinned-version Oracle remains in the verification path itself (`_check._definition` → `oracle.py:88-95`), so `native.check` raises `RuntimeError` on `duckdb>1.5.5`, an environment sql-transform's metadata allows (`duckdb>=1.5.5`). [INFERENCE: masked in practice by the repo's dev pin `duckdb==1.5.5`.]
5. **Appendix's stale #432 deserves its weight in §3, not a footnote.** The mutation-proven spline gate fixtures ("With the ±inf arm removed, every one of the five fixtures parts from its twin", `76eed1f7`) are **not on master** (verified: not an ancestor of HEAD; PR #432 still OPEN). Master covers the ±inf periodic case via one bespoke parametrized test — fine — but the report presents #406/#415 as the complete fix while the sensitivity-proving part sits unmerged since 2026-10-06.

*Minor, re the task inputs rather than the report: the native package lives at `packages/sql-transform/sql_transform/native/`, not `…/src/sql_transform/native/` as the review brief stated.*

## 4. Out-of-category issues noticed (one line each)

- Category B?: ~28% of ticket commits are build-estimate refits after confit cost changes (report Appendix) — the estimate-refit treadmill wants an automated recalibration job, not tickets.
- Category ?: `check` runs the DuckDB definition with `TRAP_QUERIES = 3` individually-queried trapped rows and the rest batched — a trapped row past the third that DuckDB *answers* while another in its batch traps is masked [INFERENCE from `_definition` batching logic]; worth a deliberate test.
- Category ?: T54 notes `check`'s definition grows as m²·n (25 MB SQL, DuckDB OOM at 64×100), capping Nystroem test widths — verification cost, not translator cost, is now the binding constraint for wide families.

**What I am uncertain about:** the exact gate-time cost of findings 2/6 (estimated, not measured — I did not run the suite); whether the underflow-zero sign variance (finding 1) can be characterized tightly enough to assert beyond the exact-product case; and whether any merged MATVEC/CENTERED family would actually fail under `OPENBLAS_CORETYPE=HASWELL` — that is precisely the measurement finding 3 asks CI to make.

---

# B process — AstraProcess

## Verdict

**Keep the loop, but change how it integrates shared changes, reviews inline work, and produces evidence.** The strongest demonstrated review happened in early worker waves. Later work became predominantly inline, without the same visible review record. Meanwhile, repeated performance calibration and increasingly large reports consumed effort that the current metrics cannot quantify.

I reproduced the supplied PR population: **94 PRs, 93 merged, one open; median open→merge 11.58 minutes, p90 50.38 minutes, maximum 134.78 minutes.** These numbers measure neither development time nor review quality.

I inspected the conversations on #367, #381 and #384, plus several later PRs and #432. I did not run tests or benchmarks or change files. Runtime figures below are attributed to their recorded measurements, not presented as measurements I repeated.

## 1. Top findings

### 1. Shared changes reach master before their catalog-wide costs are understood, creating repeated calibration work

The rule that families are “almost independent” is true of many translator modules, but not of their performance dependencies.

The clearest native-side incident is **T44 → T46**. T44 changed numeric reads across translators. [PR #500](https://github.com/ahrzb/sql-transforms/pull/500), commit `c06a15a4`, then reported:

> “Wide steps with such a guard built in up to about four times the time.”

It repaired guard reads and fitted four families’ estimates again. Its grids contained **18 + 66 + 35 + 113 = 232 fitted builds**. This was not merely an engine upgrade requiring routine recalibration: a catalog-wide optimization had introduced a catalog-wide cost.

The engine-side equivalent was confit #456. Reading 6 records forests building in **42–44 seconds instead of 5.9–6.5 seconds**, followed by confit #461 and native refits #463/#466. See `mandates/native/reports/2026-10-15-reading-6.md:705-712` and commits `318575b2`, `fbbcd249`.

**Process cause:** family tickets are the scheduling unit, but shared lowering, guard and output-shape changes invalidate several families simultaneously. The gate checks correctness, while broad cost discovery often waits for a reading. “Refit the estimates” then becomes the standard response.

**Concrete change — add to the native review checklist and cross-loop delivery rule:**

> Changes to shared reads, guards, output construction or confit lowering must run the affected catalog performance cases before merge. Compare old and new code on the same machine. Include struct field reads, list outputs, multiple instances, and shapes immediately below and above existing caps. Explain regressions before fitting new coefficients.

Start with the actual past failures: T44’s guarded spline and ordinal encoder; #456’s wide forest; #363’s multi-instance quantile. Use the existing native benchmark infrastructure, with a small fixed integration set rather than another full survey. **Reject correctness regressions; require explicit disposition of performance trade-offs. Do not simply refit around them.**

**Cost/effect:** [INFERENCE] Medium implementation cost, then several targeted measurements per shared change. It should replace some multi-family refit tickets and retrospective bisections with earlier detection. It will not eliminate legitimate calibration after engine improvements.

### 2. Review was real in worker waves; the inline mode does not preserve that protection

The 12-minute median does **not** establish rubber-stamping. Early conversations contain substantive independent work:

- [#367](https://github.com/ahrzb/sql-transforms/pull/367): the supervisor remeasured performance, rejected stale caps, identified a DuckDB-version-dependent probe failure, and checked the fixes.
- [#384](https://github.com/ahrzb/sql-transforms/pull/384): the supervisor read sklearn, measured builds, and required a build cap and scipy platform probe.
- [#381](https://github.com/ahrzb/sql-transforms/pull/381): the supervisor verified that the worker was right and the ticket’s description of scipy’s operation was wrong.

The later evidence is different. **All 48 merged PRs on `claude/project-thread-2pg6vk` have zero GitHub comments and zero formal reviews** in the queried population. This includes shared-machinery #498/#500 and the substantial bound implementation in [#513](https://github.com/ahrzb/sql-transforms/pull/513).

For #513, GitHub records CI running **12:30:05–12:40:53**, followed by merge at **12:42:22**. Most of its 12-minute open interval was CI. No independent review is recorded. This does not prove that no private review occurred.

**Process cause:** `AGENTS.md#helpers-in-cloud-sessions:152-155` requires review of “every worker PR,” whereas inline mode keeps the work in one context. Review protection is tied to execution mode rather than change risk. The on-demand review workflow only runs after `/review`.

**Concrete change — amend `AGENTS.md#helpers-in-cloud-sessions`:**

> Inline work needs a fresh reviewer when it changes a parity derivation, refusal boundary, shared machinery, or cross-host assumption. The reviewer records the reviewed head, source checks, attempted counterexamples, and unresolved findings on the PR. CI success is not review evidence.

Keep small, mechanically checkable inline fixes inexpensive. For #513-type work, give the reviewer the contract, diff and reproduction commands—not merely the author’s explanation.

**Cost/effect:** [INFERENCE] One additional review session for high-risk changes, not every ticket. This preserves the demonstrated benefits of #367/#384 without imposing worker orchestration on routine work.

### 3. The reporting workflow repeatedly discovers defects in its own measurement tools

The report says:

> “Another person must be able to take the measurements again from this table alone.”

Yet much of the measurement machinery lives outside the repository, and important mistakes recur:

- Reading 6, finding 74: an 11-commit comparison actually measured **eight commits with the earlier compiled confit**.
- Finding 75: the bounds script omitted list elements, **29% of values**.
- Reading 7, finding 83: a table reused the preceding reading’s row counts.
- Reading 8: T50 seeded rows with Python’s process-dependent string `hash`, so the measurement could not repeat exactly.

Sources: `mandates/native/reports/2026-10-15-reading-6.md:709-710`, `2026-10-16-reading-7.md:232-236`, and `2026-10-17-reading-8.md:316-320`. The latter report’s reproduction section lists numerous external scripts and orchestration chains at lines 1435–1462.

**Process cause:** measurement is treated as disposable report preparation rather than maintained product infrastructure. Each reading remeasures extensively, but its scripts can introduce new errors outside the ordinary review and gate.

**Concrete change — alter `report-format.md` and the benchmark tooling:**

> Recurring measurements use one versioned runner. Every result records source revision, loaded extension path and checksum, environment, deterministic case identifier, return shape, and counts of compared fields. Reports render counts and tables from those results.

Give that runner three specific regression checks:

1. A synthetic scalar/struct/list case verifies that every output field is counted.
2. Two fresh processes produce identical case identifiers and fixture digests.
3. A cross-revision run refuses results unless its loaded native extension matches the recorded build.

For historical A/B builds, use separate build directories and record provenance. Do not rely on copied source timestamps.

**Cost/effect:** [INFERENCE] Medium, concentrated tooling work. It should reduce repeated script invention, restore confidence in comparisons, and prevent the specific reporting errors above. It also makes independent review materially cheaper.

### 4. “Easiest uncovered row” and class counts do not measure the value of the next ticket

The picker says:

> “Take the easiest ‘not yet’ row of coverage.md.”

The scoreboard gives equal credit for a class with substantial limitations and one broadly usable in compositions. It gives **no additional class credit** for important configuration work.

For example, reading 6 records **20 new sparse configurations across eight classes** from T36, while class coverage benefits came from T37’s four projections. Conversely, the bounded-composition ruling permits bounded steps in any position, but only the first slice has shipped. The current plan still lists affine propagation, weights, mapped bounds and discontinuous successors as remaining work.

Sources: `mandates/native/README.md:51-56`; `mandates/native/reports/2026-10-15-reading-6.md:623-628`; `mandates/native/tracker.md:53-63`; [#495](https://github.com/ahrzb/sql-transforms/pull/495). After T42, another **16 tickets, T43–T58**, shipped while that remainder stayed open. This is not proof that their priority was wrong. It shows that the class total cannot justify the ordering.

Likewise, “about 28% refits/fixes” is a count of selected commit subjects, **not 28% of time, tokens, or waste**. Those subjects include useful probe corrections and configuration expansion.

**Process cause:** the picker has no explicit comparison between a new class, expanded existing coverage, an end-to-end composition, and maintenance. The records lack the effort denominator needed to optimize “coverage per unit effort.”

**Concrete change — add a short selection statement to each ticket:**

> State whether this ticket adds a class, expands configurations, completes a composition, repairs correctness, or changes cost. Name the cases it enables. State why it precedes the highest-priority alternative.

Keep class coverage, but accompany it with a **stable representative configuration/composition set** and its native/refused outcomes. Record ticket start, ready-for-review, merge, blocked intervals and execution mode. Import session cost when available; otherwise mark it unknown. Do not use PR-open time as agent effort.

**Cost/effect:** [INFERENCE] Low documentation cost and a modest scoreboard extension. It makes zero-class-gain work visible and permits a later evidence-based decision about parallelism and maintenance spend. It does not require inventing a weighted “value score.”

### 5. Reports have become research monographs, increasing generation and reading costs without a corresponding owner-facing benefit

The reporting policy correctly says:

> “A loop never writes one report for each PR.”

But reporting has still expanded substantially. I counted the current eight full readings: **5,925 lines and 70,595 whitespace-delimited words**. Reading 8 alone contains **1,470 lines and 18,378 words**. Reading 1 had 373 lines.

This is not all waste. Readings found real integration defects. The problem is coupling those campaigns to a very large owner-facing publication. The template requires every bound, every need, complete scoreboards, reproduction details and repeated measurements. Full parity sweeps alone are reported as taking roughly **40–60 minutes** in readings 5–8, before the other experiments and writing.

[#410](https://github.com/ahrzb/sql-transforms/pull/410) provides a direct process failure:

> “Each reading copied the shape of the reading before it.”

That propagated proposals and questions into three reports. The fresh-reader check addressed wording, not whether the content belonged there.

**Process cause:** one document serves as experimental notebook, comprehensive state snapshot, hand-off and owner update. Repetition increases both agent context and the opportunities for stale content.

**Concrete change — split the publication contract, not the safety campaign:**

> A milestone’s owner summary contains the KPI delta, newly enabled behavior, regressions, decisions needing attention, and next action. Limit it to about 80 lines. Link the full measurement artifact and prior unchanged findings instead of repeating them.

Preserve full integration campaigns when shared code or dependencies change. A prose publication alone should not force rerunning an already applicable result: permit reuse only with explicit revision, dependency and environment provenance. Keep #410’s decision-record separation.

**Cost/effect:** [INFERENCE] Low policy/template cost, medium automation cost if combined with finding 3. It should reduce writing and review effort substantially. **Actual owner reading time is unknown**; the tracker may already shield the owner from most of this text.

### 6. Pause/resume and board reconciliation are incomplete; #432 is an orphan, not evidence of an unfixed production bug

The worker guide promises:

> “The state lives in GitHub … and in the files of the loop.”

Yet [#432](https://github.com/ahrzb/sql-transforms/pull/432) remains open with no conversation since October 6, while the current board’s in-flight table is empty (`mandates/native/tracker.md:10-11`). Below that empty table are lengthy results for completed tickets.

Importantly, **do not infer that periodic infinity handling remains untested**. The repair is already merged. The current `catalog_test.py:2324-2355` has a dedicated infinity regression test across seven degree/knot pairs. #432 adds broader generator coverage. Its claimed extra gate cost is seven seconds, but that old timing needs reassessment before adoption.

The main pause lasted **185 hours 56 minutes** according to reading 5. That interval is owner-requested inactivity, not evidence of an autonomous decision bottleneck. The worker-environment question was also addressed by #428 and subsequently closed by owner ruling.

**Process cause:** the board is maintained by the active session’s narrative rather than reconciled against GitHub. Restart rules do not ensure that work from another session receives a disposition.

**Concrete change — add a startup/resume rule:**

> Before choosing new work, reconcile open native PRs with the board. Every open PR needs an owner and next action. Adopt it, mark its blocker, or close it with a superseding reference. Do not discard work merely because it is old.

For #432 specifically: compare its additional cases with current dedicated tests, then adopt the useful coverage or close it with that explanation. Keep completed results in their existing PRs/reports; the board should contain live work and a short hand-off.

**Cost/effect:** [INFERENCE] Low: a GitHub-to-board comparison at startup and before a wave. It prevents lost work and unnecessary reconstruction after pauses without requiring owner intervention.

### 7. The wave policy refreshes every open branch after every merge, even when only the next candidate needs integration

`AGENTS.md#helpers-in-cloud-sessions:182-185` and `mandates/native/README.md:180-183` require:

> “After each merge, the other open PRs merge master in.”

This creates avoidable repeated coordination. In [#384](https://github.com/ahrzb/sql-transforms/pull/384), the worker merged master twice, resolved common-file conflicts, and regenerated coverage each time. Its initial conflict prevented CI from running. [#381](https://github.com/ahrzb/sql-transforms/pull/381) also required the supervisor to merge master and regenerate coverage before merging.

For four simultaneously open independent PRs, the blanket policy requires up to **3 + 2 + 1 = 6 branch-refresh events** as the wave drains, before unrelated confit changes. That is a structural count, not a measured time saving.

**Process cause:** freshness is maintained eagerly on all branches rather than at the merge boundary. Shared board, plan and generated files amplify that work.

**Concrete change — change the merge rule:**

> Merge serially. Refresh the next merge candidate against current master and regenerate its coverage before final CI. Refresh another worker early only when the merged change affects its implementation or evidence.

Continue to notify every worker of relevant engine changes. Keep green CI on the final candidate and expected-head protection. Do not weaken integration to avoid conflicts.

**Cost/effect:** [INFERENCE] Low policy cost. It should reduce duplicate merges, regenerated files and cancelled CI work in worker waves. It will have little effect during the later inline phase.

## 2. What is working and should be kept

- **Evidence-based supervisor review:** #367, #381 and #384 show source checks, independent measurements, corrected ticket assumptions and concrete blocking feedback.
- **One executable gate shared by local work and CI:** `scripts/gate.py` runs Rust tests and pytest with one exit status; `.github/workflows/ci.yml` invokes it directly.
- **Cross-loop reproductions:** reading 6’s confit-only reproduction helped isolate #456 rather than obscuring the regression with native changes.
- **Readings as integration investigations:** they found problems that individual tickets missed. Reduce their publication burden, not their independent scrutiny.
- **Explicit owner decisions:** parity opt-in and composition rulings have named records. #410’s separation of decisions from reports is correct.
- **Refusal rather than pretending coverage:** named `NotNative` cases make incomplete support visible. The process should measure those cases, not penalize honest refusals.

## 3. Disagreements with the supplied report

1. **The review statistic is accurate but cannot support a rubber-stamp conclusion.** The reproduced median rounds to 12 minutes. Early reviews demonstrably changed implementations. Later inline PRs lack comparable public review evidence. These are different operating modes, not one homogeneous review population.

2. **The report overgeneralizes supervisor review to the whole period.** The 48 comment-free project-thread PRs require a separate account of inline review. Private review remains unknown.

3. **The “28%” figure is not an effort or waste estimate.** I found the 15 named ticket IDs. A history query including the original native documentation directory finds 54 ticket-tagged implementation commits, including T18, giving 27.8%. The percentage remains approximately right, but its denominator depends on the history filter.

4. **The supplied “full” path-filtered log omits the actual #406 repair commit.** Repository history contains `36d4eab4`, “native spline … (T18) (#406)”; the supplied log contains references to #406 but not its commit entry. This explains at least one count difference and cautions against using the map as complete history.

5. **The #406/#415 repair did not extend to October 14.** #415’s metadata and commit `a7cd5aa8` place the infinity-generator change on October 6. October 14 is a later measurement/resumption date.

6. **“Stale #432” is correct, but “missing regression protection” would not be.** A dedicated merged test already covers the failure. Its unresolved contribution is broader generator coverage and the absence of a disposition.

7. **No supplied evidence establishes tokens per class, active agent hours per ticket, or actual owner attention.** PR duration, subject counts and report size are useful indicators, but none is a substitute for those quantities.

## 4. Out-of-category observations

- **Numerical correctness:** #513’s cross-host inverse bound and pivot-stability assumption deserve a mathematical review; I did not validate the derivation.
- **Coverage semantics:** “51/68 native” includes bounded and configuration-limited classes while compositions are counted separately; this is not full end-to-end coverage.
- **Engine design:** native workarounds such as `x - 0.0` depend on confit’s sharing heuristics; evaluating the appropriate engine abstraction belongs to the architecture review.

---

# B process — OpusProcess

## Category B: process and workflow

**Verdict.** On correctness, the biggest process problem is that **pre-merge review stopped on 2026-10-06 and nothing replaced it**. The figures that show this: of the native PRs from #406 to #513, 51 merged with zero review comments and zero review objects. The last supervisor review is #403. The "12 min median open→merge" is how long CI takes, nothing more. Since then the milestone readings have done the reviewing, one batch later, and they take about a quarter of active time. The second problem is throughput. Picking has narrowed to a list the loop keeps writing for itself, and that list skips cheap classes.

### 1. Top findings

**F1. Inline mode has no review step, and all inline PRs merged unreviewed.** The 12 min is CI wait time.
- **Evidence:**
  - `gh pr list` with comment and review counts: every PR from #405 on has 0 comments and 0 reviews. The last review is #403, which notes "the native loop now runs from a project thread". From #407 on, every native PR shares one branch, `claude/project-thread-2pg6vk`, against the "one branch for each ticket" ruling in `AGENTS.md#helpers-in-cloud-sessions` §1.
  - `gh run list` shows CI taking 8–11 min per PR (for example #513: 12:30→12:40 UTC). So open→merge time is just CI.
  - Before this, review was real. On #367 the supervisor re-measured caps on the merged head and found that `kernel_is_duckdbs` raises on any DuckDB newer than 1.5.5. On #384 it blocked spline builds of 21–46 s with no cap. On #396 it rejected 38 s added to the gate. On #399 it ran its own sweep of 32,255 steps.
- **Escapes the old review would likely have caught.** Each needed a follow-up ticket:
  - T49/T50 merged `AdditiveChi2Sampler` with no build cap. Reading 8 caught it (finding 91: 28.5 s), and T53 fixed it. This is the same defect #384's review blocked.
  - T44 measured serve time only, and its reads slowed builds. Spline went from 4.3 to 9.2 s and `OrdinalEncoder` from 2.7 to 11.2 s. T46 fixed this the same day (board text at `af89daed`; the T44 body at `eeefa315` has no build numbers).
  - T37 produced −0.0 where the twin gives +0.0. Reading 6 caught it (finding 77), and T38 fixed it.
- **The riskiest work now has no second check.** These are bound derivations that no closed record contains.
  - T58's `K = n + k + 1`, theta margin and pivot-ratio refusal were derived by the loop. `matvec-parity-bound.md` only names FactorAnalysis (line 10). README step 6 says to "use the K that the record derives".
  - T54 swapped in its own S after "the research's S … fails at 11.1 million fields".
  - Both merged in about 12 min with no one else reading them.
- **Root cause.** `workers.md` attaches review to the *mode*: in inline mode the definition of done is just gate, lint, PLANS and merge. It should attach to the *risk* of the change. The move to a project thread changed the mode without anyone recording what was lost.
- **Change.** Paste into `workers.md` §1 and the native README step 7:
  > "In both modes, a PR that adds or changes a translator, a parity bound, a build estimate, or `_registry`/`_helpers`/`_check`/`compose` gets one independent review before merge. Comment `/review` on the PR, using `.github/workflows/claude-review.yml` (it already allows the loop's bot), with the native checklist pasted in. Or run a read-only local subagent that gets only the diff, the twin's installed source and the checklist. Answer each blocking item before merging. If K or S is not stated in a closed record, the reviewer derives K from the twin's source without reading the PR's derivation. The record also gets an addendum listed under 'Needs the owner'."
- **Cost and effect.**
  - The review runs at the same time as the 8–11 min of CI, so it adds about no wall time. Model cost is a few dollars per PR [INFERENCE].
  - Expected effect: avoid follow-up fix tickets of the T38/T46/T53 kind, which are about 4 of the 46 tickets T13–T58. It also lets readings shrink (F3).

**F2. Checklist items that review keeps missing should be mechanical checks. Build-estimate drift is costing a ticket at a time.**
- **Evidence:**
  - There are 14 hand-fitted `_build_estimate` functions. `MAX_BUILD_S = 7.0` is copied into each module (isotonic.py:76, project.py:171, quantile.py:82, spline.py:143, trees.py:103).
  - They time two different things. The isotonic, quantile, spline and trees docstrings say "trial build alone"; distance, kernel, power and project say "whole warm `to_native`". So finding 92 is documented but not fixed.
  - The `*_test.py` estimate tests only pin hard-coded numbers; none measures a build.
  - The refit tickets T13, T26, T28, T39, T40 and T46 each re-measure 29–117 shapes by hand. The scripts sit in `/mnt/project-files/transforms-loop/tNN/`, outside the repo.
  - Nothing at registry level forces a family to declare a cap, which is how T49/T50 shipped without one.
- **Root cause.** Each recurring incident turned into a ticket instead of a tool.
- **Change:**
  - Add `Entry.build_estimate` (or `uncapped="<reason>"`) and one `MAX_BUILD_S` in `_registry.py`, with one stated meaning: the whole warm `to_native`.
  - Have `coverage_test.py` assert that every registered class declares one or the other. This catches finding 91.
  - Check in `scripts/native_caps.py`: for each family, build a fixed grid of shapes near the cap and fail if build/estimate falls outside [0.6, 1.6]. Make it required in "Done" for any PR that touches shared machinery or adopts a confit change, and in each reading. This catches T44 before merge.
- **Cost and effect.** About one ticket to build. It turns the six refit tickets into one command run. Confit changes (#387, #430, #456/#461) keep landing, so the saving repeats.

**F3. Readings are the de facto review, and they cost about 26% of inline active time. Most of that goes on re-measuring and long prose.**
- **Time spent.** From commit times, the gap between the last ticket merge and the reading merge:

  | Reading | Hours |
  |---|---|
  | R5 | ≈3.0 |
  | R6 | 3.3 |
  | R7 | 2.15 |
  | R8 | 4.6 |
  | **Total** | **≈13 h** |

  Active inline time was about 49.6 h: Oct 14 10:07 → Oct 15 04:14 UTC, plus Oct 16 03:48 → Oct 17 12:41 UTC, less the 1.4 h pause after T55. Calling the gaps report work is [INFERENCE], but the loop runs one thing at a time.
- **Length keeps growing.** The readings went from 3,608 words (R1) to 18,378 words (R8, 1,470 lines). Speed alone is 445 lines and Exactness 330.
- **Machine changes force double measurement.** R7 and R8 both re-ran the previous reading's commit because the cloud machine changed ("This machine started on 2026-10-16 at 03:48 UTC … a delta … measures … the change of machine").
- **What readings actually find.** Findings 77, 81/82, 85, 86/87, 91 and 92 are all defects an earlier ticket introduced.
- **Root cause.** `report-format.md` demands "Measure every number again on master" plus deltas against absolute timings. With review gone, the reading is the only place defects get caught.
- **Change.** Once F1 is in place, cut readings to:
  - the KPI block, a findings table and needs;
  - a 200-seed parity sweep and `native_caps.py`;
  - speed measured only for families changed since the last reading, reported as in-process ratios (entry/twin, build/estimate) that survive a machine change;
  - at most 400 lines.

  Also merge the overlapping triggers: R6 and the short reading landed 2h40m apart.
- **Expected effect.** About 8 of the 13 hours back for tickets, while keeping the defect-finding parts.

**F4. Picking: "Next" feeds itself, the "easiest not-yet row" rule never runs, and a cheap class is filed as "needs a design".**
- **Evidence:**
  - PLANS.md:545–549 lists `NeighborhoodComponentsAnalysis` and `BernoulliRBM` among "Transformers that read their fit samples … Each needs a design first".
  - The installed twins say otherwise. `_nca.py:370` is `return np.dot(X, self.components_.T)`, which is `GaussianRandomProjection`'s form, already served. `_rbm.py:190-192` is a matrix-vector product plus intercept, then `expit`.
  - Meanwhile T58 (`FactorAnalysis`) took about 2.7 h of 8-BLAS-setup research for one class.
  - 18 of the 20 classes added since Oct 14 serve only with `allow_bound=True` (coverage.md rows). The KPI counts them the same as bit-exact classes.
- **Root cause.** The loop writes the next "Next" item in each PR. "Not yet" rows get blanket labels that nobody checks against the twin's source.
- **Change.** README step 2:
  > "At each reading, refresh a 'not yet' triage table in PLANS: for each class, the twin's `transform` operations (file:line), the nearest existing translator, effort S/M/L, and whether a closed record covers its bound. When 'Next' holds no owner-ordered item, take the smallest S row."

  NCA's bound fits the matvec ruling, but that needs a one-line owner addendum [INFERENCE].
- **Cost and effect.** About 30 min per reading. NCA becomes a short ticket (+1 class) built on `project.py`.

**F5. The board and PLANS have become journals, and orphan PRs are invisible.**
- **Evidence:**
  - The board table in `tickets.md` was empty both at `af89daed` and at HEAD. Under it sit 50–120 lines describing merged tickets, against `workers.md` §3 ("When the PR … merges, delete its row and its section").
  - No inline ticket ever had acceptance criteria written before work started. Instead, "Next" scheduled bundles: "findings 86 and 87 in one PR" became T48, and T46 shipped 3 concerns.
  - PLANS is 549 lines, about 60 of them the history list "Served since this catalog began".
  - #432 has been open and DIRTY since 2026-10-06 21:26 UTC. It came from another session, `pensive-knuth`, during the owner pause, and is not on the board. It is largely redundant: #406 (`36d4eab4`) already added `test_a_periodic_spline_answers_infinity_as_its_twin`, parametrized over degrees 0–5 (catalog_test.py:2325–2327).
- **Change.** In the "Sync" step:
  > "`gh pr list --state open --search 'native in:title'`. Each open PR becomes a board row or is closed with a reason. Inline tickets also get a row and 3 lines of acceptance criteria before work starts. On merge, delete the row. The narrative goes in the PR body, not the board."

  Move "Served since…" out of PLANS.
- **Cost and effect.** Minutes per cycle. A smaller context for each cycle, criteria the F1 reviewer can check, and no more orphan PRs.

**F6. Evidence lives outside the repo, so no one else can re-run it.**
- **Evidence.** `/mnt/project-files/transforms-loop/…` is cited in tickets.md (T53–T58), PLANS and readings 6–8.
- **The precedent.** The 2026-10-06 research did check in its scripts, under `decisions/research/2026-10-06/scripts/`.
- **Change.** Rule:
  > "Every number in a PR body cites a script path in the repo at the PR's head (data may stay outside)."
- **Effect.** Makes F1's reviewer and F3's slimmed readings practical, and lets the owner reproduce results.

### 2. What is working and should be kept
- **Decision records.** Batched rulings backed by research, critique and verification scripts. #404's five rulings unlocked about 18 of the 20 classes added since Oct 14. The bounded-steps record went from opened (Oct 14) to ruled (Oct 15) to built (T42, Oct 16).
- **CI as a real gate.** It caught finding 88, a cosh failure that only shows on CI, in T50's failed run at 18:54 UTC on Oct 16.
- **Defects become generator or test changes:** T15 booleans, T20 ±inf in `EDGES`, and T43's two-global-random-state fixture test.
- **Bound rules.** "Never use the measured maximum as K", and caps with stated ratio bands.

### 3. Disagreements with the report
- **§2 "Supervisor and review" describes the written scheme as if it were practice.** No review has happened since #403. The appendix's "median 12 min" is CI time.
- **The source log is incomplete.**
  - It was filtered on a path that doesn't exist, `packages/sql-transform/src/sql_transform/native`. The real path has no `src/`.
  - `git rev-list` since Oct 5 gives 145 commits with the wrong path and 154 with the real one.
  - As a result, `--stat` shows no code files (T44 shows only PLANS.md and tickets.md), and T18 (`36d4eab4`, #406) is missing.
  - So "53 ticket-tagged" and the 28% figure come from an incomplete set.
- **"About 28% refits" understates non-coverage work.** Of T13–T58, only 17 of 46 tickets add a class or configuration. The 15 "refits" mix confit-driven refits with fixes for the loop's own unreviewed merges (T38←T37, T43←T37/T41, T46←T44, T53←T49/T50). The report files the latter under "estimate drift" and "numeric edge" causes.
- **The "coordination cost" friction (direct feedback, serial merges) does not apply after Oct 6.** No workers ran after that.
- **#432 is not just stale.** Its gap is largely covered by #406's parametrized test, so it should be closed, not revived.

### 4. Out of category
- The coverage KPI counts 19 of 51 classes that serve only with `allow_bound=True` as "native" (goal/metric).
- Confit-loop PRs #409–#493 also have zero review comments; it is the same gap in the other loop.
- The nightly campaign (#435) reported FAIL every day, Oct 8–16, during the pause.
- CI runs on the PR and again on the push to master, which doubles CI minutes.
- A confit-side build budget or timeout would make all 14 hand-fitted estimates unnecessary (architecture).

**Uncertain:** the owner's reasons for the three pauses; whether time between merges was all report or ticket work; the dollar cost of the `/review` run.

---

# B process — MuseProcess

# Category B — Process and workflow: independent review

**Verdict up front:** the loop's process was excellent for the first 36 hours and then quietly changed into a different, weaker process while keeping the first one's paperwork. The worker-wave era (2026-10-05 to early 10-06) had real independent review, parallel throughput, and cheap tickets. The inline era (10-06 onward, branch `claude/project-thread-*`, 49 of 94 PRs in `native-pr-stats.txt`) kept the ticket numbers, board updates, milestone reports and essay-length PR bodies, but dropped the reviewer — precisely when the work shifted from bit-exact transcription to novel parity-bound derivations where a second reader matters most. Meanwhile ~28–50% of tickets bought no coverage (estimate refits against a moving confit), and reporting grew into the single largest prose output of the loop. Coverage per PR is falling monotonically (5 scalers in #329; one class per ticket for T54–T58), and the process has no mechanism that notices this or asks whether the remaining 17 classes are even reachable.

I sampled 9 PR conversations in full (#347, #367, #384, #404, #410, #432, #466, #494, #513), recomputed the merge-time distribution from `native-pr-stats.txt` (median 12 min confirmed, 34/93 merged in <10 min), and word-counted the living docs. Token costs are not in the inputs; where I infer them from word/PR counts I mark [INFERENCE].

## Top findings (ranked)

### 1. Review is real in worker mode and absent in inline mode — and inline mode did the hardest work

The report's "median open→merge 12 min" conflates two regimes. Worker PRs got genuine, re-measuring review: on #367 the supervisor independently re-timed the merged head, found the cap predated confit #362, found that `kernel_is_duckdbs` would *raise* instead of declining on DuckDB >1.5.5 (a correctness bug the green gate had passed), and blocked on both; on #384 the supervisor read sklearn's `SplineTransformer.transform` against the PR and imposed the 7 s build cap and the scipy platform probe. Those PRs took 72 and 47 minutes and have 2–3 substantive review comments. Inline-era PRs have **zero comments**: #494 (T41, six new bounded classes, a new K = n+1 derivation, 12 min), #466 (T40, 10 min), #513 (T58, a novel cross-host LAPACK error bound, 405 new lines in `project.py`, 12 min). Twelve minutes is CI-plus-merge time; no second agent read the derivation.

Root cause in process: `AGENTS.md#helpers-in-cloud-sessions` §5 defines review only as something a supervisor does to *workers*. Inline mode's definition of done (§1) is "gate passes, self-merge." When the owner switched modes after 10-06, the review checklist in `mandates/native/README.md` ("Check one configuration against the source of the installed sklearn…") became self-certification by the author. The gate cannot substitute: it passed #367's probe bug and would pass any bound whose K is merely generous.

**Change (paste into `AGENTS.md#helpers-in-cloud-sessions` §1):** *"In either mode, a PR that (a) registers a parity bound above 0 or a new `ErrorScale`, (b) changes `_registry.py`, `_check.py` or `_helpers.py`, or (c) introduces a build cap/estimate, merges only after a review by a session that did not write it, delivered as PR comments against the loop checklist. Bit-exact family PRs and coefficient-only refits may self-merge."* Cost: one reviewer session (~30 min) on roughly half of future tickets. Effect: restores the only mechanism that demonstrably caught gate-invisible defects (#367 items 1–2, #384 items 2–3), aimed at exactly the ticket types that now dominate.

### 2. The build-estimate refit treadmill: native caps are fitted to confit's *performance*, so every confit speedup creates native tickets

15 of 53 ticket commits are refits (appendix), and the true non-coverage share is worse: T19–T36 (18 tickets) moved coverage 31→31 (readings 4→5) before T37 added the next classes. Each family carries its own least-squares `_build_estimate` (I count 10+ across `trees.py`, `spline.py`, `quantile.py`, `isotonic.py`, `kernel.py`, `project.py`, `distance.py`, `power.py`), each fitted to one master SHA's confit build times, each invalidated by the next confit lowering change (#387→T13, guard→T26, #430→T28, #459→T39, #461→T40, T44's own `x - 0.0` spelling→T46). Two estimate families didn't even measure the same quantity (finding 92, fixed only by docstrings in #508). Every refit pays the full ritual: branch, measurement campaign, full gate, essay PR, PLANS + board edits, CI, merge.

Root cause: the loop chose *predicting* build time over *bounding* it, and owns the prediction in the wrong loop — confit's performance is confit's interface, but native re-derives it per family per confit PR.

**Change:** (a) one shared estimator definition — whole warm `to_native`, stated once in `_registry.py` — and a single `scripts/refit_estimates.py` that regenerates every coefficient; (b) rule for `mandates/native/README.md`: *"Refits are batched: at most one refit PR per wave or per confit release, never a standalone ticket unless a cap is measured wrong by more than 2× (a served step building >14 s, or a refused step building <3.5 s). Widening a cap because confit got faster is not urgent work; it waits for the batch."* Cost: ~half a day to build the script. Effect: 15 historical refit PRs would have been ~4; frees roughly a quarter of ticket capacity [INFERENCE] and removes the incentive to chase confit PR-by-PR.

### 3. A monolithic gate and a full ceremony for micro-tickets

`scripts/gate.py` runs `cargo test` (Rust, confit) plus the entire pytest suite for every change, including one-coefficient Python refits; the worker then waits for CI to run it again. By #513 the suite is 14,157 tests. `workers.md` says sub-20-minute tickets still go on the board with their own branch/PR; the native README adds coverage regeneration, pre-commit, PLANS and board edits to every cycle. Result: 94 PRs in 6 active days, a third merged within 10 minutes of opening — the ceremony, not the work, is the critical path for small tickets [INFERENCE from timing distribution].

**Change:** tier the gate: `scripts/gate.py --native` (pytest `sql_transform/native` + coverage staleness test) is the local pre-PR gate for PRs touching only `native/` family modules and fixtures; the full gate remains mandatory in CI and locally for shared-machinery or confit changes. And allow one PR to carry several same-family follow-up fixes (the T38/T43/T44 zero-sign/read-spelling sequence was three PRs in 26 hours). Cost: an hour in `gate.py` + a line in the README. Effect: removes a redundant full-suite run per ticket without weakening CI; nothing the full local gate caught historically is lost because CI still runs it before merge is possible.

### 4. "Easiest row first" with all classes weighted equally has deferred the only strategic question: are the last 17 reachable at all?

Coverage counts classes, so picking easiest-first (README step 2, "Tickets and review" source 2) maximized the scoreboard early and concentrated all difficulty at the end. The 17 `not yet` rows are `BernoulliRBM, DictionaryLearning, Isomap, KNNImputer, KNeighborsTransformer, KernelPCA, LatentDirichletAllocation, LocallyLinearEmbedding, MiniBatch*` (×3), `NMF, NeighborhoodComponentsAnalysis, PolynomialCountSketch, RadiusNeighborsTransformer, SparseCoder, SparsePCA` — neighbors/manifold methods whose `transform` reads the fitted *training set*, and dictionary/NMF methods whose `transform` itself iterates. PLANS' tail admits "Each needs a design first," but no ticket has ever done that design triage; instead T54–T58 spent one ticket each on single bounded classes of (presumably) modest pipeline value — T58's `FactorAnalysis` required an 8-BLAS-setup cross-host campaign — because the queue order was set by the 2026-10-06 research document, not by value or feasibility.

Root cause: the loop's picking rule optimizes the KPI (`coverage_native`) rather than asking what the denominator should be. The catalog contract (row-local SQL serving) may simply exclude several of the 17, in which case "51/68" understates completion and the loop will grind asymptotically.

**Change:** next ticket is a half-day triage spike, before any new family: classify each of the 17 as *closed-form row-local*, *needs fitted-data lookup (state size?)* , *iterative at transform time*, or *propose out-of-scope*; file one decision record for the out-of-scope set and re-baseline the denominator in `coverage.md`. Then order the survivors by [INFERENCE] pipeline prevalence, not ease. Cost: half a day + one owner ruling. Effect: converts an open-ended tail into a finite, honestly-sized queue — the highest-leverage owner-attention spend available.

### 5. Milestone reports are the loop's largest output and mostly re-write evidence the PRs already contain

9 reports, 71,099 words total (my `wc -w`), growing every time: reading 1 is 3,608 words, reading 8 is 18,378 words / 1,470 lines — for a delta of 7 classes over reading 7, itself written the previous day. The triggers in `mandates/README.md` overlap (batch ≈ five families, *or* KPI quarter/module, *or* 15 PRs/24 h staleness), so on a fast loop something always fires: a report every ~10 PRs. `report-format.md` then demands re-measuring *every* number on master ("Never copy a number from a PR description without a new measurement"), a full 200-seed parity run, and per-family speed tables — days after each PR body already published the same measurements at essay length (e.g. #513). The same content is then written a third time as diary paragraphs in `tickets.md` (T53–T58 entries there restate their PR bodies). And the machine-readable part is broken: reading 8's KPI block carries `coverage_partial: null` and `coverage_notnative: null` with comments admitting `coverage.md` has no such columns — the tracker contract has been silently unmet. #410 already showed the failure mode: the format mandated duplicating decision content, reports copied each other, and the only pre-merge check (fresh-reader) tests wording, not placement.

**Change:** cut `report-format.md` to: generated KPI block (script-emitted from `coverage.py --bounds`, fixing or deleting the two null keys), a delta table linked to PRs, *one* re-measured 200-seed run and widest-build/serve numbers, open findings, needs-owner list — target ≤150 lines; delete the re-measure-everything rule (PR measurements are on the merged head already; re-measure only what a later confit change could have moved). Raise triggers to "owner pause, quarter-coverage crossing, or 25 PRs." Delete the diary function from `tickets.md` (finding 7). Cost: an hour of editing policy. Effect: [INFERENCE] the largest single token saving available — readings 5–8 alone are ~54k words, comparable to the entire codebase's comments — with no loss to the tracker, which reads only the KPI block anyway.

### 6. Owner decisions are batched into mega-research, then block the whole loop

#404 bundled five rulings plus ~330 research scripts into one PR; the loop then paused 2026-10-06→10-14 (eight days, the majority of the calendar span). The bounded-steps record opened 10-06 and ruled 10-15, blocking T42 the whole time; the worker-environment record — a factual question about the Nix dev shell — also waited until #482 on 10-15. The post-pause queue is literally titled "Ruled 2026-10-06, to build": the loop serialized itself on rulings even though decision records already carry a "provisional choice that the loop works under" (`decisions/README.md`), and even though unblocked bit-exact work existed on 10-06.

Root cause: no distinction between *contract* questions (parity-bound form — genuinely owner-level, and #404's downstream label-flip check that produced the `allow_bound` amendment was worth every script) and *fact/implementation* questions (worker environment); and no rule keeping unblocked tickets ready while a record is open.

**Change (paste into `mandates/native/README.md` Rules):** *"A decision record blocks only tickets that name it in `depends on`. While any record is open, the supervisor keeps at least two unblocked tickets in PLANS 'Next'. A question answerable by running a command or reading a manual is never a decision record. One record carries one question and at most one page of summary; research lives in `decisions/research/` and is linked, not bundled into the ruling PR."* Cost: discipline only. Effect: an eight-day full stop becomes at most a partial stall; owner attention is spent one ruling at a time.

### 7. Board/PLANS upkeep became a conflict-generating diary, and orphans have no owner — exhibit A: PR #432

Early on, board maintenance was its own PR stream: #365, #379, #386, #393, #397, #400, #407 — seven PRs in ~30 hours that only move ticket rows. The habit then folded into every ticket PR: each touches `PLANS.md` and `tickets.md`, the two files most likely to conflict across parallel branches, and `workers.md` §6 + the native README require merging master into *all* open branches and regenerating coverage after *each* merge — quadratic bookkeeping per wave (#384 merged master twice before it could land). The result today: `PLANS.md` is 549 lines / 5,776 words of measurement log with a one-line "Next" that just points at another section of itself; `tickets.md` has an empty board table atop 130 lines of retrospective diary. Neither file now does its stated job (pick next work; show what's in flight) at a glance.

PR #432 is what falls through the cracks: opened 2026-10-06 21:26 by a worker session (`claude/pensive-knuth-*`), never reviewed, now DIRTY against master, absent from the board. Its own body says the actual fix already merged in #406/#415; it only adds four fixtures at +7 s gate cost. It survived the pause, the resume, and 40 subsequent merges because no rule assigns orphan PRs when mode switches abandon their workers.

**Change:** (a) cap PLANS: "Next" + "Needs from confit" one-liners only — measurements move to code docstrings (where most already are); the current prose log freezes as a report appendix. (b) In inline mode delete the board table and diary from `tickets.md`; GitHub open PRs *are* the board. (c) Add to `workers.md`: *"A PR untouched for 7 days is closed or adopted as a ticket at the next cycle start; switching modes includes sweeping open PRs."* (d) Close #432 now, or adopt its fixtures into the next spline-touching ticket with the +7 s justified there. Cost: one editing pass. Effect: removes the two highest-conflict files from every PR and the per-merge rebase cascade during waves.

## What is working and should be kept

- **Worker-era supervisor review** (#367, #384): independent re-measurement, blocking/optional marking, direct-to-worker delivery. This is the standard finding 1 restores.
- **Family-per-module tickets** with named collision files (README "Files that every ticket changes"): Oct 5's parallel waves (10+ same-day family PRs, coverage 1→24) worked because of it.
- **Confit needs with confit-only reproductions** (PLANS "Needs from confit"; #384's standalone repro script): the delivered list (#336–#390, #412, #430, #459, #461) shows this cross-loop contract functioning.
- **Generated coverage with a staleness test** and squash-merge-with-expected-SHA discipline: the scoreboard never drifted from the code in anything I checked.
- **Decision-record evidence quality where it mattered**: #404's HistGradientBoosting label-flip verification changed the contract (`allow_bound`) before bounded entries shipped by default. That is the model for finding 6's "genuine contract question."

## Disagreements with `local://native-loop-report.md`

1. **§2 "Supervisor and review"** describes review as the loop's practice throughout. Unsupported for the majority: 49 `project-thread` PRs; all four I opened from that era (#466, #494, #513, plus zero-comment #347 from the pre-worker start) have no review comments at all. The appendix's "median 12 min" is arithmetically right (I recomputed it) but measures CI latency in the inline era, not review speed.
2. **§2 "Gate and merge… 200-seed runs"**: the gate does not run 200 seeds. `catalog_test.py:1000` defaults `NATIVE_SEEDS=8`; `scripts/gate.py` doesn't set it. The 200-seed campaigns are manual, per-PR/report discipline — real, but unenforced by any check a future agent couldn't skip.
3. **Appendix "about 28%" refits** understates non-coverage work by counting only estimate subjects. Counting classes gained, T19–T36 (18 tickets, readings 4→5→6 pre-T37) added zero classes; they are guard/estimate/spelling hardening of already-shipped families. The honest figure for "tickets that added no class" after T18 is roughly half.
4. **§1 pause boundaries**: "Owner pause 2026-10-06 16:11 UTC → 10-14" sits awkwardly with #429 created 2026-10-06T17:28 and merged after; the pause start is fuzzier than stated. Minor, but the report presents it as exact.
5. **§2 "PLANS… most valuable item first"** (echoing the README): as of now PLANS "Next" is a single pointer to a 2026-10-06 ruling section; the de-facto order since 10-14 has been research-document order, not a maintained value ranking, and "easiest row" picking has not operated since the bounded-family queue took over.

## Out-of-category (one line each)

- Numeric: several K derivations (T54, T58) were rescued mid-ticket by the author's own adversarial measurement — good, but that's luck-of-the-author without finding 1's reviewer.
- Code: ten near-duplicate `_build_estimate` implementations is a design smell independent of the refit-process problem.
- Testing: fixture-generator edge gaps (±inf, all-missing strings, ARPACK seeding) recurred four times; a generator-feature checklist belongs in the ticket template.
- Confit loop: its unannounced performance regressions (#363, #456) cost native more time than any native bug; a shared perf-regression gate is a confit-loop question.

**Uncertain about:** actual token/wall-clock costs per activity (not in inputs — my report-cost and gate-cost rankings are [INFERENCE] from word counts, test counts and merge timings); whether CI sets `NATIVE_SEEDS` higher than the local gate; and sklearn-class usage frequencies, which finding 4's prioritization would need.

---

# C cross-loop — AstraCrossLoop

## Verdict

**The loops repair performance problems well, but discover them too late and place too much repair work in native.** The main problem is not slow communication. It is the lack of an enforced performance contract between the packages.

Keep empirical build estimates as admission heuristics for now. Do not treat them as time limits, or automate their refitting before checking for regressions. Otherwise, automation will merely make native accept confit’s regressions faster.

I checked live code, loop rules, incident PRs, and recorded measurements. I independently recomputed the ratios below. **I did not rerun historical benchmarks**, so their timings remain recorded evidence, not independently reproduced measurements. Implementation costs below are **[INFERENCE]**, not measured effort.

## Top findings

### 1. Known downstream performance risks can pass review as “optional”

The strongest evidence is not an unforeseen regression. Review identified the mechanism before merge and explicitly deferred it.

In [#363](https://github.com/ahrzb/sql-transforms/pull/363), the supervisor wrote:

> “Pure work from CASE arms that are never taken now runs on every row. This is correct but slower. Add a PLANS note, unless your re-measured numbers show it matters.”

The measurements covered Normalizer, where this mechanism was not costly. The change then made the three-instance QuantileTransformer fixture take **21,480 µs instead of 1,153 µs** per call. That is an independently recomputed **18.63× slowdown**. [#377](https://github.com/ahrzb/sql-transforms/pull/377) says native discovered it during a milestone reading.

The same process gap affected an unrelated feature. Scalar-subquery support in #456 introduced repeated walks of complete function bodies. Native traced the resulting build regression to that change. [#461](https://github.com/ahrzb/sql-transforms/pull/461) documents the mechanism and repair. Reading 6 records a forest increasing from **3.7 s to 18.1–18.3 s**, approximately **4.9×** slower.

**Process cause.** Confit requires before/after measurements “for performance work,” not for every change that affects binding or lowering. Its CI runs the normal gate, without the native benchmark: `mandates/confit/README.md:125–130`, `.github/workflows/ci.yml:48–60`, `scripts/gate.py:19–32`.

Merely adding the current benchmark would not solve this:

- `benchmarks/bench_native.py:65–82` chooses the widest fixture **that still translates**. A tighter cap can silently substitute a smaller workload.
- Lines 260–268 fail only when native becomes slower than the twin. An 18× regression can pass if native remains slightly faster.
- Its regular build measurement excludes the preceding `to_native` call.

**Concrete change.** Add this rule to both review checklists:

> Changes to binding, sharing, lowering, guards, or native selection must compare fixed downstream workloads against the base commit.  
> A workload becoming `NotNative` is a regression, not permission to substitute a smaller workload.  
> Review identified performance risks as blocking until a representative measurement resolves them.

Extend `benchmarks/bench_native.py` with fixed case identifiers and machine-readable results. Run a small serial, release-build comparison in a separate CI job. Include:

- three-instance QuantileTransformer, catching #363;
- 500/4,000-field struct reads, catching #456;
- a guarded spline under `extrapolation="error"`, catching T44;
- projection and forest cases in both list and struct forms.

Compare base and candidate on the same runner. Record translation, probe, trial build, serving, and admission separately. Use repeated medians and a documented noise allowance, rather than cross-host absolute timings.

**Cost/effect:** approximately 1–2 engineering days plus measured CI overhead. This moves detection before merge and prevents cap changes from concealing lost coverage. Keep the existing structural regression tests alongside it.

### 2. The cap operates at the wrong boundary: it lacks information already available to `to_native`

[#508](https://github.com/ahrzb/sql-transforms/pull/508) states:

> “The entry cannot tell a list return from a struct, so the cap reads the slower one.”

This is an interface limitation, not missing information in the product. `_registry._translate` already knows `step.returns`, the fitted instances, and the final query. It does not pass that context into family admission decisions.

The consequence is concrete. With one AdditiveChi2 sample step over 1,024 features:

- a list builds in **0.85–0.92 s**;
- a struct builds in **8.3–9.6 s**;
- the cap refuses **both**.

I independently evaluated the current formula at this shape: **9.508 s**, confirming the refusal mechanically. See `kernel.py:203–226` and `mandates/native/tracker.md:84–86`.

There is another mismatch. The projection cap explicitly applies to “one estimator” (`project.py:169–171`). `_translate` checks each instance through `translate_part`, then combines them and performs one complete build (`_registry.py:650–735`). There is no corresponding aggregate time-budget check. **[INFERENCE] Multiple distinct instances or composition parts can each pass their local cap while their combined build greatly exceeds it.** I did not measure that counterexample.

Finally, the same “7 s” means different things. Forest, isotonic, quantile, and spline estimates exclude translation and probing. Newer estimates include them. Reading 8 documents this at lines 752–795. Updating docstrings in T53 clarified the difference but did not remove it.

**Process cause.** Family tickets own cost formulas, although the cost depends on the complete emitted function and its return representation. This is **Information Leakage** and **change amplification**: engine costs become separate native policies.

**Concrete change.**

> Apply admission policy to the complete `PythonTransform`, including its return form, instances, and composition.  
> Family translators report cost features; they do not independently decide the whole step’s time budget.  
> Every estimate names its measured phase. A predicted time is not an enforced deadline.

Place that decision at `_registry._translate`, where the context exists. Preserve early family size checks where necessary to avoid constructing enormous expressions. Do not replace them with “build first, then refuse.”

Add acceptance cases for:

1. AdditiveChi2’s fast list and slow struct at 1,024 features;
2. several distinct fitted instances that individually fall below the threshold;
3. a composition containing several individually admissible parts;
4. substantial probe cost, such as wide QuantileTransformer.

If the desired guarantee is an actual time limit, that requires a separate cancellable build design. No regression fit provides that guarantee.

**Cost/effect:** approximately 2–4 days for the admission boundary and representative migrations. It removes known false refusals and gives future calibrations one coherent target.

### 3. T44/T46 moved confit’s sharing heuristic into the catalog instead of resolving the engine limitation

T44’s optimization is semantically careful but architecturally expensive.

The current registry explains:

> “The input guard reads `coalesce(x, NaN)`, five nodes, under the six that confit shares…”

It therefore adds `- 0.0` to cross that threshold. See `_registry.py:385–406` and confit’s `MIN_SHARED_SIZE = 6` at `share.rs:136–137`.

This is **not** the signed-zero correction from T38. T44 deliberately preserves negative zero: `-0.0 - 0.0` remains `-0.0`. Its purpose is to influence sharing.

The interaction then required T46. Translator-added guards also read the expanded expression, causing confit to compute it early and carry it through the body. [#500](https://github.com/ahrzb/sql-transforms/pull/500) records a guarded spline building in **9.2 s instead of 4.3 s**. T46’s rewrite reduced that to **3.6 s**.

The catalog now contains:

- `_apart`, which manufactures a larger expression;
- `_plain`, which reconstructs guard trees to remove that expression;
- `Entry.apart_of`, which selects whether to apply it;
- projection-specific cost formulas that decide between spellings;
- extra endpoint tests that make two reads occur under one CASE condition.

See `_registry.py:361–464`, `project.py:198–265`, and [#498](https://github.com/ahrzb/sql-transforms/pull/498).

The costs are not merely theoretical. Reading 8 records the forest cap shrinking from about **6,800 to 5,200 output fields** at 128 features, a **23.5% reduction**, after these reads.

**Process cause.** Package ownership prevented native from fixing confit directly, but did not require a lifecycle for cross-package workarounds. Native marked the sharing need as no longer needed while retaining growing knowledge of confit’s internals.

**Concrete change.**

> An optimization that depends on confit’s internal node counts or sharing rules requires a linked confit need.  
> Record its semantic invariant, performance cases, and removal condition.  
> Keep the need open while the workaround limits admitted configurations or requires family-specific exceptions.

Give confit a joint acceptance matrix covering guarded/unguarded reads, distinct CASE arms, list/struct returns, and few/many features. Prefer fixing sharing placement and live ranges inside confit. Do **not** simply lower the threshold: #363 already measured that sharing five-node coalesces could make serving quadratic.

After confit meets those cases, remove the catalog machinery in one coordinated cutover. Preserve tests for signed zero, NaN, infinities, and traps.

**Cost/effect:** approximately half a day to establish the contract, then several days of engine work; exact effort is uncertain. This removes a repeated source of estimate drift rather than recalibrating around it.

### 4. Caps hide unmet needs from the priority system

Confit’s current plan groups native needs under:

> “Open, low priority (the catalog builds within its caps)”  
> — `mandates/confit/tracker.md:135`

That framing is circular. The catalog builds within its caps partly because those caps exclude configurations affected by the unresolved engine cost.

Native’s corresponding need now affects forests, splines, quantiles, AdditiveChi2, PowerTransformer, sparse projections, and cluster distances. Examples include a **0.14 s list versus 7.2 s struct** for repeated sparse sums, and a distance build refused after **475 s**. These are recorded in `mandates/native/tracker.md:77–106`. They are not all evidence of newly introduced regressions, but they establish continuing product impact.

The handoff mechanism also depends on active sessions. Shared rules require a message and an entry in the requester’s PLANS. Native currently says:

> “send it to the Confit Loop when it runs again”

for the small-value reproduction, and similarly for logarithm sharing (`mandates/native/tracker.md:184–201`). Confit does have a durable “Delivered” fallback. What is missing is a common identifier, acknowledgement state, and consumer acceptance.

**Process cause.** Successful workarounds reduce ticket priority without measuring lost configurations or maintenance cost. Separate prose lists can disagree about whether the underlying need remains important.

**Concrete change.** Use one durable issue per cross-loop need, referenced from both plans:

> Every need records its reproduction, affected configurations, acknowledgement, delivering SHA, and consumer acceptance.  
> “Delivered” requires the requester’s acceptance case on the delivering SHA.  
> A workaround does not lower priority without recording lost configurations and continuing maintenance.

Use versioned **acceptance cases**, not a large runtime capability-negotiation system. A capability should state observable behavior, such as guarded struct-read scaling, rather than “sharing supported.”

Rank the unresolved struct-read need by blocked configurations across families. Do not rank it as harmless because several families now refuse those configurations.

**Cost/effect:** roughly half a day for the shared convention and existing high-impact needs. This survives paused sessions and prevents caps from disguising engine debt.

### 5. Auto-refitting would help, but only after making calibration reproducible and independent of regression approval

The loop repeatedly performs substantial calibration work:

- T13: 139 spline builds;
- T39: 93 projection builds;
- T40: grids across four families;
- T46: another four-family calibration;
- T53: 117 AdditiveChi2 shapes.

The evidence is in commits `6a26703b`, `318575b2`, `fbbcd249`, and PRs [#500](https://github.com/ahrzb/sql-transforms/pull/500) and [#508](https://github.com/ahrzb/sql-transforms/pull/508).

However, the calibration scripts and raw data are referenced through `/mnt/project-files/transforms-loop/...`, not checked into the reviewed repository. Reading 8 names separate scripts for each family and separate aggregation scripts (`2026-10-17-reading-8.md:1436–1462`).

Current unit tests mainly pin what the formula admits. For example, `catalog_test.py:2782–2792` compares calculated estimates with `MAX_BUILD_S`. Such a test can remain green when confit makes the actual build five times slower.

T53 reports fitting and evaluating the same shape collection. I found no documented held-out validation set. That does not invalidate the measurements, but it weakens claims about unseen shapes.

**Process cause.** Each refit is treated as another family implementation task. Measurement definitions, datasets, provenance, and acceptance checks remain partly session-local.

**Concrete change.**

> Check in calibration shape generators and a common measurement command.  
> Store raw measurements with the confit SHA, native SHA, build profile, host details, phase, and return form.  
> Validate candidate coefficients on held-out shapes and near-cap cases.  
> Never accept a regression by automatically refitting the estimate.

Extend the existing benchmark rather than creating another framework. Separate two operations:

1. **Compare:** detect changes in build, serving, admission, and refusal cost.
2. **Calibrate:** propose coefficient changes after those changes are understood and accepted.

Emit a proposed diff and evidence artifact. Do not automatically merge it. Reuse base-versus-candidate measurements to identify affected families, rather than refitting every family after every confit change.

**Cost/effect:** approximately 1–2 days initially, plus calibration runtime. It should remove repeated script reconstruction and inconsistent timing definitions. It will not remove the need to investigate new cost mechanisms.

## What is working and should be kept

- **Confit-only reproductions and commit searches establish causes.** The #456 investigation identified repeated subquery walks, rather than attributing all timing movement to the host.
- **Repairs add targeted regression tests.** #377 pins sharing behavior structurally. #461 adds a 500/4,000-field scaling test at `test_sql_functions.py:399–433`.
- **Cross-loop deliveries produce real gains.** #430 removed repeated guard work. #459 and #461 reduced costs enough for T39/T40 to admit previously refused configurations.
- **Measurements increasingly separate host changes from code changes.** Reading 7 reran the earlier commit on the current host. Keep that practice in the automated comparison.

## Disagreements with the report

1. **Not all listed refits were caused by confit regressions.** T26 followed native’s new input guards. T46 followed native’s T44 rewrite. T53 added a missing cap. Conversely, #387, #430, #459, and #461 improved performance. Combining these mechanisms overstates confit-caused rework.

2. **The table’s “#430 → #427/#437” ordering is wrong.** #427 calibrated the guard cost before #430 fixed it. T28/#437 subsequently recalibrated after #430.

3. **“~8 h” for #456 → #461 is unsupported by the commit timestamps.** `0def508e` to `620112e3` spans **4 h 7 min 29 s**. This is the merge-to-merge interval, not time spent diagnosing.

4. **“Late detection” omits the more important #363 fact:** review predicted the performance risk before merge and classified it as optional.

5. **A proposal to “add a performance gate” must acknowledge the existing one.** #461 already added a timed scaling assertion to the ordinary gate. The missing piece is broader downstream coverage and base-relative comparison.

6. **The supplied source paths are stale.** The live catalog is under `packages/sql-transform/sql_transform/native/`, not `packages/sql-transform/src/sql_transform/native/`. The supplied path-filtered log also omits native source statistics from several incident commits. I used live files and PR pages for those claims.

## Out-of-category issue

- **Parity:** signed-zero correctness and performance-driven `x - 0.0` spelling are separate concerns; their tests must remain separate during any sharing redesign.

---

# C cross-loop — OpusCrossLoop

# Category C review: cross-loop coordination and estimate drift

**Verdict:** The task brief says confit changes forced the native loop to refit its estimates again and again. That is only half right. Most refits came from the design of the build estimates and caps, not from coordination between the loops. The native loop maintains 14 hand-fitted formulas that predict build time in seconds. Each one is held to a 7 s cap that no owner approved. No test can notice when a formula goes stale. So any cost change on either side of the loop boundary means a new refit ticket. Of the refits in the brief's list, T26, T46 and T53 were caused by native's own changes, not confit's. Coordination problems are real but come second: confit has no performance gate, the needs channel has a priority inversion, and native depends on confit behaviour that no test pins.

## 1. Top findings

### F1. The 7 s cap and its 14 formulas are the treadmill. Much of the refitting moved the cap by less than the formulas' own error.

**Evidence**
- There are 14 functions that estimate build seconds:
  - `quantile`, `isotonic`, `spline`, `trees`, `distance`, `power`
  - `project._build_estimate`, `_centered_build_estimate`, `_sparse_build_estimate`, `_factor_build_estimate`
  - `kernel._additive_chi2…`, `_rbf…`, `_skewed…`, `_nystroem…`
- Each is a polynomial fitted on one container at one master SHA. All are checked against `MAX_BUILD_S = 7.0` (for example `quantile.py:82`, `trees.py:103`).
- Neither `goal.md` nor `catalog-contract.md` mentions build time. The 7 s first appears in `51cc626d`, "native spline: cap the build at an estimated 7 s", with no reason given.
- The tests that guard the caps are arithmetic only. For example, `catalog_test.py:2546` asserts `_build_estimate(24_702) <= MAX_BUILD_S`. Such a test can never notice drift.
- How often the formulas were refitted:
  - Commits changing coefficients: spline in T6, T13, T21, T26, T46 (`git log -G`, below); trees in T22, T26, T28, T30, T32, T40, T46.
  - T28 and T40 also refit spline (stats of `89d59f4e` and `fbbcd249`). That makes 7 spline fits in about 4 working days.
- T46 moved the caps by less than the formulas' own error band. These figures are from the docstrings, before and after T44's reads:

  | Formula | Cap before | Cap after | Ratio |
  |---|---|---|---|
  | quantile | 13,400 leaves | 15,000 leaves | ×1.12 |
  | isotonic | 19,700 thresholds | 26,800 thresholds | ×1.36 |
  | trees, 8 features | 10,300 fields | 10,000 fields | ×0.97 |
  | spline, degree 3 | 170–230 features | 170–230 features | ×1.0 |

  The formulas themselves are only accurate to 0.74–1.53× (`project.py` fits) and up to 2.57× (`_sparse_build_estimate`).
- T40 moved its caps by ×1.15–1.7. It turned 15 of 291 near-cap fits native ("7 of 29 forests, 4 of 16 isotonic, 2 of 74 quantile, 2 of 172 splines").
- No refit ticket changed the coverage KPI. Coverage stayed at 35 through T39/T40 and at 41 through T46.
- Wall-clock gaps between merges in the inline session (`gh pr list`):

  | Ticket | Hours |
  |---|---|
  | T26 | 2.0 |
  | T28 | ~2.9 |
  | T34 | 1.1 |
  | T39 | 0.5 |
  | T40 | 1.4 |
  | T46 | 4.8 (includes the `_plain` guard fix) |
  | T48 | 1.0 (includes the recursion fix) |
  | T53 | 0.5 |

  That is about 14 h. [INFERENCE] This is roughly 20% of the inline session's active time.

**Process root cause**
- No rule says what the cap is for, or how much drift is acceptable.
- Every observed cost change, speed-ups included, therefore turned into a "fit again" ticket.
- The fitting tests pin the formula, not reality, so stale formulas are found only by milestone readings.

**Change.** Add to `mandates/native/README.md` under "Rules":

> **Build estimates.** A cap protects `to_native` from builds that take minutes or run out of memory. It is not a precise budget. Refit a family's estimate only when, on the calibration shapes of `benchmarks/native_estimates.py`, the measured/estimated ratio falls outside [0.5, 2]. A faster confit is not a ticket by itself: record it in PLANS and refit in the next ticket that touches the family. Each estimate states which time it estimates, and it is always the whole warm `to_native`.

Also open `decisions/open/build-budget.md` to ask the owner what build time `to_native` may take, and whether a step that builds in 14 s is acceptable.

Longer term, file a confit need: "a build deadline that refuses by name" (`DuckDBInferFn(..., deadline_s=)`, checked between expansion, binding, sharing, lowering and per-function codegen). `_builds` would then enforce the cap directly, and all 14 formulas could be deleted. Keep only the structural, memory-driven caps such as `_SKEWED_PRODUCTS` and `_MAX_STORED`.

A deadline would make `NotNative` depend on the host. There is already precedent for that: `row_sumsq_is_numpys`, `csr_sum_is_scipys` and the FactorAnalysis pivot refusal all depend on the host.

**Cost and effect**
- The rule and the decision record take about 1 h.
- With the [0.5, 2] band:
  - T40, and the refit part of T46, would not have been tickets.
  - T13 (about 10×), T28 (2.4–3.6×) and T39 (3.5×) would still have been needed.
- The deadline need would end the treadmill entirely. [INFERENCE] It costs one confit ticket.

### F2. Confit has no performance gate for the costs native depends on; regressions reach native only through native's own readings.

**Evidence**
- Confit README lines 125–127: "**Evidence.** For performance work, give before and after numbers… CI only runs the gate." Query-class PRs have no performance duty.
- `pr://456` merged 9 minutes after it opened. Its evidence was 137 parity tests, a 2,400-seed campaign and the gate. It had no build timing.
- Even so, #456 made struct reads field by field quadratic. Reading 6 records 250 trees of depth 5 building in 42–44 s, against an estimate of 6.5 s.
- Native found it in reading 6 by crossing native and confit commits (`split/run_mixed.sh`).
- #363 was a served need. As a side effect it regressed QuantileTransformer serving from about 25× faster than the twin to 1.4×. Native's reading 2 found it about 1.3 h later (`b0a69768`).
- Fast detection happened in both cases only because a reading happened to run. Readings came about a day apart.
- The `pytest` step in CI already runs native's tests on every confit PR, but no native test times anything.
- Confit's own timing tests are almost all ceilings for catastrophic cases (`< 60` s). The only ratio test is `test_a_struct_read_field_by_field_builds_in_about_linear_time`, added reactively in #461 (`git log -S`). No test covers serving speed.
- Confit's `success-measures.md` has no measure for build time. Its serving-latency measure has "no enforced numeric threshold".
- `mandates/README.md` "Between the loops" covers only native-to-confit needs. It has no confit-to-native "cost changed" notice.

**Process root cause.** Cost regressions are found by the downstream loop, by hand, at milestone cadence. The PR that caused the regression carries no performance duty.

**Change.** In confit README "Review checklist → Evidence", replace the first sentence with:

> Every PR that changes `specializer/{frontend,plan.rs,share.rs,lower.rs,exec/}` runs `pytest packages/confit/tests/test_native_costs.py -p no:xdist` and gives its output. If a cost test moves by more than its band, the PR says so, and after merge you tell the native loop which estimates it shifts.

Create `test_native_costs.py`, owned by confit, from native's confit-only reproductions. These already exist in `/mnt/project-files/transforms-loop/`. Each test is a ratio test, so it does not depend on the host.

| Test | Asserts | Would have caught |
|---|---|---|
| field reads | 500 → 4,000 fields | #456 (this test exists) |
| shared-value reads, 256 parameters × 16 fields | Serving time ≤ 2× plain reads | #459's problem, and its reversal |
| guard read per field | Building with a guard ≤ 1.5× building without it | #430 |
| QuantileTransformer, 3 instances | Serving ≤ 1/5 of the twin's time, in the same process | #363 |
| `x - 0.0` | Shared at 2 or more reads (see F4) | Silent loss of T44's gain |

Add `test_native_costs.py` as a serial CI step after `gate.py`, without xdist, to avoid timing noise.

**Cost and effect.** [INFERENCE] About 5 tests at about 2–4 s each, around 20 s of CI. This would have caught both regressions that actually happened (#363, #456) in the PR that caused them, rather than 1–3 h later and only by chance.

### F3. The needs channel inverts priority: the caps hide the cost they create.

**Evidence**
- Confit PLANS, "For the native catalog": "Open, low priority (the catalog builds within its caps)".
- Reading 8, line 1313, repeats this: "The reason is that the caps keep the catalog's builds within about 7 s."
- Native need 1, "A struct output read field by field, built once for the call", has been open since 2026-10-06. Native PLANS says it now sets the caps of 8 families:
  - forest, spline, quantile;
  - AdditiveChi2 (struct 8.3–9.6 s vs list 0.85–0.92 s);
  - PowerTransformer (14.4 vs 4.8 s);
  - SparseRandomProjection (7.2 vs 0.14 s);
  - distances (2,000 centres take 73 s vs 1.1 s);
  - SkewedChi2 (out of memory).
- Every formula is fitted "on the slower return of each shape", so need 1's defect is built into all of them. When need 1 is served, all of them go stale at once.
- Two needs are stuck as "send it to the Confit Loop when it runs again" (PLANS 185–186, 203–204).
- The reproductions live outside the repo (`/mnt/project-files/...`).
- One measurement made in a cold container produced a false regression report. Confit spent #383 disproving it (reading 3, finding 43).

**Process root cause**
- A need's priority is judged by whether the catalog builds, which the caps guarantee by construction. It is not judged by what the need costs the catalog.
- Needs are prose with no fields for which estimates depend on them, or what the catalog has worked around.

**Change.** Add to `mandates/README.md` "Between the loops":

> Each need has: `id`, `asked`, a reproduction committed under `benchmarks/needs/<id>.py`, `blocks` (the families and the estimates or workarounds it sets), `cost` (the share of a reference sweep refused or slowed by the need, and the catalog tickets spent working around it), and `on delivery` (the estimates to refit and the workarounds to delete). Confit ranks needs by `cost`, never by "builds within caps". A performance regression is reported only after three alternating runs in a warm process.

**Cost and effect.** About 1 h of documentation plus moving the scripts. Need 1 would get ranked by its real cost: 8 families, plus T26/T28/T40/T46 fits that priced it in. When it is served, the refit list is already written.

### F4. T44 built confit's undocumented sharing heuristic into the catalog's SQL. The cascade cost about 7 h, and nothing protects the gain.

**Evidence**
- T44 (`eeefa315`) makes translators read `x - 0.0` because `share.rs:135` sets `MIN_SHARED_SIZE: u32 = 6`, and `coalesce(x, NaN)` has 5 nodes.
- `_registry._apart`'s docstring rests on "Confit folds only constant subtrees, so it keeps the `- 0.0`". No confit test or contract claim backs this. The serving contract has no claim about sharing.
- `x - 0.0 → x` is an IEEE-exact simplification, so confit may legally add it.
  - [INFERENCE] If it did, T44's serving gain (0.23–0.81× the time) would vanish silently. No parity test would fail.
  - `guard_test.py:128-130` checks native's SQL text, not confit's lowering.
- The knock-on tickets, all caused by T44:
  - T45: per-arm recomputation made the ordinal chain slower.
  - T46: the guard carried `x - 0.0` through the whole body; a spline of 128 features built in 9.2 s, not 3.6 s.
  - T48: a list over shared values builds about as n²; refitting used 512 builds.
  - The dual estimator plumbing in `Entry.apart`, `project._reads_apart` and `kernel._rbf_reads_apart`.
- Wall-clock: T44 1.1 h, T45 [overlapping], T46 4.8 h, T48 about 1 h.
- The workaround was chosen while confit was paused (`tickets.md:132`). The cost was never put to the owner as "resume confit or work around".
- On −0.0: the identity is sign-safe (−0.0 − 0.0 = −0.0), and `guard_test` pins this with (−0.0, 0.0) rows at `ulps=0`. That part is correct.

**Process root cause.** Native has no rule for spellings that depend on confit's cost heuristics. They are neither registered nor pinned, and their blast radius is not assessed.

**Change.** Add to `mandates/native/README.md` "When confit does not have a capability":

> If a spelling exists only because of a confit cost heuristic, keep it in one place in `_registry.py`. List it under PLANS "Relies on confit behaviour" with the confit line it relies on. Send confit a need to pin that behaviour with a cost test. If confit is paused and the workaround touches more than one family's estimate, note its cost in the next report as an owner question: resume confit, or keep the workaround.

Also send confit a need: "a value bound once by `SqlFunction` (#412) is computed once a row whatever its size". [INFERENCE] Lets are expanded before `share.rs` applies its size test. With this need served, native could delete `_apart`, `_plain` and `Entry.apart`.

**Cost and effect.** Writing the rule is cheap. Pinning `x - 0.0` sharing is one confit test. The explicit-binding need removes a whole class of spelling tricks and the dual estimates.

### F5. The refit harness is rebuilt every ticket, outside the repo, so refits are slow and inconsistent.

**Evidence**
- Fit scripts and data live in `/mnt/project-files/transforms-loop/t53/`, `t54/`, … `t58/` and `reading-6/7/8/`, and are cited in `tickets.md` and the readings.
- Sample sizes per fit:
  - T48: 512 builds.
  - `_sparse_build_estimate`: 116 shapes plus 50 holdout shapes.
  - `quantile`: 66 builds.
- Finding 92: T46's four formulas time "the trial build alone", while T47/T48/T51/T52 time "the whole warm `to_native`". Both are compared with the same 7 s.
- T53 (#508) only made the docstrings say which time each one estimates. It did not unify them.
- The repo has `benchmarks/bench_native.py`, but it times only the widest fixture step and is not run in CI.

**Process root cause.** No shared, versioned tool exists for estimates, so each ticket re-derives method, timer and shapes.

**Change.** Commit `benchmarks/native_estimates.py`:
- A manifest of 3–5 calibration shapes per formula, both return types.
- One timer (warm `to_native`, release build, median of 3).
- `--check`: prints measured/estimate per shape and exits non-zero outside [0.5, 2].
- `--fit <family>`: least-squares fit on the terms the formula declares; prints the new coefficients and docstring band.

Run `--check` in the nightly workflow (`nightly-campaign.yml`) and post drift to an issue titled "Native estimate drift". This mirrors confit's "Nightly campaign findings" triage.

**Cost and effect.** [INFERENCE] One ticket of about 3 h. Coefficient-only refits (T13/T28/T39/T40-type) become short mechanical PRs. Finding-92-type inconsistency becomes impossible. Drift is found nightly instead of at readings.

## 2. What is working and should be kept
- **Confit-only reproductions plus crossed bisection.** Reading 6 ran T36's catalog with T35's confit and the reverse (`run_mixed.sh`). This pinned #456 precisely and gave confit an actionable reproduction.
- **Confit turns needs around quickly.** #377 landed about 1 h after reading 2. #461 landed 4 h 7 min after #456 (20:42Z → 00:49Z). Both arrived with measured before/after numbers.
- **#461 added a ratio test.** `test_a_struct_read_field_by_field_builds_in_about_linear_time` is exactly the right test shape; F2 generalises it.
- **Estimates record their provenance.** SHA, shapes, accuracy band, and "took none slower than / refused none faster than" make drift checkable. `_builds`' trial build turns a confit refusal into `NotNative`, not a failure at serving time.
- **T44 picked a sign-safe identity.** It used `- 0.0`, not `+ 0.0`, and tested it with signed-zero rows at `ulps=0`.

## 3. Disagreements with the report
- **Causal attribution is wrong.** §2 says "Native T13, T26/T28, T39/T40, T46, T53 refit estimates or changed behaviour after these [confit changes]". In fact:
  - T26 was fitted on `6aea15e`, native's T25 guard.
  - T46 came from native's T44.
  - T53 was a cap missing since T49/T50 (finding 91).
  - Only T13, T28, T39 and T40 follow confit changes.
- **The #430 row reverses the order.** It reads "#430 → #427/#437 … Detection: re-measuring builds after #430". But #427 (T26) merged at 15:39Z, before #430 merged at 17:44Z on 2026-10-06. T26 modelled the defect #430 then removed, and was undone by T28.
- **The appendix's "15 subjects … refits … about 28%" is too broad.** It counts T27 (encoder guard), T29 (probe draws), T30 (forest spelling), T38/T43 (zero sign) and T44 (read spelling). The estimate/cap set is T13, T26, T28, T34, T39, T40, T46, T48 and T53, plus the "adopt confit" PRs #369 and #378.
- **The report misses that the refits had no effect on coverage,** and that T46's refits moved the caps by ×0.97–1.36, inside the formulas' own error band.
- **"Detected (by native, late)", from the task framing, overstates the latency.** #363 was found about 1.3 h after merge and #456 about 2–3 h after. The real problem is that detection depended on a reading happening, not that it was slow.
- **"~8 h" for #456 → #461** is really about 4 h to confit's fix and about 7.4 h to native's last refit (T40).

## 4. Out-of-category notes, one line each
- `_check`'s error-scale path treats −0.0 and 0.0 as equal. So sign-of-zero breaches in bounded families (T38's 403 fields, T43's 54) are visible only to dedicated reading scripts, not to `native.check`.
- Native PLANS "Needs from confit" carries about 60 lines of "Served since this catalog began" history. This goes against its header, "open work only".
- PR #432 has been open and stale since 2026-10-06.

**Uncertain:**
- The share of wall-clock time is inferred from gaps between merges in one inline branch. Worker-mode tickets (T13) are not counted.
- Whether `uv sync` builds confit in release mode in CI affects how noisy the cost tests would be. I did not check this.
- I did not measure whether lets avoid `MIN_SHARED_SIZE`. That is marked [INFERENCE] in F4.

---

# C cross-loop — MuseCrossLoop

# Category C: Cross-loop coordination and performance/estimate drift

**Verdict: The framing in the task — "confit changes regressed costs and forced refits" — is half wrong, and the wrong half is the instructive one.** Of the six named confit PRs, only two were unrequested regressions: #363's side effect on `QuantileTransformer` serving, and #456's quadratic struct field-reads. The other four (#387, #430, #459, #461) were capabilities the native loop itself asked for, delivered fast, which then predictably invalidated native's fitted build-time formulas. So the loop has two distinct diseases, not one: (a) confit feature work can silently wreck native's build/serve costs and nobody finds out until the next milestone reading; (b) native hard-codes confit's transient cost model into 14 hand-fitted polynomial formulas in product code, so *every* confit improvement — the desired outcome of the whole two-loop arrangement — generates a native refit ticket. Disease (b) is the larger cost and is entirely self-inflicted by native's design. Disease (a) is the larger risk. Both are fixable cheaply; neither is fixed by "communicate better," because the communication that exists (needs with confit-only reproductions) actually works well when it is triggered. What is missing is any *automatic* trigger.

---

## Top findings

### 1. No performance gate anywhere: confit PRs are correctness-gated against native, but cost-blind — both real regressions were found by native, at milestone readings, by hand-bisecting commits

**Evidence.** CI runs only `scripts/gate.py` (`.github/workflows/ci.yml:59-60`): `cargo test` + whole-repo `pytest`. That means a confit PR *does* run the native catalog tests — correctness coupling is immediate and strong. But nothing in the gate times anything native-shaped. `benchmarks/bench_native.py` exists and even "fails where a native step serves a call more slowly than its twin," but it is run only manually for readings (reading 6, §"how each number was taken"), never in CI. The sole timing assertion in the entire test suite for this surface is the one added *after* the fact in #461: `test_a_struct_read_field_by_field_builds_in_about_linear_time` (`packages/confit/tests/test_sql_functions.py:421-431`), a *relative* scaling check (4,000 fields < 25× the 500-field time) that would pass even if both builds slowed 10×.

The detection stories are damning in the same way twice:

- **#363** (`2449d7e5`, share a repeated subexpression once per row) evaluated shared values before the first item, so untaken CASE arms ran on every row. Native found it by remeasuring for reading 2: a three-instance `QuantileTransformer` went from ~25× its twin to 1.4× (`mandates/native/reports/2026-10-05-reading-2.md:107,326`). Nobody needed to search hard for the mechanism: the #363 supervisor review had already flagged it — *"🟡 Optional: `lower.rs`: `share::share(&exprs)` emits its definitions before the first item, so pure work from CASE arms that are never taken now runs on every row… Add a PLANS note"* — and the PR body lists it under Follow-ups ("It doesn't show in these numbers"). It didn't show because #363's measurements were Normalizer shapes, the thing being optimized, plus a catalog spot-check of Normalizer only. The known-risk was downgraded to optional because no quantile-shaped number was required.
- **#456** (`0def508e`, scalar subqueries — pure feature work, no native need) added a per-SELECT-item subquery walk that re-walked the whole SQL-function call at every field read, making the catalog's field-by-field query (`_registry.query`, `_registry.py:756-770`) quadratic. Its PR evidence is 137 parity tests, a campaign, and the gate — no build/serve numbers at all, which the confit review checklist permits because it requires before/after numbers only "for performance work" (`mandates/confit/README.md`, Review checklist §Evidence). Native found it in reading 6's width curves (forest of 100 depth-6 trees: 3.7 s → 18.1–18.3 s; reading 6 §3, finding 76) and then did exemplary but expensive forensics: a commit search running each commit *with its own confit build*, then cross-running T35's catalog on T36's confit and vice versa to attribute the slowdown to confit rather than its own T36. That is a full manual bisect across two packages, done at a milestone, for a regression that a single "build a 4,000-field struct field-by-field" timer on the #456 PR would have caught in seconds (native's repro: 0.92–1.01 s before #456, 5.84–6.13 s after; #461 then fixed it to 1.00–1.12 s).

**Root cause (process).** The confit loop's definition of done measures the oracle contract exhaustively and costs not at all, except when the ticket is *labeled* performance work. But the changes that move native's costs are exactly the unlabeled ones: frontend walks (#456), shared-value placement (#363, #387), CASE lowering (#459). Meanwhile native's readings — the only place costs are systematically remeasured — run at wave boundaries, so detection latency is "whenever the next reading happens," and attribution falls to whichever loop notices, which has been native both times.

**Concrete change.** Add a small, fixed, baseline-relative cost harness and run it in CI on any PR touching `packages/confit/src/specializer/` (and on native PRs touching `_registry.py`/`_helpers.py`). Paste into `mandates/confit/README.md` review checklist, replacing "For performance work, give before and after numbers":

> **Catalog costs.** If the PR touches `share.rs`, `lower.rs`, `plan.rs`, `lets.rs`, `exec/` or `frontend/`, run `uv run --no-sync python scripts/cost_shapes.py --check` on a release build and paste the table. A shape whose build or serve time is past 1.3× its pinned baseline blocks the merge, whatever the ticket is about. Repin a baseline only in the same PR that measures the change, with the reason.

`scripts/cost_shapes.py` (new, ~150 lines borrowing `bench_native._field_count`/query plumbing) times ~8 canonical shapes, each chosen because a named past incident moved it: (1) confit-only struct of 4,000 fields read field-by-field and whole (#456/#461, #430); (2) `RandomTreesEmbedding` 250 trees depth 5, build (#456); (3) three-instance `QuantileTransformer`, 64-row serve (#363/#377); (4) `PCA` 256×64 read `coalesce(x, NaN)`, build+serve (#459, finding 73/85); (5) `SplineTransformer` 32 features degree 3, build (#387); (6) `QuantileTransformer` 8,000 quantiles × 1 feature, `to_native` (function-body pass, finding 72); (7) Normalizer l2 at 128, build+serve (#363's own shape, so its regression class can't recur invisible); (8) a guarded wide step (Box-Cox 128 features; #430/T46 guard-carrying). Baselines pinned in a checked-in JSON with a machine-calibration shape (a fixed 500-field build) so times are ratios, not absolutes — this also addresses the cross-machine drift the readings keep noting (reading 8 compares machines by a 0.98× bench ratio, ad hoc). Note `bench_native`'s twin-relative failure would *not* have caught #363 (1.4× is still faster than the twin) — the baseline must be master's own prior numbers, not sklearn's.

**Cost/effect.** ~1 day to write (all eight repros already exist; they're currently stranded in `/mnt/project-files/transforms-loop/` — see Finding 5), plus a release-build CI job on a subset of PRs. Expected effect: converts both historical regression classes from "found at next reading + manual two-package bisect" to "red check on the offending PR." This is the single highest-value change in this category.

### 2. The 7-second estimated-build cap is a maintenance treadmill by construction: 14 hand-fitted formulas encode confit's current optimizer, fitted by scripts that are not in the repo, and every new family must fit another one

**Evidence.** There are 14 `*_build_estimate` functions across 8 native modules (`distance.py:105`, `isotonic.py:96`, `kernel.py:203,320,511,703`, `power.py:355`, `project.py:174,198,387,924`, `quantile.py:218`, `spline.py:146`, `trees.py:106`), each a bespoke polynomial in that family's shape parameters with magic coefficients (e.g. spline: `2.37e-5 * nodes + 7.53e-9 * nodes * width + 6.48e-6 * width * features`, `spline.py:146ff`), each docstring citing the master SHA and date it was fitted on ("T46 on master 12aca98, 2026-10-16… each within 0.80 to 1.29 times the estimate" over 113 builds). The refit-only PRs are T13/#396 (after #387), T26/#427 (own input guard), T28/#437 (after #430), T39/#463 (after #459), T40/#466 (after #461), T46/#500 (after native's own T44), plus T34/#452 (inventing the estimated cap for quantile/isotonic), T48/#502 (adding an n² term), T53/#508 (first cap for AdditiveChi2, which had none — finding 91: a 28.5 s build with no cap applied). Since T47, estimate-fitting has become a *per-family admission fee*: T47 fitted 66 shapes, T51 61 shapes × 2 readings, T53 117, T54 110, T57 116+50, T58 71+40 — before the family's actual parity work is even counted. And the fitting tooling is not version-controlled: ticket after ticket closes with "Scripts and data are in `/mnt/project-files/transforms-loop/t53/`" (likewise t54–t58, `confit-field-reads-456.py`, `widths_r6.py`). The only refit script in the repo is an unrelated research artifact. So each refit re-derives its grid from prose docstrings, on whatever machine the session restarted on (T46/#500, verbatim: "the grids before the change ran on the machine before it, whose builds were within about 10% of these"), and produces coefficients that ship to end users as a claim about *their* hardware, with no calibration.

The estimates' tests cannot detect any of this going stale, because they assert the formula's arithmetic, not reality: `assert _build_estimate(24_702) <= MAX_BUILD_S` (`catalog_test.py:2541ff`), `assert D._build_estimate(2, 800) <= D.MAX_BUILD_S` (`distance_test.py:194ff`), and so on for every family. If confit slows 5× tomorrow, every one of these tests still passes; if a worker refits, the tests are edited to the new numbers in the same PR. They are snapshots dressed as checks.

**Why the design exists — and why it's still wrong.** The cap is not vanity. `_registry._builds` (`_registry.py:772-787`) already trial-builds every entry through `DuckDBInferFn` and discards the result, refusing if confit refuses. The estimate exists to refuse *before* paying that trial build, because an uncapped trial build can take 475 s (`distance`: 4,000 centres, refused by confit only after 475 s) or OOM 16 GB (`SkewedChi2Sampler` struct, `kernel.py`/PLANS). In other words: **native maintains 14 regressions because confit offers no cheap way to ask "will this build be enormous?" and no build deadline.** That missing capability was never filed as a need. The need that *is* filed (struct field-by-field built once) would shrink the coefficients, not remove the formulas.

Note also that the 14 formulas mostly re-encode the same three underlying confit mechanisms, rediscovered per family: field-by-field struct penalty (distance's k³, AdditiveChi2's F terms, power's width terms), shared-value carrying (project's n²·k, kernel's apart/coalesce variants), and list-vs-struct asymmetry (project's n²-if-list term from T48). `spline._build_estimate` already proves a generic form works: it fits over (distinct nodes, output width, features) rather than family shape, to the same 0.80–1.29× band.

**Concrete changes (staged).**

*(a) This week — make refits one command, not one research project.* Move the grids into the repo as `scripts/refit_estimates.py`: for each capped family, a shape grid (the docstrings already specify them: shapes, ranges, "slower of struct and list," warm, release), timed through the same entry point the estimate claims to model, outputting a diff of proposed coefficients. Refit PRs then become "ran the script on master X, here is the coefficient diff," reviewable in minutes. Add to `mandates/native/README.md` §"One cycle", step 6:

> A build estimate's coefficients come only from `scripts/refit_estimates.py` on a release build. Cite its output (shapes, band, master SHA), never a session script. If the grid does not cover your family, extend the script in the same PR.

Cost: 1–2 days (transcribing grids the workers already built in `/mnt`). Kills the archaeology, makes the next seven refits boring, and is a prerequisite for (b).

*(b) This quarter — file the actual missing need, and collapse the formulas.* Two moves, either of which halves the treadmill:

- **One shared estimator.** Replace the per-family polynomials with a single `_registry._estimate_build(fields, returns, features)` over generic graph features (distinct nodes as spline counts them, output fields, input features, count of shared ≥6-node values, struct-vs-list), fitted once across all families by the script in (a). Family modules keep only the `> MAX_BUILD_S → NotNative` call. When confit changes costs, exactly one fit is stale, and one script run fixes it. Risk: a generic fit will be somewhat worse per family than 0.8–1.3× — acceptable, because the cap's job is refusing 60 s builds, not pricing 6.9 s vs 7.1 s (see Finding 6 on the false precision at the boundary).
- **Or delete the need for estimates:** ask confit (as a PLANS need, with the 475 s/OOM repros native already has) for a preflight: `DuckDBInferFn` dry-run that stops after binder+lowering and reports program size, or a `build_deadline` it honors in `define_function`. Confit already has the ingredients — a 4M-token cap, `check_size` after lowering (which itself took 6.3 s on the pinned test shape, `test_sql_functions.py:434ff`) — but #456's quadratic was in the *binder*, before any of those fire, so the preflight must bound binder work too (e.g. nodes walked). If preflight says "huge" in <100 ms, native deletes all 14 formulas and the trial build becomes the only mechanism. This is the correct long-term home for the knowledge: confit knows its own cost drivers; native is currently reverse-engineering them from timing grids, repeatedly.

**Uncertainty.** I did not benchmark a generic estimator; the claim that one fit can serve all families is [INFERENCE] from spline's generic form achieving the same band and from the three-mechanism overlap. Stage (a) is valuable regardless and would produce the data to test (b) in an afternoon.

### 3. The between-loops protocol handles *requests* and is silent about *side effects*: a confit PR has no duty to notice, measure, or declare that it moved native's costs

**Evidence.** `mandates/README.md` "Between the loops" defines exactly one interaction: a need, sent "to the session of the other loop by message," recorded in the asker's PLANS; "When the serving loop serves a need, it tells the asking loop." Confit's after-merge rule mirrors it: "If the merge serves a catalog need, say so in the PLANS 'Delivered' list. Also tell the native loop's session." #456 served no need, so no duty triggered — and it was #459's PR (native-requested work) that first put the field-by-field numbers in writing ("the forest builds in 31 s… This PR does not cause that cost: the native loop traced it to #456"). For requested work the notification does happen (confit PLANS on #430: "The Transforms Loop was told, to refit its build estimates (native T26)"), but the *refit itself* is then a separate native ticket scheduled at native's discretion — T28, the #430 refit, landed 2026-10-14, eight days after #430 (2026-10-06), admittedly across the owner pause; T39/T40 followed #459/#461 within hours only because both loops were running hot that night. Between a cost-moving confit merge and its native refit, the catalog on master is simply wrong in one direction or the other: after #463's subject confit (#459 made PCA build 7.3→3.2 s), pre-refit estimates refused steps that would have built fine; after #456, "a step within an estimated 7 s can build for 18 s or more" (reading 6), e.g. an estimate of 6.5 s building in 42–44 s. There is no concept in either loop's rules of this in-between state being a defect with an owner.

**Concrete change.** Amend `mandates/README.md` "Between the loops" with a second, symmetric interaction, and add one line to confit's review checklist:

> A **cost notice** is the other direction: a merge that changes how long the other loop's artifacts take to build or serve. The merging loop states it in the PR ("Catalog costs" table, confit README), names the native estimates it invalidates, and records a refit item in the *native* loop's PLANS "Next" — filing the item is the one edit a loop may make in the other's folder. A capped family's estimate is considered stale from that merge until the refit lands; the refit is the native loop's top item, ahead of new families.

The last clause matters: today a refit competes with coverage work in "Next," and coverage is the loop's stated goal, so refits predictably slip until a reading shames them (reading 6 §7 items 2–3 are literally "measure again on #459 / #461; if builds changed, derive the estimate again" — i.e., the reading had to re-plan work the merges should have queued). With `scripts/refit_estimates.py` (Finding 2a) the refit is small enough that confit could even attach its output to the cost-notice item. Cost: an hour of doc edits. Effect: the stale-estimate window becomes a named, owned, top-of-queue state instead of ambient wrongness.

### 4. Even with a confit-side gate, native needs its own drift canary — because estimate tests today are tautological and the cap's failure mode is silent in both directions

**Evidence.** As shown in Finding 2, every estimate test asserts formula outputs. Nothing anywhere re-times a capped shape against its estimate after the fitting day. The readings do it heroically by hand (reading 8: "Builds near the caps take 0.71 to 1.34 times their estimates"), which is how the loop knows the caps are currently honest — at wave boundaries, on one machine, if the reading's author remembers. Both drift directions hurt users silently: stale-high estimates (post-#387 spline, post-#459 projections) shrink coverage — `to_native` refuses steps that would build in time, and the user just gets Python speed with no signal; stale-low estimates (post-#456) break the cap's one promise, that a served step won't hang a deployment for a minute.

**Concrete change.** One nightly (not PR-gate) timed test, `packages/sql-transform/sql_transform/native/estimate_drift_test.py`, parametrized with two near-cap shapes per capped family chosen to build in ~1–3 s (scale down from the docstring grids; the ratio is what matters), asserting `0.4 ≤ measured/estimate ≤ 2.5` on a release build, run by a new workflow alongside `nightly-campaign.yml` and filed the same way (into the existing nightly-findings issue). ~14 shapes × ~2 s ≈ under a minute of compute. Paste into native README review checklist:

> - [ ] A new or changed build estimate adds its shapes to `estimate_drift_test.py`. A test that only asserts the formula's value does not count as coverage of the estimate.

Cost: half a day. Effect: drift is discovered the next night with the family's name attached, whether the cause was a confit merge, a native spelling change (T44 invalidated four estimates by up to 1.8× — #500: isotonic put at 12.2 s what built in 7.4 s; forest put at 5.7 s what built in 8.4 s), or machine change. This is deliberately redundant with Finding 1's PR gate: the gate guards the shapes confit knows about; the canary guards the formulas native ships.

### 5. Needs are tracked in two hand-synced prose swamps, their reproductions live outside the repo, and two current needs have literally not been sent — while "the catalog no longer needs it" converts workarounds into permanence

**Evidence.** The channel is: a message to the other session (ephemeral), an entry in native's PLANS "Needs from confit" (9 open bullets, several accreting "Since T53… Since T55… Since T57…" paragraphs — the struct field-read need is now ~35 lines spanning #430, T28, T40, T47, T53, T55–T57 measurements), a mirrored hand-maintained section in confit's PLANS ("For the native catalog," ~200 lines including its own Closed/Delivered history), and a third copy of the history as one enormous "Served since this catalog began…" paragraph at the end of native's Needs section. There are no need IDs in the PLANS files (readings invent slugs in KPI front-matter — `needs_from_other_loop: [field-reads-quadratic, …]` — that PLANS never uses), no status field, and no way to diff the two loops' views except reading both.

The reproductions — the single most valuable artifact in this relationship, the thing that got #461 fixed in ~4 hours — are files like `/mnt/project-files/transforms-loop/confit-field-reads-456.py`, `confit-shared-value-reads.py`, `t44/reads3.py`, on a shared folder outside version control. If that folder dies, every "When it lands, measure again" instruction in PLANS loses its measurand.

Worse, the channel has a silent failure mode that is currently active: `mandates/native/tracker.md` records "The Confit Loop is paused since 2026-10-16: a confit need waits, and the loop works around it." Two needs in native PLANS end with "Send it to the Confit Loop when it runs again" (the small-shared-value need behind T44's `x - 0.0`; the `ln`-computed-once need from T52) — i.e., they exist only as PLANS prose, never messaged. [INFERENCE] When confit resumes under a fresh session, nothing forces a re-read of nine accumulated prose bullets to discover which were never sent. Meanwhile each workaround demotes its own need: three open needs say "The catalog no longer needs it" (two CASE trees → T12's one-tree spelling; long-OR → T35's `greatest`; small shared value → T44's `x - 0.0`) yet remain listed as needs at "low priority for confit." The message confit receives is "solved, ignore," while native permanently carries the workaround's complexity and its own downstream refits (T44→T46 being the type specimen, Finding 7).

**Concrete change.** One registry file, owned by neither loop's folder discipline problem by putting it in `mandates/confit/tracker.md#needs` (both loops already share `AGENTS.md#helpers-in-cloud-sessions` etc.), one row per need:

> | id | from→to | status (open/sent/served/worked-around) | repro (path in repo) | workaround in catalog | refit-on-serve |

with repros checked in under `packages/confit/tests/repros/` (or as real tests from the start — #461's repro *became* `test_a_struct_read_field_by_field_builds_in_about_linear_time`; require that ending for every cost need). Migrate the 9 open bullets, delete the duplicated Served/Delivered prose histories from both PLANS files (git and the registry's served rows are the history). Amend `mandates/README.md`: *"A need is sent when its registry row says `sent` with the date; a PLANS mention is not sending. A worked-around need stays `open` until served or the asking loop's supervisor closes it in the registry, naming the workaround it keeps."* And in confit README's standing duties, on (re)start: *"Read `mandates/confit/tracker.md#needs` for rows `open` and never sent; take them as sent today."*

Cost: half a day, mostly transcription; the confit loop's PLANS section already contains most repro measurements. Effect: one diffable source of truth, repros survive, the paused-loop queue is explicit, and "worked around" stops masquerading as "done."

### 6. The cap does not mean what a user thinks it means: four estimates price only the trial build, five modules each define `MAX_BUILD_S = 7.0`, and struct worst-case pricing refuses list steps that build 10× under the cap

**Evidence.** Reading 8 finding 92 (owned up to, to the loop's credit): T46's four estimates — forest, isotonic, quantile, spline — estimate "confit's trial build alone (`_registry._builds`), without the translation and the probe that `to_native` also runs" (e.g. `isotonic.py:96ff`, `quantile.py:218ff`, `spline.py:146ff`, `trees.py:106ff`), while distance/kernel/power/project estimate "the whole warm `to_native`" (e.g. `distance.py:105ff`). T53/#508 "fixed" this by making docstrings *say* which — documentation of an inconsistency, not its removal. Consequences are already in the record: `QuantileTransformer` at 8,000 quantiles passes its 7 s cap while `to_native` takes 8.8–9.1 s, 1.7–1.9 s of it in confit's Python `SqlFunction` pass that "the build estimates do not count" (native PLANS, function-body-pass need); reading 6 notes isotonic at 11,000 thresholds taking 7.2–7.3 s of `to_native` under a sub-7 estimate. Separately, `MAX_BUILD_S = 7.0` is defined independently in `isotonic.py:76`, `project.py:171`, `quantile.py:82`, `spline.py:143`, `trees.py:103` (kernel/distance/power at least import project's) — five literals waiting to diverge. And because estimates are fitted on "the slower of a struct and a list return" while a translator cannot always tell which it is serving, fast list steps are refused on struct pricing: `AdditiveChi2Sampler`, 1 sample step × 1,024 features builds in 0.85–0.92 s as a list and 8.3–9.6 s as a struct, "and the cap refuses both, as the entry cannot tell them apart" (native PLANS); Box-Cox at 960 features, 4.8 s list vs 14.4 s struct, same story. That is real coverage being surrendered to a modeling shortcut — whether the return is struct or list is known at the *step* level (`step.returns`, used in `_registry.query`), just not plumbed into the per-estimator estimate call.

**Concrete change.** Three small, paste-ready rules for `mandates/native/README.md` (Bound step) plus one refactor:

> An estimate prices the whole warm `to_native` of the estimator — translation, probe and trial build. `MAX_BUILD_S` is defined once, in `_registry.py`; a family imports it. Where the slower return prices the cap, say so in the refusal message ("as a struct").

Refactor: hoist the constant (one hour), convert the four trial-only estimates at their next refit via Finding 2a's script (the grids already time `to_native` in several cases — T40's grids were "warm `to_native` trial builds," so this is mostly choosing the other column), and pass return-kind into the two estimate calls where the struct/list ratio exceeds ~3× (AdditiveChi2, PowerTransformer) — either as a boolean term or by pricing struct and list separately where `query()` already distinguishes them. Expected effect: the 7 s promise becomes a promise about the thing users wait for, and a measurable slice of currently-refused list steps (up to ~10× headroom on the cited shapes) serves natively.

### 7. `x - 0.0` (T44) is native programming against three uncontracted confit internals at once — `MIN_SHARED_SIZE = 6`, no algebraic folding of `x - 0.0`, and per-CASE-arm sharing locality — and T46 already demonstrated the blast radius

**Evidence.** `_registry._apart` (`_registry.py:385ff`) hands every translator each number as `x - 0.0` because confit shares only subexpressions of ≥6 nodes (`packages/confit/src/specializer/share.rs:135`), `coalesce(x, NaN)` is 5 nodes, and `x - 0.0` is 7. The docstring states the other two assumptions plainly: "Confit folds only constant subtrees, so it keeps the `- 0.0`," and a translator whose reads sit in different CASE arms "reads it twice in one arm, so that confit shares it." That third assumption has already spread contortions into `isotonic.py` (test-both-ends in one condition), `encode.py` (T46's both-ends ordinal test), and `discretize.py` (T45 bisect inside a both-ends test) — three translators restructured so a value is read twice *in one arm*, a shape that means nothing to sklearn and everything to `share.rs`. The correctness side is genuinely tight: `x - 0.0` rather than `x + 0.0` because $-0.0 - 0.0 = -0.0$ while $-0.0 + 0.0 = +0.0$, and sign-of-zero is load-bearing at `ulps=0` — the same loop spent T38/#462 (403 sign-only differing fields; fixed by *adding* `+ 0.0` where the twin's BLAS sum yields +0.0) and T43/#497 (the 1×1 exception where `np.dot` preserves −0.0 and the added `+0.0` had to be withheld) learning exactly where each spelling flips a zero. `guard_test.py` pins that `_apart` preserves the sign. Good. What is *not* pinned anywhere: the performance premises. If a future confit change lowers `MIN_SHARED_SIZE`, adds `x - 0.0 → x` folding (bit-safe! a reasonable cleanup), or changes first-reader placement (#387's exact move), native loses the sharing it restructured around — silently, since no confit test asserts "`x - 0.0` read many times is computed once" and native's tests assert only bits. And T46/#500 shows the coupling cuts the other way too: within a day of T44, it emerged that guard tests reading the shared `x - 0.0` caused confit to compute it *ahead of the body and carry it through* — spline "error" at 128 features built in 9.2 s vs 4.3 s pre-T44 (256 features: 43 s) — requiring `_registry._plain`, a post-order rebuild that surgically un-shares guard reads, plus refitting four estimates. That is the #363/#387 shared-value-placement mechanism biting through a new door. Native is now maintaining a paired in/out transformation (`_apart` to force sharing, `_plain` to undo it where it hurts, `apart=`/`_reads_apart` switches in `project.py:253ff` and `kernel.py:349ff` that choose per-shape based on… the build estimates of Finding 2, so estimate drift now also flips *spellings*).

**Root cause (process).** A confit heuristic gap (a 5-node `coalesce` that #412 bound once at the SQL level still isn't shared at lowering; #459 made such reads cheaper but, with a guard present, still 32–33 µs vs 19–22 µs per PLANS) was routed around in the *client* because the client loop couldn't edit confit (`mandates/README.md`: "A loop never edits the folder or the package of the other loop") and the confit fix was deprioritized the moment the workaround landed ("The catalog no longer needs it: since T44…"). The folder-ownership rule is sound for code; applied to a one-line heuristic threshold it manufactured a permanent, cross-cutting encoding of confit's internals in the catalog.

**Concrete change.** Convert the workaround's premises into contract, in confit's test suite, and re-raise the underlying need at its true priority:

1. In `packages/confit/tests/test_shared_subexpressions.py`, add two pins: (i) a value read ≥2 times outside CASE arms — spelled both as `coalesce(x, NaN)` and as `x - 0.0` — is computed once (instruction-count style, like the existing Rust `fmul`-count test from #363), and (ii) Finding 1's shape (4): 256 `coalesce` features read by 64 sums with a `null_when` guard serves within 1.3× of plain reads. Pin (ii) is the exit condition for `_apart`: file the confit need as "make (ii) pass" (most plausibly: share a *bound let's* value regardless of node count, since #412 already decided it's computed once), and add to native PLANS, under that need: *"On serve: delete `_apart`, `_plain`, `Entry.apart_of`, `_reads_apart`, `_rbf_reads_apart` and the both-ends spellings in `isotonic.py`/`encode.py`/`discretize.py`; refit estimates (script). These exist only for this need."* Naming the deletion list now is what prevents the workaround from becoming load-bearing folklore.
2. Until then, one native test that fails loudly if the premise dies: in `guard_test.py` beside the existing sign test, assert the served SQL body still contains the `- 0.0` reads (`project_test.py` already does this for the projection switch: `assert apart not in native.sql_body`) for one bit-exact family — cheap insurance that a confit folding change shows up as a red native test in confit's own gate run rather than as a serve-time regression at the next reading.

Cost: an hour for the pins; the confit fix is confit's to size, but it replaces ~5 sites of arm-placement trickery with the straightforward spelling everywhere. Expected effect even if the confit fix slips: the coupling becomes explicit and alarmed instead of implicit and silent — which is the difference between T44-as-bridge and T44-as-trap.

---

## What is working and should be kept

- **The confit-only reproduction discipline is excellent and is why fixes are fast when triggered.** Native's #456 repro (`confit-field-reads-456.py`, confit alone, no catalog) let confit merge #461 roughly 4 hours after #456 (2026-10-14T22:42+02 → 2026-10-15T02:49+02), with a permanent scaling test. #377 followed #363 in ~2.3 h. The rule already in native's checklist ("Each new item under 'Needs from confit' has a reproduction") earns its keep — Findings 1 and 5 only ask that repros be promoted into CI and into the repo.
- **Requested-capability throughput.** Native PLANS' "Served since" list (#336–#339, #341, #346, #348, #350, #353, #358, #362, #363, #374, #375, #377, #387, #390, #412, #422, #430, #459, #461) with measurements like Box-Cox 3,448 µs/row + 15.6 s build → 52 µs + 0.18 s (#363-family trap-free work) is a genuinely productive two-loop interface. The priority rule (confit builds native needs second, ahead of its own query classes, `mandates/confit/README.md` ticket sources) is being honored.
- **The readings' measurement hygiene** — per-commit confit builds in the commit search, catching their own earlier error (finding 74: reading 5 had timed with reading 4's confit), cross-running catalog/confit versions to attribute #456, docstrings that cite SHA/date/band for every fit. When fresh, the estimates are honest: spline T46 "took on none slower than 7.05 s and refused none faster than 7.15 s." The problem is never the quality of a single fit; it's that freshness decays silently.
- **The gate running the whole repo's pytest** means confit cannot break native *correctness* unnoticed. Keep the monorepo gate; add costs to it (Finding 1) rather than inventing per-loop CI.

## Disagreements with `local://native-loop-report.md`

1. **§3 row "#430 → #427/#437" misattributes T26.** T26/#427 ("the forest and spline build estimates count the input guard," merged 2026-10-06, before #430 landed that evening) refit estimates for native's *own* T25 input guard, not for confit #430. The #430 refit is T28/#437 alone, eight days later. This matters for the category: roughly half the "confit forced a refit" instances on closer reading are native's own changes (T25 guard → T26; T44 reads → T46) forcing refits through the same brittle mechanism — evidence the treadmill is a native design property, not a confit-behavior problem.
2. **The appendix's "15 of 53 ticket commits (~28%) are refits…" bucket is mislabeled for cost purposes.** As worded ("refits of build estimates, caps, guard reshaping or numeric fixes") it is arithmetically defensible, but it bundles T27 (encoder guard), T29 (kernel probe draws), T30 (forest spelling choice), T38/T43 (zero-sign correctness) with actual estimate refits. Pure estimate-refit/new-cap tickets are T13, T26, T28, T34, T39, T40, T46 (part), T48 (part), T53 ≈ 7–9 of 53 (~15%). Conversely the bucket *undercounts* the treadmill's true cost: since T47 every new family pays a 61–177-shape fitting grid before shipping (Finding 2), which never appears as a "refit" subject. Net: the report's 28% is roughly right as "perf/cap overhead" but for the wrong line items.
3. **§3's #387 row ("old cap too strict. Detection: A/B timings") makes a requested change sound like a discovered regression.** #387's PR body: "part 2 of the native loop's SplineTransformer need (#384)." The T13 refit was planned work on a delivered need, branched the same night. Same for #459 (native's T37 need) and #430 (native repros). Only #363's quantile side effect and #456's quadratic were surprises. The report's §2 "Confit interaction" paragraph is more careful than its §3 table; the table is what a skimming reader will believe.
4. **"~8 h" for #456→fixed is ambiguous, not quite wrong.** #461 merged 4 h 07 min after #456; ~7.5 h brings you to T40/#466 completing the native refits. The report should say which endpoint it means, since the distinction (confit fix speed vs total time master carried wrong estimates) is precisely the process question.
5. **Report §3 row "#452: Count cap did not predict cost… T34: 7 s estimated-build cap" is accurate but incomplete in a way that flatters the fix:** four of the cap's estimates price only the trial build, so the 7 s cap T34 introduced never bounded `to_native` for quantile (8.8–9.1 s observed) — see Finding 6. The report notes the T46-vs-later methodology split in §2 ("Practice vs written rules" (3)) but doesn't connect it to the cap's user-facing promise.

## Out-of-category, one line each

- AdditiveChi2 shipped with no build cap at all until finding 91 caught a 28.5 s build — a review-checklist gap (no "what bounds this family's build?" item) that belongs to the testing/review category. 
- Reading 5's timing on the *wrong confit's build* (finding 74) and the two-CPython bench variance suggest the readings need a hermetic, scripted measurement environment — a reporting-tooling issue.

---

# D instructions — AstraInstructions

## Verdict

**The instructions need consolidation, not more rules.** Their strongest parts explain the oracle, package boundaries, and review responsibilities. Their main weakness is that the opening prompts, cycle, review checklist, and contract no longer describe the same process.

The highest-value change is to make the worker brief a short, authoritative execution guide. It should point to the current numerical contract and require evidence for known traps. It should not preserve an older numerical policy.

I checked the live instructions, rule diffs, relevant PR bodies and diffs, and existing regression tests. I independently counted the reading footprint. I did not run tests or reproduce historical performance campaigns. Costs and expected effects below are **[INFERENCE]**, not measured savings.

## 1. Top findings

### 1. The opening prompt teaches an obsolete bound-selection procedure

The native brief says:

> “measure the max ulp distance over >= 200 seeds … register that”

It then says:

> “A bound that is not small is a question for the owner”

These instructions are in `mandates/native/brief.md:41–46`. They do not mention derived error scales, mapped comparisons, or explicit opt-in.

The contract now requires `allow_bound=True` for nonzero bounds and says that an error-scale coefficient is “derived, not measured” (`packages/sql-transform/spec/native/catalog-contract.md:20–45`). Commit `98033e88`, [PR #458](https://github.com/ahrzb/sql-transforms/pull/458), added that distinction to the README’s cycle. It did **not** update the worker brief or the README’s review checklist. That checklist still accepts a bound measured over 200 seeds and uses the undefined threshold “not small” (`mandates/native/README.md:171–173`).

This does not prove that current implementations bypass the contract. For example, `project_test.py:113` tests explicit opt-in. It proves that the prompt’s main execution path gives incomplete instructions for the families the loop now builds.

**Process cause:** the same numerical policy has several independently maintained copies. Updating the contract and implementation did not invalidate the older worker instructions.

**Concrete rewrite:** replace the numerical paragraph in the brief and the corresponding review item:

> Before selecting a bound, read the catalog contract and the closed ruling for this family.  
> Name the applicable rule in the PR. Distinguish a measured ulp allowance from a derived error-scale bound.  
> For an error scale, implement the ruled K, S, τ, and comparison map. Measurements test that bound; they do not define K.  
> Check that a bounded entry stays Python without `allow_bound=True`.  
> If no existing rule covers the proposed bound, write a decision record.

Keep the campaign procedure in **one** README section. The brief and checklist should point there rather than restate it.

**Cost/effect:** approximately half a day to reconcile these three passages and their links. Expected effect: fewer wrong investigations and fewer reviews that accept empirical maxima as mathematical bounds.

---

### 2. “Follow the reason” needs an explicit limit, because the documents already disagree about authority

Both the native README and shared worker guide say:

> “If a rule and the situation disagree, follow the reason. Then record the decision in the PR.”

See `mandates/native/README.md:31–33` and `AGENTS.md#helpers-in-cloud-sessions:12–15`.

This clause is useful for implementation choices. [PR #410](https://github.com/ahrzb/sql-transforms/pull/410) gives a legitimate example: it corrected old reports despite “Never edit an older report,” disclosed the exception, and preserved the measurements.

But the clause does not distinguish workflow advice from owner decisions or role boundaries. There are real conflicts for an agent to resolve:

- The brief directs workers to follow cycle steps 3–7 (`mandates/native/brief.md:15–16`). Step 7 tells them to squash-merge (`mandates/native/README.md:89`). The shared opening prompt says, “You never merge” (`AGENTS.md#commands-repo-root:17`).
- For shared machinery, the README requires the **ticket** to announce the change first and serializes overlapping work (`mandates/native/README.md:157–160`). The brief instead permits a minimal change named first in the **PR** (`mandates/native/brief.md:26–27`).
- The shared brief prohibits model names in commits and immediately mandates a model-named trailer (`AGENTS.md#commands-repo-root:24–26`).

**Process cause:** a general permission to exercise judgment substitutes for a defined precedence order. Some conflicts concern permissions, not implementation judgment.

**Concrete rewrite:**

> Owner rulings and the catalog contract define the permitted behavior. Workflow discretion cannot change them.  
> Workers do not merge. Shared-file changes require an updated ticket before editing.  
> You may change an implementation procedure when its stated reason still holds. Record the changed procedure, evidence, and preserved invariant in the PR.

Then make step 7 role-specific:

> The worker opens the PR and addresses review. The supervisor merges after approval and green CI on the current head.  
> In inline mode, the loop session performs both roles.

Replace “name it first in the PR” with “ask the supervisor to update the ticket before editing.” State an explicit exception for the required attribution trailer.

**Cost/effect:** a few hours. Expected effect: less authority inferred from document order and fewer shared-file surprises. **I found ambiguity, not evidence of an unauthorized merge caused by it.**

---

### 3. The brief asks workers to know the traps, but does not transmit the lessons that identify them

The ticket instructions require:

> “the known traps: dtype or NaN handling, a kernel that depends on the platform, or an order of reduction”

(`mandates/native/README.md:139–144`).

That is a category list, not a usable memory of past failures. The cycle asks for constructor configurations and generator extensions, but it does not require a worker to show that important branches receive effective inputs.

Two incidents show why the distinction matters:

- [PR #406](https://github.com/ahrzb/sql-transforms/pull/406) says the periodic spline breach hid because the generator drew no infinities. Adding another seed would not create a value absent from its domain.
- [PR #497](https://github.com/ahrzb/sql-transforms/pull/497) fixed both a one-feature signed-zero case and an ARPACK fixture that depended on global randomness.

The repairs are already durable in code. `catalog_test.py:971` contains infinities; `catalog_test.py:1306–1323` checks independence from global random state. `project_test.py:324–349` tests the one-feature signed-zero case using exact comparison.

**Process cause:** findings become reports and family-specific repairs, but the worker prompt does not route future workers to the relevant evidence. “Read the twin” is necessary, but “constructor parameter” is too narrow: fitted state, dimensions, representation, and kernel dispatch also select behavior.

**Concrete replacement for the brief’s fixture instruction:**

> Before implementation, list the twin’s branches selected by parameters, fitted state, shape, dtype, and sparse or dense representation.  
> For each served branch, name its fixture and the input that exercises it. Otherwise, name its refusal.  
> Check these traps where applicable:  
> - NaN, NULL, infinities, signed zero, and both sides of branch boundaries;  
> - one-feature and one-output shapes, cancellation, overflow, and underflow;  
> - fit randomness and host-dependent numerical kernels;  
> - twin validation errors as well as successful answers.  
> Cite existing regression tests. Add a test only where the existing coverage misses the branch.

Link the existing determinism and spline tests beside those bullets. For reductions, link the signed-zero tests. This is a **branch-to-evidence checklist**, not another universal campaign.

**Cost/effect:** roughly half a day for the initial checklist; a short evidence table per relevant ticket. Expected effect: catches missing input domains before a large seed count creates false confidence. It should reuse, not duplicate, the tests added by #415 and #497.

---

### 4. The measurement instructions specify activity, but leave the measured quantity ambiguous

The brief asks for:

> “build and serve times for the widest configuration you test”

(`mandates/native/brief.md:49–50`).

The report template asks for “the build time” (`mandates/native/README.md#reporting-contract:69–74`). Neither defines whether this includes probing, translation, trial compilation, or the full `to_native` call.

[PR #508](https://github.com/ahrzb/sql-transforms/pull/508) explicitly documents the resulting split: some estimates cover the trial build alone; others cover the whole warm `to_native`. It fixed the estimate docstrings, but the worker measurement instruction remains unchanged.

There is also an unsound claim in the report prompt:

> “Use the same commands as the previous report. Then … the deltas are measurements, not noise.”

(`mandates/native/README.md#reporting-contract:46–48`).

The same command does not control changes in generated shapes, hardware, Python builds, or numerical libraries. [PR #496](https://github.com/ahrzb/sql-transforms/pull/496) used a stronger method: it reran the previous report’s commit on the current machine to separate machine effects from code effects.

**Process cause:** prompts name tasks—“measure builds”—without specifying the evidence contract. Successful measurement practice remains in reports rather than becoming the next worker’s procedure.

**Concrete rewrite:**

> For every timing, name the timed operation and its start and end points.  
> Record the commit, release build, environment, shape, return form, warm or cold state, and repetitions.  
> If the result supports a cap, name the quantity capped and measure that quantity.  
> Check both list returns and field-by-field struct reads where both apply.  
> Attribute a speed change to code only after measuring both revisions under the same conditions. Otherwise, label the comparison uncontrolled.

Replace “widest configuration” with “largest supported cases near each relevant limit,” while retaining the existing report workload for continuity.

**Cost/effect:** several hours to change the prompts; paired baseline runs cost extra measurement time when conditions change. Expected effect: fewer estimates fitted to different quantities and fewer misleading explanations of valid numbers. This recommendation does **not** require a new benchmarking framework.

---

### 5. The fresh-reader test is correctly scoped for readability, but it cannot validate an agent procedure

The simple-English skill gives the reader:

> “only the text, `GLOSSARY.md`, and this prompt”

It asks for confusing terms, hard sentences, and forbidden synonyms (`.claude/skills/simple-english/SKILL.md:102–110`). It does not supply the governing policy.

[PR #410](https://github.com/ahrzb/sql-transforms/pull/410) correctly identified that this check did not catch misplaced decision content. The problem was not that the small reader failed its assigned job. It was that readability was the only check being used for a different property.

The #410 repair is good: the current report format requires exact agreement with the open-record list (`mandates/native/README.md#reporting-contract:145–156`). But there are still cross-document conflicts outside that local repair. For example, the native format requests a hand-off message (`:34–36`), while shared reporting says not to message the owner (`mandates/README.md:117–118`).

**Process cause:** changes to prompts are reviewed as prose rather than as changes to a procedure. Examples also become implicit templates: #410 records that each report copied its predecessor.

**Concrete changes:**

Add one sentence to the simple-English skill:

> This review checks readability only. It does not check facts, policy compliance, or permission to change a claim.

Add this manual check to the report instructions:

> Draft from the current report format. Use the previous report only for comparison data and links.  
> Before merge, compare the draft with the reporting policy and the open records at the report’s named commit.  
> Check that owner questions occur only in “Needs the owner,” with one item per open record.  
> Check that the YAML list names the same records.

For changes to agent instructions, require a manual walk-through of one worker launch and one relevant exception. Check the resulting actions against the contract and role boundaries.

Remove the native hand-off requirement; shared reporting should own delivery.

**Cost/effect:** a few hours to consolidate instructions, then a short manual review. Expected effect: detects procedural contradictions that a glossary-only reader cannot see.

**Do not restore the prose hook.** [PR #405](https://github.com/ahrzb/sql-transforms/pull/405) states that the owner rejected automatic text checking. Sentence-length automation would not have caught #410 anyway.

---

### 6. Reading cost is real, but “reread everything each cycle” is not established

The brief requires all four of these example modules before code:

> “`scalers.py`, `impute.py`, `select.py`, `encode.py`”

(`mandates/native/brief.md:15–18`).

I counted **691 lines** across those files. The two briefs, native README, catalog contract, and root `AGENTS.md` contain **3,544 whitespace-delimited words**, before the ticket, twin source, shared worker guide, or examples.

Those counts are reproducible file-size facts, not token or latency estimates. I found no instruction requiring every linked document to be reread on every iteration. Therefore, I would not claim a measured per-cycle cost.

The more important problem is relevance. A kernel or projection worker must read four early-family examples, while the brief does not require the current `project.py` or `kernel.py` pattern.

**Process cause:** the initial examples became a permanent reading list. The documents also repeat merge, reporting, numerical, and package-boundary rules rather than routing by role and task.

**Concrete rewrite:**

> Before coding, read the catalog contract, this ticket, and one existing translator with the same numerical and output structure.  
> Read its focused tests and the shared helpers it uses. The ticket names the example.  
> Load reporting instructions only when writing a report. Load the simple-English skill only for its stated text types.

Cut duplicated numerical procedures from the brief. Remove publishing and common reporting rules from the native report format. Keep the detailed supervisor protocol in `AGENTS.md#helpers-in-cloud-sessions`.

**Cost/effect:** half a day to consolidate and repair pointers. Expected effect: lower irrelevant reading and fewer stale copies. The main gain is consistency, not a demonstrated token saving.

## 2. What is working and should be kept

- **The twin-first procedure has the right altitude.** It requires the operation sequence, dtype, source inspection, and exact-class registration, with a reason for the latter (`mandates/native/README.md:55–65,111,167–169`).
- **The cross-loop boundary includes an executable next action.** A missing capability requires a reproduction, a named refusal, and a recorded need—not merely “do not edit confit” (`mandates/native/README.md:95–105`).
- **The worker guide explains harness-specific failures.** It explains why same-account GitHub reviews may not reach workers and requires direct delivery (`AGENTS.md#helpers-in-cloud-sessions:161–171`). This is useful operational knowledge.
- **The #410 repair has a checkable completion condition.** The owner-question section and YAML must match the open records. Keep that.
- **Simple English is conditionally loaded.** `AGENTS.md:7–11` exempts ordinary PR bodies, working plans, and code comments from the full writing procedure. Keep that scope.
- **Known failures have become shared checks.** The fixture determinism check is stronger than another warning to “remember random_state.”

## 3. Disagreements with the report

1. **The infinity-fixture fix did not wait until October 14.** The report places T20/#415 in the resumed-work group and gives the spline/generator repair an “Oct 6–14” range. Commit `a7cd5aa8` is dated October 6, and [PR #415](https://github.com/ahrzb/sql-transforms/pull/415) was created and merged that morning. This matters when assessing how quickly the loop learned from the omission.

2. **“Practice vs written rules” understates the prompt problem.** The documents themselves disagree about shared-file permission and worker merging, and retain an older bound-selection procedure. This is not only agent noncompliance.

3. **The report’s explanation of #410 is supported, but needs a distinction.** The fresh-reader check missed policy compliance because it was not asked to check that property. Its failure is not evidence against the readability check.

4. **The supplied rule-diff artifact is incomplete for this category.** It does not contain the report-format diff for #410 or the simple-English removal in #405. I used the PR diffs and live files for those claims.

## 4. Out-of-category observations

- **Measurement reproducibility:** several campaign records point to session-mounted scripts; I did not establish whether another reviewer can retrieve and rerun all of them.
- **Numerical assurance:** the sampled tests demonstrate existing targeted protections, but this review does not establish the soundness of the derived bounds or their cross-host coverage.

---

# D instructions — OpusInstructions

# Category D review: the loop's instructions as agent prompts

**Verdict.** The rule docs describe the loop as it was on 2026-10-05: bit-exact scalers, run by workers, with a supervisor reviewing each PR. Since then, findings 51–92 changed what every native ticket has to do. Only one of them reached the operating rules: #458's error-scale sentences. The other lessons were written into PLANS.md, the board and the readings as running prose that never becomes an instruction.

The "follow the reason, record the decision" clause did not stop the drift. #408 added it to the README in the same commit that removed most of the README's stated reasons. Departures from the rules have been silent, except in #410.

The biggest gains are a trap checklist built from the finding ledger, moving the checklist and the trap fields out of the worker-only section, and making departures change the rules.

What I could not verify: whether the 18-item checklist would have caught specific bugs earlier. That is marked [INFERENCE] where it comes up.

---

## 1. Top findings

### F1. The cycle is missing the work that now dominates every ticket: build estimate and cap, input guard, kernel probe, and the error-scale bound order

**Evidence**
- `mandates/native/README.md` steps 3–7 and `mandates/native/brief.md` never mention:
  - an input guard;
  - a kernel probe;
  - a build estimate or the 7 s cap;
  - `allow_bound`;
  - release-build timing.

  I grepped README, both briefs, `workers.md` and `catalog-contract.md` for "build", "guard", "probe" and "estimate". The only build-time instruction is the brief's "Record build and serve times for the widest configuration … the supervisor forwards width limits to the confit loop". That reflects the Oct-5 idea that confit owns width limits.
- The cap exists only in code. `MAX_BUILD_S = 7.0` is defined separately in `isotonic.py:76`, `project.py:171`, `quantile.py:82`, `spline.py:143` and `trees.py:103`. `kernel.py`, `distance.py` and `power.py` import it from `project`.
- 14 of the 92 numbered findings are about build estimates or caps: 22, 23, 41, 56, 58, 61, 64, 66, 68, 80, 87, 89, 91, 92. This is the largest single class.
- Two of them come straight from the missing rule:
  - **Finding 91.** `AdditiveChi2Sampler` "has no build estimate, so no cap": `to_native` took 28.5 s. T53/#508 fixed it.
  - **Finding 92.** T46's estimates time the trial build alone, while T47/T48/T51/T52 time the whole warm `to_native`. So the 7 s cap measures two different things.
- Step 3 ("Read the twin") covers order, dtype, NaN and constructor parameters. It does not ask about input validation, which is what decides the guard. It does not ask about paths the twin takes only inside a composition: finding 40 (object arrays skip the finiteness check), 70 (sparse-input branch), 57 (row dtype). It does not ask about degenerate fits (78, 79).
- Since Oct 6, `README.md` changed once in substance (#458), and both briefs only had a path fix (b4b8d033). Over the same period, findings 51–92 were logged.

**Process root cause.** Nothing has a home for a cross-cutting lesson. `workers.md` §3 says "move what the ticket taught into PLANS.md", but PLANS is a list of open work. So lessons became per-family prose there: PLANS grew from 1,163 to 5,776 words between `49acad5d` and `40f6ab17`. Readings number the findings but never feed them back into the steps.

**Change.** Add `mandates/native/brief.md#traps` as the single source. Point to it from README step 3. Append it as part 3 of the worker prompt (`workers.md` §4: shared brief, native brief, traps.md, ticket). Add a rule to `workers.md` §6 after-merge: "If the PR fixed a finding that a later ticket could repeat, add one line to traps.md in that PR."

Draft for `traps.md`, about 450 words:

```
# Native traps — check each before you translate; mark applies / n.a. in the PR's "Twin" section
Twin
1. Validation: list every input the twin rejects (±inf, NaN, sign, range). Each needs the input guard; the entry never traps where the twin answers. (T25; findings 27, 44)
2. Compositions take other paths: object arrays skip finiteness (40); a sparse input runs another branch (70); sparse output differs from dense (71); dtype follows the whole row (57). Test as a Pipeline/ColumnTransformer part too.
3. Degenerate fits (inf/huge components, IndexError in fit): refuse by name (78, 79).
4. Read the installed code path, not the docs: interp1d delegates to np.interp (38); sklearn bugs are followed, then refused at the shape that breaks (37).
Numerics
5. DuckDB orders NaN above every number; numpy comparisons with NaN are false (1).
6. Signed zero: the twin's sign depends on np.dot shape, the call's row count, .toarray(), SIMD min/max lane order (39, 77, 81, T58). An error-scale bound treats -0.0 = 0.0: check signs separately.
7. Reduction order: einsum, pairwise, BLAS, LAPACK, FMA builds (2, 5, 42). A host-dependent kernel needs a bound that holds for another host's twin (T58).
8. libm kernels: serve only behind a kernel probe run through confit; any probe failure declines (24). Tests pass on both probe branches, AVX-512 locally and glibc on CI (45, 88).
9. Overflow: use the twin's own overflow edges (cosh, expm1) (88, T55, T56).
Fixtures
10. Every random fit gets random_state; never numpy's global RNG (82).
11. Each configuration seeds its own shapes (8); edges include ±inf, ±0, NaN, NULL, booleans, all-missing columns (7, 51/52, 55).
Build cost
12. If the SQL grows with the fit, add `_build_estimate` and refuse past MAX_BUILD_S (7 s) with NotNative (91). It times the whole warm `to_native` — translation, probe, trial build — release build, the slower of list and struct (89, 92). Fit on ≥30 shapes; cite the ratio range on ≥20 fresh shapes.
13. Counts are not costs: cap by estimated seconds (22, 68).
14. Probes: one transform call per feature at most, never per category (65, 84).
15. Depth: confit parses ~500 nested levels, Python recurses 1,024. Use balanced trees; `to_native` raises only NotNative (86).
16. After a confit merge that touches lowering, sharing or struct reads, rerun the estimate tests and refit (58, 61, 64, 76, 80, 87).
Measurement
17. Timings: release confit (`maturin develop --release --skip-install` in the Nix shell, 67), on the merged head (23).
18. A comparison script reads every output field, list elements included (75).
```

**Optional check (about 30 lines of pytest).** In `coverage_test.py`, assert that every family module either references `MAX_BUILD_S` or is on an allowlist of fixed-width families (scalers, impute, select). This would have failed at T49/T50, before finding 91.

**Cost and effect.**
- About 2 h to write. Adds about 600 tokens to each worker prompt and inline read.
- [INFERENCE] Of the 14 catalog findings in readings 6–8 (77–92), about 9 (77, 81, 82, 86, 87, 88, 89, 91, 92) fall under a checklist line. Each was found a wave late, by a reading.

### F2. The trap fields and the review checklist sit in the worker-only section, but 36 of 58 tickets ran inline with no ticket and no review

**Evidence**
- The README's "Every native ticket also names … the known traps" and its review checklist are both under "Tickets and review". That section only applies in worker mode. The inline cycle (steps 1–9) goes from "Done" straight to "Squash-merge it".
- `git log -G "^## T[0-9]+" -- mandates/native/tracker.md`: the last ticket section appears at T22/#420 (Oct 6). T23–T58 had no ticket text at all.
- `gh pr list`: every native PR with a comment is #404 or earlier (#356–#404). PRs from #414 on have zero comments.
- Ticket PRs from #414 on merged at a median 11 min after opening (p90 25 min), against 15/72 min before.
- The bugs found later (77, 81, 82, 86, 89, 91, 92) were all caught in milestone readings. In practice, readings became the review, and each bug stayed on master for a wave.
- The board itself shows the drift. While T47 was in flight, the table at `af89daed^` was empty. Today `tickets.md` holds about 1,400 words of merged-ticket summaries, against its own header ("A merged ticket leaves the board").

**Process root cause.** Material that both modes need was placed under the branch for one mode. Inline was supposed to be for small items, so it got no review step. It then became the only mode used: 45 of the 47 PRs since #414 came from one session's branch.

**Change.** Move the checklist into the cycle as a step every PR runs:

```
9. Self-review. Copy the review checklist into the PR body under "Self-review", tick each item, and give its evidence (a line number, a command and its output, a finding number). In worker mode the supervisor checks this block. In inline mode it is the review. A PR without it does not merge.
```

Add these items to the checklist:
- "traps.md: each item marked";
- "build estimate and cap, or the reason the width is fixed";
- "`to_native` raises only NotNative".

**Cost and effect.** About 30 min. [INFERENCE] It moves the reading-time catches to PR time at no extra agent cost: the inline session already writes long PR bodies (#513's runs to many paragraphs).

### F3. "Follow the reason, record the decision" causes silent drift: reasons were removed, departures go unrecorded, rules never change

**Evidence**
- **Reasons removed.** #408 (c9245858) added "The sections below give working rules and their reasons. If a rule and the situation disagree, follow the reason" to the README. In the same diff it deleted reasons and examples:
  - step 6 lost "(BLAS sums, NumPy pairwise reductions)";
  - step 5 lost "(strings, categories, a narrower domain)";
  - step 7 lost "(`--no-sync` keeps the environment the install step built, as the shared brief says)".
- **Branch rule not followed.** 45 of the 47 PRs since #414 came from `claude/project-thread-2pg6vk`, against README step 1 ("named `claude/native-<name>`") and the owner's ruling in `workers.md` §1 ("Use one branch for each ticket"). No PR records this; I grepped the git log for "project-thread" and "branch rule".
- **One-family rule not followed.** 23 of 52 ticket commits touch more than one family module or the shared machinery, against "One PR holds one family" and "the ticket says so first". Examples: T25/#423 touches 8 modules plus `_check`, `_helpers` and `_registry`; T53/#508 touches 7 modules. Only #388 (T11) announced a shared-machinery change.
- **The one recorded departure changed nothing.** #410 said plainly: "One rule this PR does not follow … 'Never edit an older report'". Both copies of that rule (`reporting.md:112`, `report-format.md:22`) are still unconditional.

**Process root cause.** The clause only fires when the agent notices a conflict. Even then, it writes to the PR body, which no later cycle reads. When the environment simply makes a rule impossible (a branch assigned by the harness), nobody notices, and the rule decays quietly.

**Change.** Replace the clause in `README.md` and `workers.md`:

```
Each rule here states its reason. If the reason does not hold for your case, depart from the rule and add a line to the PR body under "Departures": the rule, why its reason does not hold, what you did. If a rule has a departure for the second time, change the rule in that PR. A rule that the environment makes impossible is changed, not departed from.
```

Apply it now to three rules:
- **Branch.** "Use the branch the session gives you; one PR at a time from it."
- **Scope.** "A ticket is one family, or one cross-cutting change (a refit, a machinery change) named as such in its title."
- **Old reports.** "Never edit an older report, except to remove content that the policy forbids, with a note at its top" (the #410 case).

The reading's retrospective can grep merged PR bodies for "Departures".

**Cost and effect.** About 30 min. It turns three rules that are currently dead into rules that are true.

### F4. The simple-English rule applied to agent prompts made them longer, removed reasons, and adds nothing to execution

**Evidence**
- Word counts before and after the STE rewrites (#401 e2720646 for the shared docs, #408 c9245858 for the native ones), with no rule added:

  | doc | before | after |
  |---|---|---|
  | `workers.md` | 1,148 | 1,443 |
  | `reporting.md` | 649 | 810 |
  | `worker-brief.md` (shared) | 313 | 360 |
  | `mandates/README.md` | 407 | 541 |
  | native `README.md` | 1,145 | 1,380 |
  | `goal.md` | 402 | 518 |
  | `report-format.md` | 921 | 1,072 |

  That is +15–33% per doc, and 4,985 → 6,124 words (+23%) across the seven.
- The native `worker-brief.md`, which was never rewritten, stayed at 386 words. It is the densest and most concrete of the rule docs; it is the only one with `NATIVE_SEEDS=200`.
- The skill's sentence rules are the mechanism: "Do not join two sentences with a semicolon or a dash", "Put the condition first", at most 20 words per instruction, "Do not remove words". Together they turn "a reason in a parenthesis" into either a separate sentence or nothing, and the rewrite mostly chose nothing (F3).
- The skill's own scope already says "Working files for agents … need only be clear to the next agent". But AGENTS.md rule 1 and the native README's last "Rules" bullet pull the README, `goal.md` and `report-format.md` back under STE.

**Process root cause.** One document had two readers: the owner, who maintains it, and the agent, who executes it. The rule was written for the owner reader.

**Change** (owner-authored rule, so ask the owner first). In AGENTS.md rule 1, and in the native README's "Rules" bullet, limit simple English to "reports, decision records, the package spec pages and large-PR summaries". Add:

> Loop READMEs, worker briefs, `workers.md`, `reporting.md` and `traps.md` are agent prompts. Write each rule as an imperative with its reason in the same sentence. Examples and parentheses are welcome.

Restore the three parentheticals from `c9245858^`. This needs no checker, so it respects the #405 ruling against automatic text checks.

**Cost and effect.** An edit, plus owner approval. It recovers about 1,100 words of every-cycle read and puts back the reasons the clause depends on.

### F5. The rule docs contradict each other and the contract on bounds, merging and machinery, and point at stale examples and commands

Each item below was checked in the live files.

- **Bounds**
  - README step 6 says "measure the largest distance in ulps … Register that ulp bound". Later in the same step: "Do not use the measured value as K". The contract (`catalog-contract.md:28`) says "K is derived, not measured".
  - The brief's "Discipline" section has only the measured-ulp path. It never mentions `ErrorScale`, the `allow_bound=True` opt-in from #404, or what a "not small" bound means. The README checklist uses the same undefined "not small".
  - Measured ulp bounds still exist (`function.py:277`), so the real order of preference is: bit-exact, then an owner-ruled error scale, then a measured ulp bound only for libm kernels behind a probe. No document says this. F1's step-6 wording should.
- **Merge.** The brief tells workers to follow README "steps 3-7". Step 7.5 is "Squash-merge it with the expected head SHA". The shared brief says "You never merge".
- **Shared machinery.** README: "If a ticket must change them, the ticket says so first … the change to the machinery is its own ticket and goes first." Brief: "keep it minimal and name it first in the PR." These are two different rules.
- **Stale examples.** The brief tells workers to copy "the existing family modules (`scalers.py`, `impute.py`, `select.py`, `encode.py`)". Those are the Oct-5 bit-exact patterns. The patterns for guard, probe, `ErrorScale` and `_refuse_past_cap` live in `project.py`, `kernel.py` and `power.py`.
- **Stale command.** The shared brief's install line is `uv sync --locked --group spark --reinstall-package confit`. The session hook now runs `uv sync --locked --group spark` inside the Nix shell (`.claude/hooks/session-start.sh:57`). Building confit there needs `--skip-install` (finding 67, `worker-environment.md`). Neither brief mentions either.

**Root cause.** Briefs and checklists were frozen copies of the README. Edits went into one copy only: #458 changed README step 6, not the brief.

**Change.**
- Make the native brief point at README steps instead of restating them. Keep in the brief only what the README lacks: what not to touch, and the PR body sections.
- Change the examples line to: "copy the pattern of `project.py` (error scale, cap) and `power.py` (guard, probe) for a bounded family, and `scalers.py` for a bit-exact one."
- Change the install line to: "The session hook runs `uv sync --locked --group spark` in the Nix dev shell. To rebuild confit: `maturin develop --release --skip-install`."

**Cost.** About 30 min.

### F6. State files read every cycle have become prose logs, and the report rules are copied in up to four places, which is the same structure that caused #410

**Evidence**
- PLANS.md says "open work only", yet holds 5,776 words. "Next" has one item. "Needs from confit" is lines 65–280, mostly history: "confit #430 served the input guard's part (2026-10-06) … Since T53 … Since T55 … Since T57 …".
- `tickets.md` has an empty table and 1,563 words of T53–T58 summaries.
- Every cycle reads README, PLANS and tickets: about 8,800 words.
- The root cause of #410 was two files owning one rule. #345's `report-format.md` §7 said "Then the questions for the owner, each with the evidence the owner needs to rule". #352's `reporting.md` added a "Needs the owner" section. Readings 1–3 therefore listed the questions twice. #408 removed §7 about 35 minutes before #410, and #410's fix added a third statement.
- The same split still exists for other rules:
  - "Never edit an older report": 2 copies.
  - The fresh-reader test: 2 copies.
  - "Measure every number again on master": 4 copies (`reporting.md:92`, `README.md:187`, `report-format.md` Style and §"worker-driven").
  - "A worker never writes one": 2 copies.

**Change.**
- In `workers.md` §3, add: "The board holds only rows and ticket sections in flight. A merged ticket's results go to its PR and the next reading. Delete the summary."
- In the PLANS header, add: "An item says what is open and its numbers today. Its history is in the PRs and readings it cites."
- Give each report rule one owner. Shared rules (`reporting.md`): triggers, KPI block, placement, the immutability rule and its exception, Needs the owner. Native-only rules (`report-format.md`): the section shape. Delete the other copies.

**Cost and effect.** About 1 h. [INFERENCE] PLANS plus tickets should drop from about 7,300 to about 2,000 words. It removes the place where the next #410-style divergence would come from.

---

## 2. What is working and should be kept

- **Concrete, copy-paste commands with reasons**: "`git add` … because pre-commit skips untracked files", and "Never drop `--group spark`, because that uninstalls pyspark". These are the right level of detail.
- **Finding slugs that persist across readings** (`report-format.md` §5). This is what made the trap checklist above possible to build. Keep it, and treat it as the input to `traps.md`.
- **"Measure, then write" and "never copy a number from a PR description"**. Readings re-measuring caught findings 74, 75, 83 and 91.
- **"Decisions stay in their records"**. Readings 4–8 kept owner content out of the reports.
- **The single-sentence `goal.md`, which links to the contract** (b4b8d033). This is the right split between the goal and the oracle contract.

## 3. Disagreements with the report

- **§2 "Supervisor and review" describes the written process as current practice.** It says reviews happen "on GitHub and direct to worker" and that each ticket "names … traps". In fact no ticket section has been written since T22 (Oct 6), and no native PR after #404 has a comment. From T23 on, work ran inline from one session, essentially unreviewed until the next reading.
- **§2 "Practice vs written rules" lists four gaps and misses the three largest.** It omits the branch rule (45 of 47 PRs on one branch), one family per PR (23 of 52 ticket commits cross-cutting), and the board used as a changelog.
- **The #410 explanation is supported but incomplete.** "Report-format instructed that content" is right for §7 (#345 to #408). But the structural cause was that the rule was split between `report-format.md` and `reporting.md`, and §7 had already been removed by #408 before #410 merged.
- **The input `local://native-rule-diffs.txt` has no `report-format.md` diffs**, although the task says it covers that file. `report-format.md` has nine commits: #345, #352, #366, #388, #408, #410, #458, #460, #501. Rule-change analysis built only on that file will miss these.
- **Minor.** "11 commits changed rule files" undercounts once `report-format.md` is included.

## 4. Out-of-category notes, one line each

- `MAX_BUILD_S = 7.0` is defined in five modules; it should be one constant in `_registry`.
- About 28% of ticket commits are refits after confit changes. A confit CI job that runs native's `-k build_estimate` tests would catch drift at the confit PR.
- Reading 7 needed at least three fresh-reader rounds (see its commits). The test's cost per report is not tracked.
- PR #432 is still open and stale.
- [INFERENCE] Squash-merging 45 PRs from one reused branch relies on the session resetting that branch after each merge. A merge-base check would make that safe.

---

# D instructions — MuseInstructions

# Category D — the loop's instructions as agent prompts

**Verdict.** The instruction *architecture* is good (shared brief → loop brief → ticket, state in files, decision records for the owner). The instruction *maintenance* failed: the documents a worker actually receives froze on 2026‑10‑05/06 while the catalog contract they must implement changed five times afterwards, every fact is stated three or four times so updates land in one copy and silently contradict the others, and 92 numbered findings were never distilled into the one place a fresh worker reads. What the loop learned is in the code, the PLANS file (5,776 words) and the reports (71,099 words total) — none of which is in a worker's opening prompt. The supervisor's reviews absorbed the gap (#367 is the clearest specimen); that is expensive, unscalable compensation for a stale prompt.

I am uncertain about one thing throughout: ticket sections are deleted from `tickets.md` on merge (`AGENTS.md#helpers-in-cloud-sessions` §3), so I cannot verify how much of the missing material supervisors pasted into individual tickets. Where I infer harm from a stale brief rather than observe it, I mark [INFERENCE].

## 1. Top findings

### 1. The worker brief still teaches the pre-ruling bound discipline — the opposite of current law

`mandates/native/brief.md` has had **zero substantive edits since its creation** in #366 (`49acad5d`, 2026‑10‑05); its only later commit is `b4b8d033`, a link/path change. Its "Discipline" bullet (lines 42–47) says: bit-exact unless the twin's order cannot be reproduced, *"then measure the max ulp distance over >= 200 seeds … register that"*. That predates, and contradicts in sequence, everything that followed: the 2026‑10‑06 ruling (bit-exact by default, a bound above 0 only via `to_native(step, allow_bound=True)`, #404/T19 #414), error scales with a **derived** K (T24 #421, T37 #458), input guards that trap where the twin raises (T25 #423), and the 7 s estimated-build cap (T34 #452). `grep allow_bound` returns 0 hits in both `mandates/native/README.md` and `worker-brief.md`; README step 6 got exactly one ErrorScale sentence, riding inside code PR #458 (`98033e88`), and the brief never got it. A worker launched today on a projection-type family is instructed to register a *measured* bound — precisely what README step 6 now forbids ("Do not use the measured value as K") — and is told nothing about `allow_bound`, guards, or build caps. It is also told to read `scalers.py, impute.py, select.py, encode.py` as exemplars: all pre-bound families; the bounded exemplars (`project.py`, `distance.py`, `kernel.py`) are not named.

**Root cause (process).** Rule updates ride along in whichever code PR the author is holding and touch the file nearest the work — report-format accreted bound-reporting bullets in #388, #458 and #501 (`af89daed`), README got one sentence in #458, the brief got nothing. No rule obliges a contract-changing PR to update the prompt documents.

**Change (paste-ready).** Replace the brief's bound bullet with:

> - Bit-exact (`ulps=0`) by default. A bound above 0 serves only under `to_native(step, allow_bound=True)` and only as a closed decision record states it: an ulp bound measured over ≥200 seeds (`NATIVE_SEEDS=200`), or an error scale `ErrorScale(k, s, tau[, g, on])` whose K is *derived in the record* — measure max |entry−twin|/(eps·S) over ≥200 seeds as evidence, never as K. Cite seeds, maximum, date in the docstring. Never raise a bound to make a failure pass; find the twin's order first. A family with no closed record for its bound stops and writes one in `decisions/open/`.
> - The entry traps where the twin's validation raises and nowhere else (input guard). Refuse by name (`NotNative`) before building anything whose build estimate passes 7 s; state in the docstring whether the estimate times the trial build alone or the whole warm `to_native`.

And add to the README review checklist: *"[ ] If this PR changes the catalog contract or implements a ruling, the same PR updates `worker-brief.md` 'Discipline' and README step 6 in the same words."*

**Cost/effect.** ~1 h to write, +~90 words per worker prompt. Removes the single most dangerous stale instruction in the system [INFERENCE: no merged PR I sampled actually registered a measured K — supervisor review and ticket text appear to have compensated; the hazard is for the next fresh worker, not a past error].

### 2. There is no trap checklist, and the fixture rule enumerates the wrong space

README step 5 requires "one factory for each configuration that changes `transform`" — constructor-parameter space. Nearly every recorded escape lived in **input-value space or host space**, which no instruction enumerates: ±inf never drawn (#406/#415), booleans (#399), all-missing string feature with `[None]` categories (#347), DuckDB NaN ordering and einsum order (#329), −0.0 vs +0.0 in 403 fields (#462) and again in the 1×1 `np.dot` (#497), an ARPACK fixture on the global RNG (#497), a kernel argument shifted ≥1 far from zero (#509), a bound that holds on this host's LAPACK and fails on another's (#513: 46/80 alternate twins). README step 3 names exactly one edge ("what it does with NaN"). The ticket template (README:144–145) requires each ticket to name "the known traps" but gives the supervisor no list to name them *from*. Meanwhile the loop dutifully records each lesson as a numbered finding in reports §5 (up to finding 92) and, per `workers.md` §3, "moves what the ticket taught into PLANS.md" — a 549-line file whose "Ruled 2026‑10‑06" section is mostly history of what *already shipped*, in flat violation of its own header ("open work only… Remove an item when it lands", PLANS:3–5). The memory exists; it is just not addressable by a worker.

**Change.** Add to `worker-brief.md`, before "Discipline", a checklist the PR body must answer item-by-item (n/a requires a reason). Each line is keyed to the incident that earned it:

> ## Traps — answer each in the PR, with the test that covers it
> - Special values per numeric feature: NaN, NULL, ±inf, ±0.0, subnormal, far-from-zero (cancellation in any squared-distance/kernel argument). [#406, #509]
> - Dtypes the step hands the twin: float32 fits, boolean features (DOUBLE 0/1), strings incl. a feature missing in every fit row. [#399, #402, #347]
> - Summation order: pairwise, einsum, BLAS/LAPACK, scipy CSR stored order, or left-to-right? Name it and how you verified it; signed zero of an empty/zero sum (+0.0 seed). [#329, #462, #497, #512]
> - Kernel probe, if any: probes confit (the engine that serves), never the DuckDB test oracle; any probe failure declines with `NotNative`, never raises. [#367]
> - Trap parity: rows where twin validation raises trap; no row the twin answers traps. [#423]
> - Sparse output/`transformer_weights`: dense answer; −0.0 from `.toarray()`; +0.0 after a weight. [#455, contract]
> - Width is three numbers (features × instances × output fields): build estimate over a grid of shapes, what it times, 7 s cap; expression depth (confit refuses ~512-deep nesting; 1,024-deep chains raise RecursionError → decline by name). [#452, #502]
> - Fixture determinism: explicit `random_state` everywhere; per-configuration seeds, never shared shapes. [#497, #347]
> - Host variation, for any BLAS/LAPACK/kernel-dependent twin: verify the bound against a second kernel/BLAS setup, or derive it host-free. [#513]

**Cost/effect.** Half a day to distill from the findings already catalogued; ~170 words in every prompt, replacing supervisor archaeology. This is the highest-leverage addition available: 6 of the scout report's 23 incidents are "numeric/oracle edge semantics" and 4 are fixture gaps, and every one of them was knowable from the previous one.

### 3. Triplication has produced literal contradictions an agent must resolve unaided

Every operational fact is stated in three or four places (coverage regeneration 4×; the 200-seed rule 4×; one-family-one-PR 3×), and the copies have drifted apart:

- The brief tells a worker to follow README "steps 3–7" (`worker-brief.md:15`); step 7.5 says *"Squash-merge it"* (`README.md:89`); the shared brief it just read says *"You never merge"* (`AGENTS.md#commands-repo-root:17`, `workers.md:43`).
- The cycle orders step 8 "Update PLANS" *after* step 7's merge, so read literally the PLANS edit cannot be in the PR; the brief meanwhile says remove your item "when it lands", and `workers.md` §6 has the supervisor remove it if the PR didn't. Three owners, no sequence.
- Shared machinery: README's ticket rules require the ticket to say so first and make concurrent machinery changes their own prior ticket; the brief's "What you must not touch" section then permits the same files if "minimal and named first in the PR".
- The shared brief bans model names "in commits" (line 24) and in the next breath mandates ending every commit with `Co-Authored-By: Claude Opus 5.5` (lines 25–26) — a hard-coded model identifier that has already drifted in practice (recent commits read "Claude Opus 5.5 (1M context)").
- `git add` the new files (README step 7) vs `git add -A` (shared brief).

**Root cause.** No fact has a single owning document; the layering rule ("the brief holds only what the shared brief does not say") was enforced between the two briefs but not against the README, which both briefs then re-summarize.

**Change.** De-duplicate by role: README owns the cycle and checklist; the worker brief shrinks to not-touch + trap checklist (finding 2) + PR-body template, with pointers into README sections instead of restatements. Split README step 7 into "(inline) … merge" / "(worker) … stop at green CI; the supervisor merges", and move "Update PLANS" inside the PR step. Delete the model-name trailer from the shared brief (keep `Claude-Session:`), resolving the self-contradiction in favour of line 24. Cut README's 30-line top index, which restates `mandates/README.md`.

**Cost/effect.** ~2 h, net −400 to −600 words from the worker's mandatory read (currently brief 360 + brief 386 + README 1,461 + contract 1,114 words before any code; a supervisor cold-start is ≈11,900 words across README/workers/reporting/loops README/AGENTS/goal/PLANS/tickets). Fewer copies means future updates like #458 land everywhere because there is only one place to land.

### 4. "Follow the reason, record the decision" helps once and is structurally unauditable

The clause (verbatim in `mandates/native/README.md:31–33`, `AGENTS.md#helpers-in-cloud-sessions:12–15`, `mandates/confit/README.md:25`) produced the best-documented deviation in the repo: #410's PR body has a section literally headed **"One rule this PR does not follow"**. That is the clause working as intended. But three structural defects remain. (a) Many rules state no reason to follow: of README's five "Rules" bullets only "Register exact classes only, *because*…" gives one; "Never edit an older report" never says why (the tracker diffs immutable readings) — an agent told to "follow the reason" must invent it. (b) The simple-English rewrites actively deleted reasons (finding 5), sawing off the branch the clause sits on. (c) Recording the deviation "in the PR" contradicts the loop's own state principle — state "does not live in the memory of any agent" but in repo files (`workers.md`, Switching); a merged PR body is not read by the next cycle. Consequently #410's exception was never codified: the rule still stands absolute in two files, so the next agent facing owner-ordered report surgery must re-break it blind. I found no other recorded deviation in the path-filtered history, so either the clause is invoked once in 165 commits, or other invocations went unrecorded — indistinguishable from the outside, which is itself the defect.

**Change.** Replace the clause in all three files with:

> If a rule and the situation disagree: name the rule, the reason it exists, and why this situation is outside that reason, in the PR. In the same PR, amend the rule or add a dated exception under it — a deviation recorded only in a PR body is not recorded. If the rule states no reason, treat it as binding and raise it in `decisions/open/`; do not guess its reason.

Backfill the missing reasons and the #410 exception in one pass, e.g. under "Never edit an older report": *"because reports are compared as immutable readings; exception (2026‑10‑06, #410): owner-ordered removal of content the format forbids, annotated at the top of each edited report."*

**Cost/effect.** ~2 h. Converts an honour system into a ratchet: every deviation either improves the rule or is visible as rule-breaking at review.

### 5. Simple English is applied at the wrong altitude: the agent's prompt is classified as owner prose

`AGENTS.md` rule 1 and README's Rules classify `mandates/native/README.md`, `goal.md` and `report-format.md` as "text that the owner reads", subject to the STE-derived skill — although the skill itself says working files for agents "need only be clear to the next agent". The result was predictable and measurable: the #408 rewrite inflated README 1,145 → 1,380 words (+20%), #401 inflated `workers.md` 1,148 → 1,443 (+26%) and goal 402 → 518, while *deleting* agent-useful content under a commit message claiming "The content is unchanged, except…": the Bound step lost "(BLAS sums, NumPy pairwise reductions)" — the two canonical irreproducible orders; Fixtures lost "(strings, categories, a narrower domain)"; Done lost why `--no-sync`; the milestone step lost its enumerated triggers for a pointer; and report-format §7 silently dropped "Then the questions for the owner…", the very sentence #410 later had to diagnose. The enforcement detour was worse: #401 (`e2720646`, 04:52) built a 137-line prose checker wired into pre-commit; #405 (`c02990cc`, 05:01 — nine minutes later) deleted it entire because "the owner does not want an automatic text check"; #408's message then still claimed the docs "join the prose checker's SCOPE", false at merge because nobody re-read the branch text against merged master — the same stale-head failure as #367's timings (finding 7).

**Root cause.** One file serves two audiences, and style rewrites were treated as free of content risk, so nobody reviewed them for semantic loss; the checker was built as an "implementation choice" (README reserves decision records for contract/scope) although it was in fact a policy choice the owner then reversed.

**Change.** Split by audience: keep STE for reports, decision records and goal.md; exempt the operational sections (One cycle, Tickets and review, both briefs) explicitly in `AGENTS.md` rule 1 ("…needs only to be clear" — move the briefs and cycle into that list), and restore the dropped parentheticals to steps 3/5/6/7. Add one line to `workers.md` §5: *"A style-only PR lists its content changes in the body; 'content unchanged' is a claim the reviewer checks against the diff."*

**Cost/effect.** 1 h of edits; roughly −15–20% tokens on the files read every cycle, with examples and reasons permitted back in. The owner's reading experience is unchanged, because the owner never needed the cycle steps in STE. [INFERENCE on that last point: the tracker and reports are the owner surface named in `reporting.md`.]

### 6. Exemplars steer agents more than rules, and only rules are managed

#410's own causal analysis is the best evidence in this category: readings 1–3 argued owner decisions inside the report because (i) a superseded §7 once asked for a second question list, (ii) **"each reading copied the shape of the reading before it"**, and (iii) the only pre-merge check, the fresh-reader test, "checks the words, not what the report holds". The fix — the "Decisions stay in their records" section (report-format:122–140) — is well-aimed but local and manual, and the general lesson was not drawn. The format still instructs copying: "Use the same commands as the previous report" (§0), a confit goal reading as "a full example", and the worker brief points at four stale family modules as models (finding 1). Nothing obliges a rule change to sweep the exemplars that embody the old rule — #410 had to edit three *published* reports, breaking a rule to fix a rule.

**Change.** Three small additions. (a) In report-format's pre-merge step, require the report PR body to paste the output of two commands — `ls questions/README.md` and the regenerated coverage counts — next to the `needs_owner` block (a checklist the reviewer eyeballs, not a CI prose gate; the owner's #405 rejection of automatic text checks is respected). Note the directory currently doesn't exist (`ls questions/README.md` fails on this checkout), while four documents instruct writing records into it; commit an empty `open/.gitkeep` so the check is runnable. (b) Extend the fresh-reader prompt in the skill with placement questions: "List any question for the owner that appears outside 'Needs the owner'. List any number with no reproducing command in section 8." (c) Add to `workers.md` §5: "A PR that changes a loop rule greps the latest report, the briefs and PLANS for the old rule's wording and fixes or annotates each hit in the same PR."

**Cost/effect.** ~1 h plus one command per report. Would have caught the #410 duplication at reading 2 instead of after reading 3, and the stale brief of finding 1 at #458.

### 7. Measurement instructions are the thinnest where rework actually happened

All three blocking items in the supervisor's #367 review correspond to rules that exist for confit workers but were never written for native workers: timings/caps "predate confit #362… re-measured on master" (confit brief: "merge `origin/master`… then re-run the gate", numbers "on the merged head"); the probe must target the serving engine and decline on failure (finding 2); and PLANS numbering conflicts. The native brief's entire measurement instruction is "Record build and serve times for the widest configuration you test" (`worker-brief.md:49`) — no release build (confit brief line 23 requires reinstalling it before timing, because a debug `.so` silently rebuilds), no 64-row call or median convention (worker's PR used 1,024-row batches; reports use per-64), no merged-head rule, and "widest" undefined although width failures are multi-dimensional (#452: 10 features × 800 quantiles; T47: 2,000 centres × 2 features, 73 s). The same under-specification recurred as finding 92 — T46's estimates time the trial build, T47/T48/T51/T52's time the whole warm `to_native` — fixed in #508 by docstrings in code, not in any instruction a worker reads before writing the next estimate.

**Change.** Port the confit brief's four lines, adapted, into the native brief's Discipline:

> - Before timing, reinstall the release build (`uv sync --locked --group spark --reinstall-package confit`); a debug extension makes every number wrong.
> - Merge `origin/master` before you ask for review; then re-run the gate and re-take every timing and cap number on the merged head. A confit change on master can invert a cap (#367).
> - Report serve time as the median of 64-row calls against the twin on the same rows, and build time for the shape grid of the trap checklist.
> - State what each build estimate times (trial build alone, or whole warm `to_native`).

**Cost/effect.** 30 min, zero new tooling. Directly removes the #367 review round and the #508 docstring archaeology from future tickets; it will not stop confit-driven estimate refits (another category), only the self-inflicted stale-head kind.

## 2. What is working and should be kept

- **The three-layer prompt** (shared brief → loop brief → ticket section, "the prompt must stand alone", `workers.md` §4). The confit brief proves the format carries real operational content well; the native brief is under-filled, not mis-shaped.
- **Review discipline** (`workers.md` §5: blocking/optional, file + defect + what would satisfy, delivered on the PR *and* directly to the worker because the shared GitHub account hides reviews). #367's review is the standard working as written.
- **The decision-record route** (`AGENTS.md` rule 3, only 26 lines and the best-edited file in the set). It demonstrably fired: the worker-environment question became a record in #408 instead of report prose, and #404's five rulings closed the bound question cleanly.
- **Measure-then-write, on the named SHA, with repro commands** (report-format §0/§8 and Style). Findings with stable slugs across reports are good raw material — finding 2 above is about distilling them, not about the habit.
- **#410's self-diagnosis form** — naming the broken rule in the PR body. Keep it; finding 4 just asks that it also amend the rule.

## 3. Disagreements with `local://native-loop-report.md`

1. **#410 causation is overstated.** The report says readings duplicated decision content "because report-format instructed that content" (§1) / "#410 says report guidance caused it" (§2). Verified against history: the pre-#408 §7 instructed only a second *list of questions* — never proposals or recommendations, which readings carried by copying each other; and #408 (`c9245858`, 05:49) had already silently deleted that sentence 35 minutes before #410 (06:24), under a "content is unchanged" claim. The durable cause was exemplar-copying plus a word-level check, as #410's own body says.
2. **"Ticket names … traps" is a rule, not observed practice.** §2 states ticket content as fact. The live `tickets.md` contains no ticket sections at all — an empty table followed by ~1,400 words of merged-ticket journal prose (T53–T58), which also violates `workers.md` §3's board shape. Historical ticket text is unrecoverable from the board by design, so the claim is unverifiable from the sources the report cites.
3. **The appendix's "11 commits changed rule files" implies maintenance that didn't happen.** Of the substantive prompt files, `mandates/native/brief.md` changed never (only the #366 move and a path fix), `AGENTS.md#commands-repo-root` never after #401, and most listed commits are the move, the STE rewrite, link fixes, or bound-reporting bullets riding code PRs (#388, #501). The honest metric is: one substantive instruction about bounds reached README in 12 days; none reached the brief.
4. **§2's "Practice vs written rules (2)"** frames #410's report edits as loop deviation; the PR was owner-requested ("Requested by AmirHossain · project thread") and explicitly flagged the rule it broke. It is better evidence *for* the follow-the-reason clause than against rule compliance.
5. The report's #408 entry ("rewrote the native docs in simple English and opened a worker-environment record") misses both the unlisted content deletions and the stale "join the prose checker's SCOPE" claim after #405 deleted the checker — the two facts that make #408 instructive for this category.

## 4. Out-of-category, one line each

- Estimate refits after confit cost changes (15/53 ticket commits) are an interface problem, not a prompt problem — no wording fixes confit moving the cost model.
- `decisions/README.md` summarises the 2026‑10‑15 ruling as "may sit in any position" while `catalog-contract.md` contracts only last-step/unweighted-part — ruling vs built scope should be labelled as such in one of them.
- Reading 8 at 18,378 words suggests report-format §4's "list every bound per configuration" no longer scales; `coverage --bounds` already generates it.
- Stale open PR #432 (per the appendix) is board hygiene, untouched here.