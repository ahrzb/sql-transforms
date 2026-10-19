# Is the build budget right?

**Asked by:** native catalog, 2026-10-17. **Holds:** changing `MAX_BUILD_S` or what it times.
**Against:** `40f6ab17`.

## Context

Is 7 s the right budget for one `to_native` call, and what work may that call perform?
`to_native` translates a fitted step, probes its behavior and trial-builds the function.
The translation creates SQL expressions. Probes call the twin to find input checks and compare math kernels.

**Interim:** Keep `MAX_BUILD_S = 7.0` as the single budget for the whole warm `to_native` of one step.
Measure with a release build of confit. Warm means that first-use initialization has already run.
An estimate above the budget gives `NotNative` before the trial build.
This is an estimate, not a deadline that stops running work.

## Read first

- [Native catalog brief](../../mandates/native/brief.md)
- [The native loop review](../../research/native-loop-review/README.md)

## Options

### Option 1

Keep 7 s for the whole warm call, with estimates until confit provides a deadline or preflight.

### Option 2

Choose another budget for that same quantity. State the target host or calibration rule with it.

### Option 3

Give probes and trial builds separate budgets. Decide whether a cold first-use call has another limit.

## Recommendation

Option 1, until confit provides a build deadline or preflight that refuses by name; then revisit. It would change if a common deployment needs builds under or over 7 s.

## Evidence

PR #452 introduced estimated build caps for quantile and isotonic families.
[Reading 8](../../mandates/native/reports/2026-10-17-reading-8.md), finding 92, records two different meanings for the same cap.
Some estimates priced only the trial build; others priced the whole warm call.
PR #508 documents that split. The common meaning now includes probes and translation.
At master commit `40f6ab17`, the historical native PLANS records a distance-to-centres function that refused after 475 s.
It also records a SkewedChi2Sampler function that exhausted memory while building repeated logarithms.
The [build-deadline need](../../mandates/confit/tracker.md#needs) asks confit to stop before that cost and name the limit it passes.
Current family formulas estimate seconds from fitted dimensions; they do not stop a running build.

What would close it: The maintainer chooses the budget, timed quantity and allowed work before refusal.

## Answer

