# What evidence may set an ulp bound?

**Asked by:** native catalog, 2026-10-17. **Holds:** changing how a family chooses an ulp bound.
**Against:** `40f6ab17`.

## Context

May a family register the largest ulp distance measured over seeds 0–199, or must a derivation or probe justify it?
An ulp is one unit in the last place of a floating-point result.

**Interim:** Keep the current ulp-selection policy in README step 6 and the closed rulings.
Keep error-scale K derived, not fitted to measurements. Do not widen an existing bound without a closed record.

## Read first

- [Native catalog brief](../../mandates/native/brief.md)
- [The native loop review](../../research/native-loop-review/README.md)

## Options

### Option 1

Keep measured ulp maxima with the current kernel probe and recorded evidence.

### Option 2

Require a derivation for each registered ulp bound. Refuse configurations without one.

### Option 3

Accept an accuracy contract for named math code, checked by a confit probe.
   Define what sampled inputs can establish and when the probe must refuse.

## Recommendation

Option 3 for libm-style kernels behind a probe, and option 2 elsewhere. It would change if no accuracy contract can be stated for the named math code.

## Evidence

Commit `40f6ab17` is the master revision inspected for this question.
README step 6 asks for the largest ulp distance over at least 200 seeds.
The function translator (`native/function.py`) keeps its per-function limits in the map `_BOUNDS`.
That map registers 3 ulps for numpy's cube root, `cbrt`.
Its comment cites catalog seeds 0–199 and a larger kernel experiment at commit `b4b8d03`.
PR #394 serves cube root after confit #390 changes the math code that computes it.
The [matvec ruling](../../proposals/closed/matvec-parity-bound.md) requires a derivation for an error-scale K, not a measured maximum.
It does not change the existing ulp families.
The kernel probe compares numpy's math code with confit's on sampled inputs.

What would close it: The maintainer states what may justify a new ulp bound and whether existing families must change.
This question is separate from how error-scale K is derived.

## Answer

