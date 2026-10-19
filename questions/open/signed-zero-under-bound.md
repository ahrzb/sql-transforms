# Should a nonzero parity bound distinguish zero signs?

**Asked by:** native catalog, 2026-10-17. **Holds:** making `check` fail on a zero-sign difference under a parity bound above 0.
**Against:** `40f6ab17`.

## Context

When a parity bound is above 0, should `check` fail if only the sign of zero differs?
`check` is the catalog comparison in `native/_check.py`.

**Interim:** Keep the current comparison. A nonzero bound treats −0.0 and +0.0 as one value.
Bit-exact comparisons still distinguish their signs. Keep existing signed-zero regressions.

## Read first

- [Native catalog brief](../../mandates/native/brief.md)
- [The native loop review](../../research/native-loop-review/README.md)

## Options

### Option 1

Keep numeric equality under nonzero bounds, with separate zero-sign regressions.

### Option 2

Fail only where every product is exactly zero. Define whether underflowed products qualify before implementation.

### Option 3

Fail every zero-sign difference, even under a nonzero bound. Compare with the row-local twin, not a batch call.

## Recommendation

None yet: the reviews disagree. Two prefer option 2; one prefers option 1 with the differences tracked separately ([the native loop review](../../research/native-loop-review/README.md#where-the-reviewers-disagree)).

## Evidence

Commit `40f6ab17` is the master revision inspected for this question.
The comparison function `near` in `native/_check.py` accepts equal mapped values.
Python equality makes −0.0 equal +0.0.
[Reading 6](../../mandates/native/reports/2026-10-15-reading-6.md), finding 77, records a sum whose products all round to zero.
The entry returns −0.0 where its twin returns +0.0.
PR #462 adds +0.0 to these sums in linear projections, which multiply a row by a fitted matrix.
PR #497 fixes a one-feature case: its twin returns the signed product without adding +0.0.
These differences passed the nonzero bound; they were not bounded-parity failures.
The same fitted matrix can return different zero signs for a one-row call and a call with several rows.

What would close it: The maintainer chooses the zero-sign rule and its scope.

## Answer

