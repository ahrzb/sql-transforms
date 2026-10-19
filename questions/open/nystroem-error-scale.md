# Does the maintainer ratify Nystroem's changed error scale?

**Asked by:** native catalog, 2026-10-17. **Holds:** any further change to Nystroem's error scale, and counting its bound as covered by a closed proposal.
**Against:** `40f6ab17`.

## Context

Does the maintainer ratify or reject the error scale S implemented for Nystroem in PR #509?
Nystroem makes features from similarities between a request and fitted sample rows.
A kernel computes each similarity. A fitted matrix then combines those values into output fields.

**Interim:** Leave the merged implementation unchanged while this question waits.
It serves only with `allow_bound=True`. Make no further widening without a closed record.
This provisional choice is not the maintainer's approval of the changed scale.

## Read first

- [Native catalog brief](../../mandates/native/brief.md)
- [The native loop review](../../research/native-loop-review/README.md)

## Options

### Option 1

Ratify the changed S after reviewing its derivation and retained failing cases.

### Option 2

Reject it and keep Nystroem `NotNative` until an approved scale exists.

### Option 3

Require another derived scale, then decide its admitted configurations.

## Recommendation

Option 1, after an independent review of the derivation and of the retained failing cases; the change narrows nothing the old scale admitted. It would change if the review finds the new S unsound.

## Evidence

The Nystroem ticket, T54, merged in PR #509 as commit `1a14ce60`.
The [matvec ruling](../../proposals/closed/matvec-parity-bound.md) sets its K to `max(n + 5, m + 3)`.
Here n is the number of features and m is the number of fitted sample rows.
T54 keeps that K but changes S in `native/kernel.py`, the kernel-sampler family module.

For output field l, the implementation uses:

```text
S_l = sum_j |N_lj| * k_tilde_j * (1 + gamma * S2_j)
S2_j = ||x||² + ||c_j||² + 2 * sum_i |x_i * c_ji|
k_tilde_j = exp(-gamma * max(q_j - (n + 4)/2 * eps * S2_j, 0))
q_j = sum_i (x_i - c_ji)²
```

Here x is the request, c_j is fitted sample row j, and N is the fitted matrix.
The fitted factor gamma controls how quickly similarity falls with distance; eps is 2^-52.
The term subtracted from q_j allows for rounding of the squared distance.
The old research scale uses `exp(-gamma * q_j)` instead of `k_tilde_j`.
Far from zero, rounding can move the squared distance enough to change its exponential by a large factor.

At master commit `40f6ab17`, the translator's docstring reports 11.1 million fields past the old bound.
That experiment used 16 seeds, up to 64 features, and 8 or 64 fitted sample rows.
Its scripts were outside the repository at `/mnt/project-files/transforms-loop/t54/`.
Treat the count as historical evidence, not a reproduced measurement.
No closed decision record in this checkout ratifies that replacement.

What would close it: The maintainer ratifies or rejects the changed S.
The ruling must name the formula, assumptions and admitted configurations, not only K.

## Answer

