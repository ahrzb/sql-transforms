---
loop: native
date: 2026-10-15
master: fbbcd24
previous: mandates/native/reports/2026-10-15-reading-6.md
kind: short
trigger: cross-loop delivery
kpis:
  coverage_native: 35
  coverage_partial: null
  coverage_notnative: null
  coverage_not_yet: 33
  nonzero_ulp_bounds: 6
  families_merged_since_last: 0
  widest_build_s: null
  widest_serve_us_per_64: null
  twin_us_per_64: null
  open_needs_from_confit: 3
needs_owner:
  - questions/README.md
  - questions/README.md
needs_from_other_loop: [struct-field-reads, verifier-error-refusal, function-body-pass]
---

# Native catalog: short reading, 2026-10-15

This reading names master `fbbcd24` and follows
[reading 6](2026-10-15-reading-6.md). Its trigger is "cross-loop delivery"
([reporting policy](https://github.com/ahrzb/sql-transforms/blob/40f6ab17/loops/reporting.md)): two confit changes that the
native loop needed merged. A number such as #459 is a PR.

## What moved

**The confit loop served two needs of the native loop.** When no part of a
CASE can trap, and the CASE is small or a NULL test, confit #459 compiles
it without branches. On master bea1372, a `PCA` of 256 features to 64
components then serves 64 rows in 3.7 to 4.3 ms, against 77 to 83 ms
before and 29.7 to 34.0 ms for its twin. Confit #461 checks one time
whether a SQL function call holds a subquery, not again at each field read
of a struct output.

**The catalog entries now serve 49 of the measured configurations that
only the twin served before.** Each entry estimates the time that confit
takes to build it, and raises `NotNative` when the estimate is above 7 s.
In #463 and #466, the loop measured the builds again on confit #459 and
#461, and computed new estimates. No configuration goes back to the twin.
By family, with the configurations measured and the build times:

- `PCA` and `GaussianRandomProjection`: 34 of 144, 2.8 to 6.9 s.
- `RandomTreesEmbedding`: 7 of 29, 2.9 to 8.3 s. For trees without a
  maximum depth, a build takes 1.1 to 1.3 times the estimate, so two of
  the 7 build in more than 7 s.
- `IsotonicRegression`: 4 of 16, 4.1 to 6.5 s.
- `QuantileTransformer`: 2 of 74, 5.2 and 6.1 s.
- `SplineTransformer`: 2 of 172, 6.3 s each.

**Where its twin answers 0.0, a projection now answers 0.0 too (#462).**
At reading 6, the parity run found 403 output fields that answered -0.0
there. On master c21d0c4, the same run finds none.

## Needs the owner

- [May a step with a parity bound sit in a composition?](../../../questions/README.md)
  On master `fbbcd24`, `PCA(n_components=2)` over 5 features serves alone
  with `allow_bound=True`, the caller's request for a parity bound. In
  `make_pipeline(StandardScaler(), PCA(n_components=2))`, `compose.py`
  raises `NotNative` for the `PCA` step, so the `Pipeline` keeps its twin.
- [Will the owner add a setup script to the cloud environment?](../../../questions/README.md)
  The record describes sessions that offer only CPython 3.14.0rc2, where
  `sql_transform` does not import. In this reading's session, the Nix dev
  shell (#428) gave CPython 3.14.7, and `sql_transform` imported.

## Next

The loop stopped on 2026-10-15, at the owner's request. If the loop starts
again, its next work is item 1 of the work that the owner ruled on
2026-10-06 ([`PLANS.md`](https://github.com/ahrzb/sql-transforms/blob/40f6ab17/loops/native/PLANS.md)): the other linear projections, such
as `FastICA`. An untested draft of their first ticket, T41, is in the
project's shared files, in the folder `transforms-loop/t41-draft/`.
