# May a step with a parity bound sit in a composition?

**Question.** `compose.py` serves a Pipeline, ColumnTransformer or
FeatureUnion of catalog entries. Today it refuses any step whose bound is
not 0. The parity-bound rulings of 2026-10-06 let more families serve with a
bound: matvec, power and additive chi2. Which compositions may hold such a
step?

**What it decides.** If compositions keep refusing these steps,
`make_pipeline(StandardScaler(), PCA())` stays `NotNative`. That is true
although each of its two steps is native alone. The same holds for
StandardScaler before KMeans, and for any bounded family inside a
ColumnTransformer.

**Options.**

1. *The bounded step is last.* It is the last step of a Pipeline, or one
   part of a ColumnTransformer or FeatureUnion. The steps before it are
   bit-exact, so its input is the input of its twin. `native.check` can then
   compute the error scale from that input, as for the step alone. This
   needs no new theory.
2. *The bounded step feeds continuous steps,* for example a scaler or
   another projection. The error scale must then pass through those steps.
   To first order, the new scale is Σ|∂y/∂v|·S_v. Nobody has derived this
   for the catalog yet.
3. *The bounded step feeds a discontinuous step,* for example
   KBinsDiscretizer, Binarizer, the knots of a spline, IsotonicRegression or
   a threshold. No parity bound exists here. A value that is one rounding
   from a threshold can take the other branch.

**Provisional choice.** `compose.py` keeps refusing every step with a bound
that is not 0.

**What would close it.** An owner ruling on options 1 to 3. The loop
proposes option 1 now, option 2 when a derivation exists, and never
option 3. The critique of the 2026-10-06 research found this question
(research/2026-10-06/critique.md, section 2).

**Ruling (owner, 2026-10-15).** A composition may hold a step with a
parity bound above 0 in any position. This ruling combines the
three options above. The rule of 2026-10-06 stays: a bound above 0 serves
only when the caller asks for it with `to_native(step, allow_bound=True)`.
`native.check` then does one of these two things for each output field:

- If a derivation gives the bound of the field, the check holds each row to
  that bound. This is true when the bounded step is the last step, or one
  part of a ColumnTransformer or a FeatureUnion (option 1). It is also true
  when only continuous steps with a derived error scale follow the bounded
  step (option 2). For an affine step y = A·v + b, the error scale passes
  through as |A|·S, and the step adds its own rounding.
- If no derivation gives a bound, the check reports the flip rate: the
  fraction of the test rows in which the field differs from the twin. This
  is the case when a discontinuous step follows the bounded step (option 3),
  for example a threshold, a bin edge, a tree or a nearest centre. The
  report gives the flip rate as a measurement, not as a parity guarantee.

These are the reasons of the owner. If `compose.py` refuses a bounded
step, it does not protect the users who ask for a bound. It only keeps
`native.check` from a bound that nobody derived. And the twin itself is not
stable to the last bit. It moves by
about 9.9e9 ulps between a call with one row and a call with a batch (the
amendment of 2026-10-06).
