# How close an additive chi2 sampler must be

**Question.** The parity ruling (confit's
`docs/decisions/closed/native-transform-parity-bounds.md`) holds an entry
whose order cannot be the twin's "within a small per-family ulp bound",
measured. `AdditiveChi2Sampler` has no small bound in ulps of the result.
What bound does it declare? This is the question of
[matvec-parity-bound.md](matvec-parity-bound.md) and
[power-parity-bound.md](power-parity-bound.md), for a third family.

**The twin.** sklearn 1.9 `AdditiveChi2Sampler._transform_dense` (a dense
row). For each feature `x`, with `s` the sample interval and
`j = 1 .. sample_steps - 1`:

- lane 0 is `sqrt(x * s)`;
- `factor = sqrt((2 * x * s) / cosh(pi * j * s))`;
- the two lanes for `j` are `factor * cos(j * (s * log(x)))` and the same
  with `sin`;
- every lane of a feature at 0 is 0.

**Why it is not small.** Every operation but one is reproducible:
- `sqrt`, `*` and `/` are correctly rounded on both sides.
- `cosh(pi * j * s)` is a fitted constant.
- numpy's `sin` and `cos` equal DuckDB's on this platform (the catalog's
  kernel probe, `function.kernel_distance`).

The one that is not is numpy's `log`. It is its own SIMD kernel on x86-64
with AVX-512, up to 1 ulp from glibc's, which is DuckDB's `ln` and
confit's. An ulp apart in `log(x)` moves the argument `j * s * log(x)` by
about `|j * s * log(x)| * eps`, and `cos` and `sin` pass that through
unchanged in absolute terms. Near a zero of `cos` or `sin` the lane itself
is tiny, so the same absolute difference is any number of ulps of the
result.

Measured on x86-64 with AVX-512, numpy 2.5.1, glibc 2.39, 2026-10-05.
The entry side is glibc's `log` and `cos`/`sin` through Python's `math`,
which are DuckDB's and confit's.

| draws of `x` | max, ulps of the result | max K, term scale |
|---|---|---|
| 200,000 uniform in (0, 1e3), each default `s` and `j` | 339 | 0.91 |
| 200,000 log-uniform in (1e-300, 1e300) | 2 | 0.02 |
| 200,000 near 1 (`1 + N(0, 1e-3)`) | 4 | 0.01 |
| 400,000 (native T11, #388; `s = 0.5`, `j = 1, 2`) | 8,192 | — |

The default `s` and `j` are `s = 0.8` with `j = 1`, `s = 0.5` with
`j = 1, 2`, and `s = 0.4` with `j = 1, 2, 3`, as sklearn picks `s` for
`sample_steps` 1 to 3.

K is `|native - twin| / (eps * factor * (1 + |j * s * log(x)|))`. The
ulps of the result grow with the number of draws, because more draws
land closer to a zero. K does not: it stays below 1.

**Options.**

1. *A term-scale bound,* as the matvec and power records propose. Per
   lane, `|native - twin| <= K * eps * S`, with
   `S = factor * (1 + |j * s * log(x)|)` and K = 1, which the measurement
   above holds with room. `check` gains the per-lane scale.
2. *A ulp bound measured over N seeds:* not small (above), and not a
   bound: the next seed can exceed it.
3. *It stays Python* until numpy's `log` can be spelled to the bit. It
   cannot be portably: numpy picks its kernel by CPU.

**Provisional choice.** Option 3: `AdditiveChi2Sampler` has no entry and
stays "not yet". Option 1 is the loop's proposal, with the matvec and
power families'.

**What would close it.** The owner's ruling on matvec-parity-bound.md, if
it covers a per-lane condition other than a dot product's term scale, or a
ruling here.

**Methodology (2026-10-06).** The research notes are in
[research/2026-10-06/](../research/2026-10-06/README.md). This decision
record uses four of them:

- chi2.md is the note on this family, the additive chi2 (chi-squared)
  sampler. A second agent re-ran the note and tried to refute its key
  claims. This agent is the adversarial verifier.
- framework.md is the note on the general form of a parity bound. This
  record uses its §2.3, the section on `AdditiveChi2Sampler`.
- critique.md is the note that settles the contradictions between the notes
  and their verifications.
- inference.md is the note that tests if the catalog entry changes a
  prediction. An adversarial verifier checked it too.

The research used this setup and these terms:

- **Environment.** The host is x86-64 with AVX-512 (Advanced Vector
  Extensions, 512-bit), a set of CPU instructions on 512-bit vectors. The
  software is numpy 2.5.1, scikit-learn (sklearn) 1.9.0, glibc 2.39 and
  DuckDB 1.5.5. Here glibc is the GNU C library. The code is master at
  commit 113fba7.
- **Kernel.** In this record, a kernel is the code that computes one math
  function, for example numpy's `log`. numpy selects its kernels by CPU.
  This selection is its dispatch.
- **The transformer.** `AdditiveChi2Sampler` is the sklearn transformer of
  this family. s is its sample interval. Its parameter `sample_steps` sets
  the values of j. For each input value x, the transformer gives one sqrt
  field, and one cos field and one sin field for each j.
- **Field.** A field is one output value of the transformer. Some numbers
  below are counts of fields. In them, each field of each row counts once.
- **Error scale and K.** For one field, the error scale is S = f·(1 + |a|).
  Here f is the `factor` of the field, sqrt(2·x·s / cosh(π·j·s)). The
  product a is j·s·ln x, where ln x is the natural logarithm of x. K is
  `|native − twin| / (eps·S)`, with eps = 2^-52. In this formula, native is
  the result of the entry, and twin is the result of the twin. So K is the
  distance between the two results in units of eps·S.
- **Labels.** "Sourced" means that the note cites the code of a package.
  "Derived" means that the note gives a derivation.

1. **What the twin computes** (sourced from `kernel_approximation.py`, the
   sklearn module that holds `AdditiveChi2Sampler`).
   - j runs over `range(1, sample_steps)`, that is, from 1 to
     sample_steps − 1. So the list of the default s and j earlier in this
     record is off by one. The correct list is this:
     - `sample_steps=1` has only the sqrt field, `sqrt(x * s)`. That field
       is bit-exact today.
     - `sample_steps=2` is s = 0.5, j = 1.
     - `sample_steps=3` is s = 0.4, j = 1, 2.
   - A negative input makes the twin raise an error. An input that is not a
     number (NaN) or that is infinite also makes it raise an error.
   - An input of −0.0 gives +0.0 in every field.
2. **Which kernels the twin runs** (sourced from numpy v2.5.1, and
   identified on the host).
   - **`log`.** On Linux x86-64 with AVX-512, numpy's float64 `log` is
     `__svml_log8_ha`. This kernel is the high-accuracy `log` of Intel's
     Short Vector Math Library (SVML). On other hosts, numpy's `log` is
     libm, the C math library of the platform.
   - On AVX-512, a numpy build without SVML runs numpy's own `log` kernel.
     Examples are numpy on macOS x86 and a build with the option
     `-Ddisable-svml`. Nobody measured that kernel.
   - **`sin` and `cos`.** numpy's `sin` and `cos` are libm's. numpy and
     libm gave different results on 0 of 3M arguments.
   - **`cosh`.** numpy's `cosh` (`np.cosh`) is SVML's too. So the "fitted
     constant" `cosh(π·j·s)` differs from glibc's for 2 of the 3 default
     pairs (s, j).
3. **Kernel accuracy** (measured against MPFR). MPFR is the GNU Multiple
   Precision Floating-Point Reliable library. It rounds correctly.
   - numpy's `log` is within 0.62 ulp of the correctly rounded result.
     glibc's `log` is within 0.52 ulp.
   - The two logs are at most 1 ulp apart. They differ on 1.4e-4 of draws
     over (0, 1e3), and on about 2% of draws near 1.
4. **The log-uniform row in this record's table is biased.**
   - An exp(uniform) draw is x = exp(t), with t uniform. For |t| ≥ 2, ln x
     lies within ulp/4 (a quarter of an ulp) of t. So the two kernels never
     disagree there (0 of 15M draws).
   - The significand of a number is that number scaled by a power of two
     into the range [1, 2). With random significands, the two kernels
     disagree at rates from 2.5e-4 down to 7e-7.
   - The kernel probe of the repository (`kernel_distance`, in
     `native/function.py`) uses exp(uniform) draws too. The probe measures, on
     the host, how far numpy's kernel for a function is from the entry's SQL
     for that function.
   - The measurements behind `_BOUNDS` use exp(uniform) draws too.
     `_BOUNDS` holds the ulp bounds of the family of function transformers.
5. **The bound** (derived, and re-derived by the verifier).
   - K ≤ max(d·A(s, j), κ). Here L is ln x, and t is the rounded product
     s·L. The terms are these:
     - d = 1 is the distance between the two logs, in ulps.
     - A(s, j), or A, is how far the roundings of s·L and j·t amplify a
       1-ulp difference in log. Let σ(s) be the significand of s. If s is
       a power of two, A = 1. Otherwise, A = 2/σ(s). At s = 0.4, A = 1.25.
       For any j, A < 3.
     - κ ≈ 2.1 is the term for the rounding of the result itself. If the
       error of glibc's cos and sin is at most 1 ulp, κ ≤ 3.
   - A model moves the log by 1 ulp. Over 2M draws, this model reproduces
     A exactly: 1, 1.25, 1.667 and 1.923.
   - If the `cosh` constant comes from a dispatch other than the twin's, it
     adds at most 1 to the bound on K.
6. **Measured K** (with the real kernels, in three independent test
   harnesses).

   | configuration | max K |
   |---|---|
   | `sample_steps=3` (default s = 0.4) | 1.14–1.24 |
   | `sample_steps=2` (default s = 0.5) | ≤ 0.98 |
   | s = 0.2501, j = 3 (legal, not default) | 2.57 |
   | j = 17 (legal, not default) | 2.80 |

   Near zeros of cos and sin, the distance in ulps of the result grows
   without limit. The largest distance measured is 8.8e11 ulps.
7. **Without the AVX-512 kernels.** If a process turns off numpy's feature
   group `X86_V4` (its AVX-512 kernels), the twin is bit-exact with the
   entry. The two differed on 0 of 23.4M fields. An independent run found 0
   of 7.2M.
8. **Spelling SVML's `log` in SQL** is not feasible now. A spelling here is
   SQL that gives the same bits as the kernel. The spelling needs these
   three things:
   - 11 operations of the fused multiply-add (FMA) class. An FMA computes
     a·b + c with one rounding.
   - a 17-step table from `vrcp14`. This is the AVX-512 instruction for an
     approximate reciprocal.
   - a value bound once, which confit does not have. Such a binding names a
     value that the SQL computes one time and reads many times.
9. **Downstream** (inference.md, verified). This item is about the
   predictions of models that use the fields as input.
   - On 4 datasets, 0.02–0.08% of the fields differ. No prediction changed.
   - On integer pixel values, the entry is bit-exact.
   - HistGradientBoosting is sklearn's histogram-based gradient-boosting
     model. The matvec record describes a mechanism in it: a repeated
     training value lies on a threshold. That mechanism applies here too,
     but on fewer values.
   - With `sample_steps=3`, the entry equals the twin on numpy without
     AVX-512 only if the `cosh` constant is glibc's. Otherwise, 177,541 of
     600,000 fields differ.

**Recommendation.** Take Option 1, with the error scale
S = f·(1 + |j·s·ln x|).

- **K = 4, derived.** The derivation gives A < 3 and κ ≤ 3. A `cosh`
  constant that another host computed adds 1. If the constant is the twin's
  own (`np.cosh` in the same process), K = 3.
- **Where the kernel probes read 0, declare K = 0.** This needs the probes
  for `log`, `sin`, `cos` and `cosh` to read 0, as on any host without
  AVX-512. There the entry is bit-exact. Draw the probe inputs with random
  significands, not as exp(uniform) draws.
- **Ship `sample_steps=1` now.** It has only the sqrt field, and that field
  is bit-exact.
- **Fix this record.** Correct its default (s, j) list and its exp(uniform)
  row.

These are the reasons, judged against the goal that inference (the
predictions of a model) does not change:

- **K = 1 fails at sklearn's own default.** At `sample_steps=3`, three
  harnesses measured K = 1.14–1.24. The derivation explains why:
  2/σ(0.4) = 1.25. A K taken from measurements would break again on the
  next legal configuration, which already reaches 2.8.
- **The twin's bits depend on the host.** Without numpy's AVX-512 kernels,
  the twin is bit-exact with the entry. With them, it is not. The bound
  promises parity with any legitimate twin, not with the bits of one host.
- **No prediction changed** downstream in these cases. A repeated training
  value on a HistGradientBoosting threshold can still flip a predicted
  label, as the matvec record measured. That is why the owner's ruling
  below makes bit-exact parity the default. Where the probes read 0, this
  family reaches that default.

**Ruling (owner, 2026-10-06).** The owner approved the recommendation:
option 1, with K = 4, or K = 3 when the `cosh` constant is the twin's own.
The amendment in [matvec-parity-bound.md](matvec-parity-bound.md) applies.
So with K above 0, the entry serves only when the caller asks for it. Where
the probes read 0, the entry is bit-exact and serves by default. The same
holds for `sample_steps=1`.
