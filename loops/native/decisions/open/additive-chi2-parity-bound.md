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

**Methodology (2026-10-06).** The notes are in
[research/2026-10-06/](../research/2026-10-06/README.md): chi2.md, re-run by
an adversarial verifier, plus framework.md §2.3, critique.md and inference.md
(inference.md's verification is pending). The environment is x86-64 with
AVX-512; numpy 2.5.1, scikit-learn 1.9.0, glibc 2.39, DuckDB 1.5.5;
master 113fba7. The scale is S = f·(1 + |a|), with f the lane's `factor` and
a = j·s·ln x. K is `|native − twin| / (eps·S)`.

1. **What the twin computes** (sourced from sklearn's
   `kernel_approximation.py`).
   - j runs over `range(1, sample_steps)`, so the record's default list is off
     by one:
     - `sample_steps=1` has only the sqrt lane, which is bit-exact today;
     - `sample_steps=2` is s = 0.5, j = 1;
     - `sample_steps=3` is s = 0.4, j = 1, 2.
   - Negative, NaN and infinite inputs raise. −0.0 maps to +0.0 on every lane.
2. **Which kernels the twin runs** (sourced from numpy v2.5.1, and
   fingerprinted on the host).
   - `log`: on Linux x86-64 with AVX-512, numpy's float64 `log` is Intel
     SVML's `__svml_log8_ha`. Elsewhere it is libm. An AVX-512 build without
     SVML (macOS x86, `-Ddisable-svml`) runs numpy's own `log` kernel, which
     nobody measured.
   - `sin` and `cos` are libm's: 0 of 3M arguments differ.
   - `np.cosh` is SVML too. So the "fitted constant" `cosh(π·j·s)` differs
     from glibc's on 2 of the 3 default (s, j).
3. **Kernel accuracy** (measured against MPFR).
   - numpy's `log` is within 0.62 ulp of the correctly rounded result, and
     glibc's within 0.52.
   - The two are at most 1 ulp apart. They differ on 1.4e-4 of draws over
     (0, 1e3), and on about 2% of draws near 1.
4. **The record's log-uniform row is biased.**
   - For x = exp(t) with |t| ≥ 2, ln x lies within ulp/4 of t, so the kernels
     never disagree there: 0 of 15M draws.
   - With random mantissas they disagree at rates from 2.5e-4 down to 7e-7.
   - The repo's `kernel_distance` probe (`function.py:214`) and the `_BOUNDS`
     measurements use the same construction.
5. **The bound** (derived; re-derived by the verifier).
   - K ≤ max(d·A(s, j), κ), where:
     - d = 1 is the distance between the two logs;
     - A is how far the roundings of s·L and j·t amplify a 1-ulp difference
       in log. A = 1 when s is a power of two, 2/σ(s) otherwise (σ the
       significand of s; 1.25 at s = 0.4), and A < 3 for any j;
     - κ ≈ 2.1 is the result's own rounding. It is ≤ 3 if glibc's cos and
       sin are 1-ulp functions.
   - A 1-ulp model over 2M draws reproduces A exactly: 1, 1.25, 1.667, 1.923.
   - A `cosh` constant computed under the other dispatch adds at most 1.
6. **Measured K** (real kernels, three independent harnesses).

   | configuration | max K |
   |---|---|
   | `sample_steps=3` (default s = 0.4) | 1.14–1.24 |
   | `sample_steps=2` (default s = 0.5) | ≤ 0.98 |
   | s = 0.2501, j = 3 (legal, not default) | 2.57 |
   | j = 17 (legal, not default) | 2.80 |

   Ulps of the result grow without limit near zeros of cos and sin: up to
   8.8e11.
7. **Pinning.** With numpy's `X86_V4` disabled, the twin equals the entry bit
   for bit: 0 of 23.4M lanes, and 0 of 7.2M in an independent run.
8. **Spelling SVML's `log` in SQL** is not feasible now. It takes 11
   FMA-class operations, a 17-step `vrcp14` table, and a value bound once,
   which confit does not have.
9. **Downstream** (inference.md, verification pending). On 4 datasets,
   0.02–0.08% of entries differ and no prediction changed. On integer pixels
   the entry is bit-exact.

**Recommendation.** Option 1, with S = f·(1 + |j·s·ln x|).

- **K = 4, derived:** A < 3 and κ ≤ 3, plus 1 for a `cosh` constant computed
  on another host. Where the constant is the twin's own, `np.cosh` in the
  same process, K = 3.
- **Declare 0 where the probes read 0.** This needs the probes for `log`,
  `sin`, `cos` and `cosh` to read 0, as on any non-AVX-512 host. There the
  entry is bit-exact. Draw the probes with random mantissas, not exp(uniform).
- **Ship `sample_steps=1` now:** it is the sqrt lane only, and bit-exact.
- **Fix the record:** the default (s, j) list, and its exp(uniform) row.

Why, judged against the goal that inference does not change:

- **K = 1 fails at sklearn's own default.** At `sample_steps=3` it measured
  1.14–1.24 in three harnesses, and the derivation explains why: 2/σ(0.4) =
  1.25. A measured constant would break again on the next legal
  configuration, which already reaches 2.8.
- **The twin's bits depend on the host.** Pinned to baseline kernels it
  equals the entry exactly; on AVX-512 it does not. The bound promises
  parity with any legitimate twin, not with one host's bits.
- **No prediction changed** downstream.
