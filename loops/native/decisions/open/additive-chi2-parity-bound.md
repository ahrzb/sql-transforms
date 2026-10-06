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
