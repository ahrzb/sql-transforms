# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

T29 merged on 2026-10-14: a quarter of the kernel probe's draws
(`function.kernel_distance`) are now doubles of random sign, binary
exponent and significand (`function._random_doubles`), in place of
`exp(uniform)`, on which two accurate `log` kernels always agree (#404,
`chi2.md` section 2). On this machine the probe reads what it read
before: 1 ulp for `exp`, `log`, `log2`, `log10` and `tan`, 3 for `cbrt`,
and 0 for `sin` and `cos`. The bounds hold over 2,500,000 draws against
confit (500,000 each of five kinds, master b4b8d03): `log10` reaches 2,
`cbrt` 3 and the others 1. Coverage is 31 of 68. The loop runs from the
"Transforms Loop" project thread.

Next: the native catalog's fifth reading, then PLANS "Next" (the paths
spelling first), the densified sparse outputs and the families on the
parity bound.
`decisions/open/bounded-steps-in-compositions.md` waits on the owner;
`compose.py` keeps refusing bounded steps meanwhile.
