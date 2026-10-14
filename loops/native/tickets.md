# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

T28 merged on 2026-10-14: the forest and spline build estimates are
fitted again on confit #430, which stopped building the input guard's
tests again at each field read of a struct output. A forest of 8
features takes one CASE per tree up to about 5,990 output fields (2,450
before): of the 43 forests measured, the estimates of T26 refuse 16 and
those of T28 refuse 3. Of the 359 spline configurations measured, 7 are
refused (43 before #430). Every encoder measured builds, in 0.03 to
5.2 s: `OneHotEncoder` over 8 string features of 125 categories and over
16 of 125, refused before, builds in 0.46 s and 1.2 s. Coverage is 31 of
68. The loop runs from the "Transforms Loop" project thread.

Next: the kernel probe's random significands (PLANS "Next" item 1),
then the other "Next" items, the densified sparse outputs and the
families on the parity bound.
`decisions/open/bounded-steps-in-compositions.md` waits on the owner;
`compose.py` keeps refusing bounded steps meanwhile.
