# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

Reading 5 merged on 2026-10-14
([`reports/2026-10-14-reading-5.md`](reports/2026-10-14-reading-5.md)), on
master 44052cf: coverage 31 of 68, no new class. On one machine, the
default forest serves a row 3.8 times as fast as reading 4's code, and the
input guard costs a median 8% of a 64-row call. Its new findings 68
(`QuantileTransformer`'s cap builds past 7 s there) and 69 (a `Pipeline`'s
guard costs 4.4 times its build) are PLANS "Next" items 1 and 2. T30
(#446) dropped the forests' paths spelling: one CASE per tree is their one
spelling, and a forest estimated past 7 s stays Python. T31 (#447) probes
an encoder's categories in about four calls of `transform` a feature, not
one a category: `to_native` of `OrdinalEncoder` over 32 string features of
125 categories takes 1.3 s, 40.8 s before. T32 (#448) names a value
of a step's output an output field, not a lane, in the catalog's code and
in `_udf.py`, as GLOSSARY.md now defines it. T33 (#450) finds a number
among a run of consecutive integer categories by a range in the encoders'
tests: `OrdinalEncoder` over 32 number features of 125 categories builds
in 0.74 s, 4.6 s before. The loop runs from the "Transforms Loop" project
thread.

Next: PLANS "Next" in order (`QuantileTransformer`'s cap as a build
estimate first), then the densified sparse outputs and the families on the
parity bound.
`decisions/open/bounded-steps-in-compositions.md` waits on the owner;
`compose.py` keeps refusing bounded steps meanwhile.
