# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

T26 merged on 2026-10-06: the forest and spline build estimates count
the input guard. confit builds the guard's tests again at each field read
of a struct output, so a build grows with the fields read times the
guard's tests; `trees._case_seconds`, `trees._paths_seconds` and
`spline._build_estimate` have a term for it, refitted on master 6aea15e.
Coverage is 31 of 68. The loop runs from the "Transforms Loop" project
thread.

Next up: the encoders' input guard in one test a feature (PLANS "Next"
item 1: wide encoders build slowly or are refused since T25), then the
densified sparse outputs and the families on the parity bound.
`decisions/open/bounded-steps-in-compositions.md` waits on the owner;
`compose.py` keeps refusing bounded steps meanwhile.
