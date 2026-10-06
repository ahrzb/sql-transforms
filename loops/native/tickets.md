# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

T22 merged on 2026-10-06: `trees.py` spells a forest as one nested CASE
per tree, which every output field of the tree reads, where its build is
estimated within 7 s, else as the paths where theirs is, else refuses it.
The default forest (100 trees of depth 5) serves a row in 253 µs against
528 µs for the paths. One unbounded tree over 2,000 rows is now served;
200 trees of depth 5 are now refused (7.9 s to build on #417's confit).
Coverage is 31 of 68. The loop runs from the "Transforms Loop" project
thread.

Next up: the parity bound in `native.check` and the families on it
(PLANS "Ruled 2026-10-06, to build").
`decisions/open/bounded-steps-in-compositions.md` waits on the owner;
`compose.py` keeps refusing bounded steps meanwhile.
