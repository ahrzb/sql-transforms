# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|

T27 merged on 2026-10-06: the encoders' input guard is one test a
feature, and under `handle_unknown="error"` an output field's ELSE
answers its largest group, so a one-hot output field is one comparison.
Past 100 strings an ordinal encoder's guard is one substring search.
`OneHotEncoder` over 8 string features of 50 categories builds again
(5.3 s; refused since T25), over 8 of 10 it builds in 0.21 s against
0.46 s and serves a row in 18 us against 30, and `OrdinalEncoder` over 32
features of 125 categories builds in 1.2 s against 26 s. Coverage is 31
of 68. The loop runs from the "Transforms Loop" project thread.

The loop is paused at the owner's request (2026-10-06): it takes no new
ticket until the owner resumes it. Next when it resumes: the kernel
probe's random significands (PLANS "Next" item 1), then the densified
sparse outputs and the families on the parity bound.
`decisions/open/bounded-steps-in-compositions.md` waits on the owner;
`compose.py` keeps refusing bounded steps meanwhile.
