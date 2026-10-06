# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T20 | ±inf in the fixture generator's edge values, and `check` fails when it compares no row | `claude/project-thread-2pg6vk` | T19 | `catalog_test.py`, `_check.py`, every class's draws | inline | | next |

T19 merged on 2026-10-06: a configuration within a bound above 0 serves
only with `to_native(step, allow_bound=True)`. Coverage is 31 of 68, and
reading 4 (#411) covers waves 4 to 6. The loop runs from the "Transforms
Loop" project thread.

T20 changes every class's draws, so it runs alone. Then the parity bound
in `native.check` and the families on it (PLANS "Ruled 2026-10-06, to
build"), beside the spline and the trees on confit's binding of a value
once (#412; PLANS "Next", items 2 and 3). `decisions/open/bounded-steps-in-compositions.md` waits on the
owner; `compose.py` keeps refusing bounded steps meanwhile.

## T20: ±inf in the fixture generator, and a `check` that compares rows

**Why.** Finding 52: the fixture generator's edge values (`EDGES` in
`catalog_test.py`) hold no ±inf, so the parity test never sends one. The
periodic spline breach that #406 fixed hid there. And `check` skips each
row where the twin raises (4.9% of the rows at reading 4), so a step whose
twin raises on every row compares nothing. The parity test asserts that it
compares a row, but other callers of `check` do not (PLANS "Next", item 1).

**Do.**
- Add `inf` and `-inf` to `EDGES`.
- Make `check` raise `ParityError` when it compares no row.
- Run the 200-seed parity run. A failure is a finding: fix the entry or
  decline the configuration by name, and say which in the PR.
- Every class's draws change, so measure nothing else in this PR.
