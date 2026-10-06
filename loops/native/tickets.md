# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T22 | One CASE per tree, and the trees cap fitted again | `claude/project-thread-2pg6vk` | T21 | `trees.py` | inline | | next |

T21 merged on 2026-10-06: the spline compares its arms, and the step
its instances' output fields, by tree instead of by SQL text, and the
spline's build cap is fitted again on confit's binding of a value once
(#412). Coverage is 31 of 68. The loop runs from the "Transforms Loop"
project thread.

T22 is next (PLANS "Next", item 1). Then the parity bound in
`native.check` and the families on it (PLANS "Ruled 2026-10-06, to
build"). `decisions/open/bounded-steps-in-compositions.md` waits on the
owner; `compose.py` keeps refusing bounded steps meanwhile.

## T22: one CASE per tree

**Why.** `trees.py` spells each output field as the conjunction of the
branch tests on its leaf's path. Since #412, one nested CASE per tree that
answers the leaf id, read by every output field of its tree, binds once.
On the default forest (100 trees of depth 5, 2,286 output fields) it
serves 118 µs a row against 202 µs for the paths, and builds in 1.28 s
against 1.84 s (the confit loop's measurement, release build of 4865d9f).
The trees cap (`MAX_PATH_STEPS`) assumes 0.28 ms a path step, where
reading 4 measured 0.25 ms at 1,173 steps and 0.44 ms at 21,353 (finding
56; PLANS "Next", item 1).

**Do.**
- Build each tree's nested CASE once, NaN routed as `_apply_dense` routes
  it, and spell output field k of the tree as `leaf = k`.
- Measure warm release builds and serves over forests (trees, depth,
  unbounded), and fit the cap again on the new spelling, in the unit the
  build follows.
- Update the module docstring's table.
- Run the 200-seed parity run of `RandomTreesEmbedding` and the gate.
