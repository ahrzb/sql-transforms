# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T22 | One CASE per tree, and the trees cap fitted again | `claude/project-thread-2pg6vk` | T23 | `trees.py` | inline | | next |

T23 merged on 2026-10-06: past the first output field, a field that
every instance answers with the same tree is that tree alone, without the
CASE on the instance id (every field, in a step of one instance). Steps of
one instance build 1.1-1.4x and serve 1.1-1.3x as fast. Coverage is 31 of
68. The loop runs from the "Transforms Loop" project thread.

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
56; PLANS "Next", item 1). Measured here (one instance, a list return,
10,000-row batches, release build before #417): one CASE per tree served
1.6-1.9x as fast as the paths, but its build grew faster than the trees.
At 30, 100 and 250 trees of depth 5 it built in 0.55, 3.8 and 21.6 s
against 0.43, 1.75 and 6.2 s for the paths; without the CASE on the id
(T23), in 0.32, 1.95 and 10.5 s against 0.27, 1.22 and 4.0 s.

**Do.**
- Build each tree's nested CASE once, NaN routed as `_apply_dense` routes
  it, and spell output field k of the tree as `leaf = k`.
- Measure warm release builds and serves over forests (trees, depth,
  unbounded) on master's confit (#417, a whole struct as one output value,
  landed since), and fit the cap again on the new spelling, in the unit
  the build follows.
  If its build still grows faster than the trees, spell a forest past its
  cap as paths, under their own cap.
- Update the module docstring's table.
- Run the 200-seed parity run of `RandomTreesEmbedding` and the gate.
