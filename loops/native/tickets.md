# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T21 | The spline on confit's binding of a value once: structural keys, and `_build_estimate` fitted again or dropped | `claude/project-thread-2pg6vk` | T20 | `spline.py`, `_registry.py` | inline | | next |
| T22 | One CASE per tree, and the trees cap fitted again | `claude/project-thread-2pg6vk` | T21 | `trees.py` | inline | | next |

T20 merged on 2026-10-06: the fixture generator draws ±inf, and `check`
raises when it compares no row. Its 200-seed parity run passed 35,055
steps of 176 configurations, with 0 failures and 145 named skips.
Coverage is 31 of 68. The loop runs from the "Transforms Loop" project
thread.

T21 and T22 change separate files, inline, one at a time (PLANS "Next",
items 1 and 2). Then the parity bound in `native.check` and the families
on it (PLANS "Ruled 2026-10-06, to build"), which overlap T21 in
`_registry.py`. `decisions/open/bounded-steps-in-compositions.md` waits
on the owner; `compose.py` keeps refusing bounded steps meanwhile.

## T21: the spline on confit's binding of a value once

**Why.** Since confit #412, a value that a SQL function body reads twice
or more binds once, so the spline's expression no longer doubles per
degree when confit builds it. `_build_estimate` still models the old
build: it estimates 5.9 to 9.4 s for steps that build in 1.1 to 3.5 s, so
the 7 s cap refuses four of the five steps that the confit loop measured.
And the translation now costs about as much as the build (0.5 to 1.0 s in
`SqlFunction(...)` at degree 5 over 32 features): the `.sql()` keys in
`spline._arms` and in `_registry`'s `select` write each value out in
full, which doubles per degree (PLANS "Next", item 1).

**Do.**
- Compare arms and output fields by a structural key with a memo per
  node, linear in the distinct nodes, not by `.sql()`.
- Measure warm release builds of master over the grid of `spline.py`'s
  docstring (1 to 64 features, degrees 1 to 5, 5 to 8 knots, the five
  extrapolations). Fit `_build_estimate` again, or drop it if no step the
  entry takes on passes 7 s. Say which in the PR, with the fit's error.
- Update the module docstring's widths and build times.
- Run the 200-seed parity run of `SplineTransformer` and the gate.

## T22: one CASE per tree

**Why.** `trees.py` spells each output field as the conjunction of the
branch tests on its leaf's path. Since #412, one nested CASE per tree that
answers the leaf id, read by every output field of its tree, binds once.
On the default forest (100 trees of depth 5, 2,286 output fields) it
serves 118 µs a row against 202 µs for the paths, and builds in 1.28 s
against 1.84 s (the confit loop's measurement, release build of 4865d9f).
The trees cap (`MAX_PATH_STEPS`) assumes 0.28 ms a path step, where
reading 4 measured 0.25 ms at 1,173 steps and 0.44 ms at 21,353 (finding
56; PLANS "Next", item 2).

**Do.**
- Build each tree's nested CASE once, NaN routed as `_apply_dense` routes
  it, and spell output field k of the tree as `leaf = k`.
- Measure warm release builds and serves over forests (trees, depth,
  unbounded), and fit the cap again on the new spelling, in the unit the
  build follows.
- Update the module docstring's table.
- Run the 200-seed parity run of `RandomTreesEmbedding` and the gate.
