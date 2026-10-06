# Native ticket board

What is in flight in the native loop, inline work included, in the shared
form of [`../workers.md`](../workers.md) §3. A merged ticket leaves the
board; what it learned is in [`PLANS.md`](PLANS.md), and its results are in
the reports ([`reports/`](reports/)). A worker's prompt is
[`../worker-brief.md`](../worker-brief.md), then
[`worker-brief.md`](worker-brief.md), then its ticket's section below.

| id | ticket | branch | depends on | overlaps | worker | PR | state |
|---|---|---|---|---|---|---|---|
| T19 | `to_native(step, allow_bound=False)`: a bound above 0 serves only on request | `claude/project-thread-2pg6vk` | reading 4 | `_registry.py`, `__init__.py`, `function.py`, `power.py`, coverage | inline | | next |

T16 (#403), T17 (#402) and T18 (#406) merged on 2026-10-06, and coverage
is 31 of 68. The loop runs from the "Transforms Loop" project thread.
Reading 4 covers waves 4 to 6 and merges before T19, because its
measurements predate T19.

After T19 comes ±inf in the row generator (PLANS "Next", item 1). It
changes every class's draws, so it runs alone. Then the parity bound in
`native.check` and the families on it (PLANS "Ruled 2026-10-06, to
build"). `decisions/open/bounded-steps-in-compositions.md` waits on the
owner; `compose.py` keeps refusing bounded steps meanwhile.

## T19: `to_native(step, allow_bound=False)`

**Why.** The owner's amendment in
`decisions/closed/matvec-parity-bound.md`: bit-exact is the default, and a
configuration with a bound above 0 serves only when the caller asks. Today
`to_native` serves Box-Cox (4 ulps) and `FunctionTransformer`'s `exp`,
`log`, `log2`, `log10`, `tan` and `cbrt` by default.

**Do.**
- Add `allow_bound: bool = False` to `to_native`. Where the step's bound is
  above 0 and `allow_bound` is false, decline: return the step, or raise
  `NotNative` under `strict`, with a message that names the bound and the
  argument.
- A bounded function whose `kernel_distance` reads 0 on this platform is
  bit-exact here, so it serves by default.
- Say why in the docs and on the coverage page: the HistGradientBoosting
  labels that flipped under the Box-Cox entry (the amendment's evidence).
- Test the refusal by default, the serve with `allow_bound=True`, and the
  0-distance case. Keep the parity tests on the bounded configurations
  (they pass `allow_bound=True`).
