# Native catalog: brief

## Standing instructions

### Terms

- The twin is the fitted sklearn estimator itself; it is the reference. Never change it to make an entry pass.
- A family is the translators of one module (`scalers.py`, `impute.py`, `project.py` and so on).
- `to_native(step, allow_bound=True)` lets a caller accept a parity bound above 0.
- A kernel probe compares the twin's math kernel with confit's on sampled inputs.
- A warm call excludes first-use initialization.

### One task

A task is one family, or one named cross-cutting change (a refit, a machinery change). One task has one branch and one pull request.

1. **Sync.** Fetch `master` and branch from it as `claude/native-<name>`.
   List open native pull requests (`gh pr list --state open --search 'native in:title'`); give each a tracker task or close it with a reason.
2. **Pick.** Take the top open task in the [tracker](tracker.md).
   If none is open, take the easiest "not yet" row of [coverage.md](../../packages/sql-transform/spec/native/coverage.md), and write its task description first.
3. **Read the twin.** Find the class's `transform` in the installed sklearn (`.venv/lib/python3.*/site-packages/sklearn/`).
   Write down its exact sequence of operations, its dtype, what it does with NaN, and every constructor parameter, fitted state, shape and representation that changes `transform`.
   Go through **Traps** below; for each that applies, name the fixture or refusal that covers it.
4. **Translate.** Write the translator in the family module; a new family gets a new module, which `__init__.py` imports.
   Map the fitted attributes onto the features in the twin's order, using `confit.sql` and `_helpers.py`.
   Raise `NotNative` naming each configuration the entry does not serve.
   Register exact classes only, because a subclass may override `transform`.
5. **Fixtures.** Add the class to `FIXTURES` in `catalog_test.py`, one factory for each configuration that changes `transform`, as one block.
   If the family needs inputs the row generator does not make, extend `_fit_matrix` and `_rows`; never add a special case for one entry.
6. **Bound.** Register the entry bit-exact (`ulps=0`); bit-exact is the default.
   - A bound above 0 needs `allow_bound=True` and a closed proposal that gives it.
   - If the entry cannot follow the twin's order, measure the largest distance in ulps over at least 200 seeds; register it and cite seeds, maximum and date in the translator's docstring. First look for the twin's order and follow it.
   - For an error scale, register the K the closed proposal derives, in `translates(cls, scale=ErrorScale(...))`. Measure max |entry − twin| / (eps·S) over at least 200 seeds and cite it; the measurement is evidence that K holds, never the source of K.
   - The input guard traps exactly where the twin's validation raises, never where it answers.
   - A kernel probe runs through confit and declines with `NotNative` on any failure.
   - Regenerate the bounds manifest with the coverage command. A widening needs a closed proposal in the same change; CI runs `scripts/check_bounds.py <base-ref>` and flags a widening unless the change touches `proposals/closed/`.
7. **Build cost.** Declare `builds=(estimate, ...)` in `translates`, or `uncapped="<truthful reason>"`.
   - `MAX_BUILD_S` in `_registry.py` is the single budget: 7 s for the whole warm `to_native` of one step on a release build of confit, including translation, probes and the trial build. Import it; never copy its value.
   - Price the slower of a list output and a struct read field by field.
   - If the estimate exceeds `MAX_BUILD_S`, refuse with `NotNative` before the trial build.
   - Run `benchmarks/native_estimates.py --check FAMILY` when the family or its build machinery changes. Refit only when measured/estimate falls outside [0.5, 2], and take coefficients only from `--fit FAMILY`. `--check` is meaningful only on the reference Linux host; elsewhere it measures the host.
   - If confit alone became faster, add a refit line to the family's next task; it is not a task by itself.
8. **Done.**
   1. Regenerate coverage and bounds: `uv run --no-sync python -m sql_transform.native.coverage --write`.
   2. `git add -A`, because pre-commit skips untracked files.
   3. `uv run --no-sync pre-commit run --all-files`.
   4. The gate: `uv run --no-sync python scripts/gate.py`. It prints `gate: green`.
   5. Merge `origin/master`, re-take timings and re-run the gate on the merged head.
   6. Open the pull request with the **Review** section and wait for CI; squash-merge with the expected head SHA.
9. **Afterwards.** Delete the task's tracker line and description in that pull request. Move any lasting call into the [audit log](audit.md), and any repeatable lesson into **Traps**.
   If a report trigger fired, write the report as [the README](README.md#reporting-contract) says.

### Pull request

- Show the twin's operation sequence, a branch-to-fixture-or-refusal table, the trap answers, the bound and the proposal it follows, the build estimate or uncapped reason, timings, the gate result and any new need.
- In the **Review** section, check: the twin's order against installed sklearn (and scipy or numpy where the twin calls them) for one configuration; a fixture or a named refusal for each configuration that changes `transform`; bounds against step 6 and their proposal; the build declaration; regenerated, not hand-edited, coverage and bounds; a reproduction for each new need.
- Every number cites a script in the repository at the measured head; data may stay outside.
- For each timing, name the timed operation, revision, shape, return form, warm or cold state and repetitions. Report median serve time for 64-row calls against the twin on the same rows.

### When confit lacks a capability

- Never change `packages/confit/`. Add the need to the [confit tracker](../confit/tracker.md#needs) with a reproduction against confit alone, and link it as the blocker in the affected task.
- An entry that waits on confit stays `NotNative`, with a reason that names the capability.
- Keep a SQL spelling that exists only to trigger a confit cost rule (such as `x - 0.0`, which makes confit share repeated reads) in `_registry.py`, and keep its need open while the spelling remains.

### Files that tasks share

- `catalog_test.py` (`FIXTURES`): a conflict is mechanical; keep both sides.
- `coverage.md`, `bounds.md`: after merging master, regenerate them; never resolve their conflicts by hand.

### Environment

- Cloud sessions enter the pinned Nix dev shell through `.claude/hooks/session-start.sh`; do not install a separate uv or Python.
- Before timings, build the release extension from `packages/confit` with `uv run --no-sync maturin develop --release --skip-install`.

### Helper procedure

A runbook helper does steps 3 to 8.5 of **One task** for the task its assignment names, on that task's branch, then opens the pull request and stops at green CI. It answers each review comment with a push or a reply that gives its reason. It never merges.

## Traps

Finding numbers refer to the tables in the [reports](reports).

### Twin

- Validation: the input guard traps exactly where the twin's validation raises, never where it answers. (#423; findings 27, 44)
- Composition paths: inspect object-array validation, sparse input, sparse output and whole-row dtype, not only the bare estimator. (findings 40, 57, 70, 71)
- Sparse compositions: densification can turn −0.0 into +0.0; test the actual stack and weights. (#455; [sparse outputs](../../proposals/closed/sparse-outputs.md))
- Degenerate fits: inspect non-finite attributes, empty spectra, noise components and failed fits; refuse an unserved fitted state by name. (findings 78, 79; #494)
- Installed code: follow the installed sklearn/scipy/numpy call path, including delegation and upstream defects; refuse the shape that cannot be translated. (findings 37, 38)

### Numerics

- NaN ordering: DuckDB puts NaN above numbers; guard comparisons where the twin preserves NaN. (finding 1; #329)
- Signed zero: test zero products, empty sums, ties, one-feature and one-output shapes and one-row twins; a bound above 0 currently permits either zero sign. (findings 39, 77, 81; #462, #497)
- Reduction order: name pairwise, einsum, BLAS/LAPACK or scipy CSR stored order; an exact entry follows that order, including its initial zero. (finding 2; #329, #512)
- Cancellation: use the ruled error scale, not ulps of a near-zero result; include kernel arguments far from zero. (#509; findings 5, 46)
- Kernel probes: run through confit, never the pinned test oracle; any failure declines with `NotNative`. (finding 24; #367)
- Probe inputs: include random significands, not only `exp(uniform)`; exercise both probe outcomes without assuming this CPU's kernel. (findings 45, 54, 88)
- Overflow: test partial sums, mapped comparisons, underflow and both overflow edges; checker overflow is inconclusive, never a pass. ([matvec parity bound](../../proposals/closed/matvec-parity-bound.md); #513)
- Host-sensitive constants: use a derivation that covers the admitted hosts; agreement on one BLAS setup does not establish it. (#513)

### Fixtures

- Randomness: set every estimator's `random_state`; fixture draws must not depend on numpy's global random state. (finding 82; #497)
- Shapes: seed each configuration separately, so configurations do not all draw the same shapes. (finding 8; #347)
- Edge values: cover NaN, NULL, ±inf, ±0.0, subnormals, finite extremes, booleans, float32 fits, all-missing strings and branch neighbours. (findings 7, 51, 52, 55, 57; #399, #402, #415)
- Generator conversions: verify missing markers stay absent after rounding or dtype conversion. (finding 3; #332)
- Effective comparisons: NULL ids and rows where both sides trap do not establish numeric parity; require answered rows with non-NULL ids. (#415; findings 9, 52)

### Build cost

- Declaration: an `uncapped` declaration is not a time bound; some entries remain uncapped. (finding 91; #508)
- Quantity: estimate the whole warm `to_native`, not the trial build alone. (findings 61, 84, 92)
- Return form: list output and field-by-field struct reads grow differently; price the slower. (findings 76, 87; #502)
- Shape: counts of features, path steps or output fields are not costs; calibrate the family over shapes near its limits. (findings 22, 56, 58, 68, 80, 89)
- Probe calls: count calls per feature and category, including their work over every feature. (findings 65, 84)
- Depth: excessive expression depth and `RecursionError` decline with `NotNative`. (finding 86; #502)
- Heuristics: count the cost of confit-specific spellings in the estimates. (finding 85; #498)

### Measurement

- Release build: verify the loaded extension and revision; a Cargo build folder shared across revisions once measured the wrong confit. (findings 67, 74)
- Merged head: re-take timings after merging master; a confit merge can invalidate a cap. (finding 23; #367)
- Conditions: separate cold-container effects from code changes, record Python and BLAS settings, and avoid multi-threaded BLAS in each xdist process. (findings 4, 43; #340)
- Every field: count scalars, struct fields and list elements, with fresh row counts. (findings 75, 83)
