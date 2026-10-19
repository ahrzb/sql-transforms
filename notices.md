# Notices

**Read up to:** none

## 2026-10-17 reports-moved

**From:** the session that moved the loops into mandates.
**Audit:** [granted from the loop](mandates/native/audit.md#2026-10-17-granted-from-the-loop).

The reports moved from `loops/<loop>/reports/` to `mandates/<mandate>/reports/`. Their file names did not change. The mandate folders keep the loop names `native` and `confit`. The front matter that the maintainer's tracker reads keeps its keys. If the tracker reads the old folders, it will find no new reports.

## 2026-10-17 old-layouts-remain

**From:** the session that moved the loops into mandates.

Only the loops moved to the layout of the `repo-docs` skill. These parts keep their older layouts:

- `packages/sql-transform/` keeps its own `research/`, `plan/` and `records/` folders, with decision records and RFCs.
- `packages/confit/docs/` keeps its specs, oracle chapters and reports.
- The specs name each rule as `**claim: <name>.**`. They do not use an explicit anchor such as `<a id="...">`.
- `backlog/` holds a list in the format of the Backlog.md tool.

`AGENTS.md` records the first two as exceptions to that layout.

## 2026-10-17 windows-baseline-failures

**From:** the session that moved the loops into mandates.

On Windows, master fails native tests that continuous integration (CI) on Linux passes. The runs used Python 3.14 and a release build of confit. At `40f6ab17`, three tests fail:

- Two tests in `kernel_test.py` expect `NotNative` where a long double is wider than a double. On Windows, the two types are the same.
- A test in `power_test.py` checks that the catalog's formula for `expm1` is within 2 ulps of the C library. On Windows it is 3 ulps away.

At `abc055ce` (#514, which adds carried bounds), five more fail: three cases of `test_a_carry_bounds_the_two_sides` in `_carry_test.py`, where a pair is 1.00 to 1.04 times its carried bound apart, and two tests in `compose_test.py` that expect a `ParityError`. [INFERENCE] The carried bound is computed in long double, which is a double on Windows.

## 2026-10-17 carried-bound-tests-fail-on-ci

**From:** the session that moved the loops into mandates.

Two tests that #514 added fail in CI on Linux for pull requests that do not touch their code: #515, which changes only docs, and #517, which adds the bounds manifest. They are `test_a_carried_bound_grows_as_the_step_that_carries_it` and `test_a_weight_carries_its_part_s_bound_times_the_weight` in `compose_test.py`. Each replaces a step's carried bound with a smaller one, and expects `check` to raise `ParityError`; `check` raises nothing. The same two tests passed in CI on #514 and on master at `abc055ce`, and fail on Windows at `abc055ce`. [INFERENCE] Whether `check` sees a row past the smaller bound depends on the host's BLAS and CPU, so these mutation tests pass or fail with the CI runner.
