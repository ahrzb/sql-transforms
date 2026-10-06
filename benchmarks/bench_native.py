"""Native catalog bench: what a native entry buys over its Python twin.

For every catalog class, the widest step the catalog's fixtures draw
(`sql_transform.native.catalog_test`, seeds 0-7, every configuration) is
measured three ways, on 64-row calls of rows every twin answers (ids of
fitted instances, finite numbers, inside the fitted range where the
transform raises outside it, and for an encoder the row's instance's
fitted categories; a row the twin still rejects is drawn again):

  build    building the query that reads every lane, with the native entry
           (`DuckDBInferFn`, what `to_native`'s trial build and serving pay
           once per deployment)
  native   one 64-row call served by the native entry (median)
  twin     the same call served by the Python step (median)

The speedup is twin / native. Timings need confit's release build.

With --build-curve, it times the build alone against the width instead,
`to_native` on one step shape over n DOUBLE features: PolynomialFeatures
(degree 2), a Normalizer (l1, l2), or a three-group StandardScaler.

Run: uv run python -m benchmarks.bench_native [--calls 30]
     uv run python -m benchmarks.bench_native --build-curve 20 30 45 60
         [--shape polynomial|l1|l2|standard]
"""

from __future__ import annotations

import argparse
import math
import numbers
import random
import statistics
import time
import warnings

import confit
import numpy as np
import pyarrow as pa
from confit import DuckDBInferFn
from sklearn.preprocessing import Normalizer, PolynomialFeatures, StandardScaler
from sql_transform._udf import PythonTransform
from sql_transform.native import catalog_test as fixtures
from sql_transform.native import to_native
from sql_transform.native._registry import NotNative, query
from threadpoolctl import threadpool_limits

ROWS = 64
SEEDS = range(8)


def _lanes(step) -> int:
    r = step.returns
    if pa.types.is_struct(r):
        return r.num_fields
    if pa.types.is_fixed_size_list(r):
        return r.list_size
    return 1


def widest(cls) -> tuple[object, object, int, int, list[str]]:
    """The widest fixture step of `cls` that translates: (step, native,
    configuration, seed, and why each wider one stays Python)."""
    drawn = []
    for j, make in enumerate(fixtures.FIXTURES[cls]):
        for seed in SEEDS:
            step = fixtures._step(make, seed, j)
            drawn.append(((_lanes(step), len(step.takes)), j, seed, step))
    drawn.sort(key=lambda d: d[0], reverse=True)
    declined = []
    for _, j, seed, step in drawn:
        try:
            return step, to_native(step, strict=True), j, seed, declined
        except NotNative as e:
            declined.append(str(e))
    raise SystemExit(f"no fixture of {cls.__name__} translates")


def _domain(est, i: int) -> tuple[float, float] | None:
    """Feature `i`'s fitted range, where `est`'s transform raises outside
    it: a spline's knots under `extrapolation="error"`, an isotonic fit's
    thresholds under `out_of_bounds="raise"`. None elsewhere."""
    splines = getattr(est, "bsplines_", None)
    if splines is not None and getattr(est, "extrapolation", None) == "error":
        t, k = splines[i].t, splines[i].k
        return float(t[k]), float(t[len(t) - k - 1])
    if hasattr(est, "X_min_") and getattr(est, "out_of_bounds", None) == "raise":
        return float(est.X_min_), float(est.X_max_)
    return None


def _value(rng: random.Random, est, i: int, f: pa.Field, positive: bool):
    if f.type == pa.bool_():
        # A boolean feature (the step hands it to `transform` as a bool).
        return rng.random() < 0.5
    cats = getattr(est, "categories_", None)
    if cats is not None:
        # A category the step can hand over (NaN and None are missing: a
        # feature with only those reads NULL).
        fitted = [c for c in cats[i] if isinstance(c, str | numbers.Real)]
        fitted = [c for c in fitted if c == c]
        return rng.choice(fitted) if fitted else None
    if f.type == pa.string():
        return rng.choice(fixtures.VOCAB)
    domain = _domain(est, i)
    if f.type == pa.int64():
        if domain and math.ceil(domain[0]) <= math.floor(domain[1]):
            return rng.randint(math.ceil(domain[0]), math.floor(domain[1]))
        return rng.randint(1, 3) if positive else rng.randint(-3, 3)
    lo, hi = domain or ((1e-3 if positive else -1e3), 1e3)
    return rng.uniform(lo, hi)


def rows(step, seed: int, positive: bool = False) -> pa.Table:
    """64 rows the twin answers: fitted ids, finite values (strictly
    positive for an estimator that rejects the rest, as the fixture's
    `positive` flag says; inside the fitted range where the transform
    raises outside it), and for an encoder each row's values among its own
    instance's categories. A drawn row the twin still raises on is drawn
    again."""
    rng = random.Random(seed)  # noqa: S311
    ids: list[int] = []
    drawn: list[list] = []
    for _ in range(50 * ROWS):
        k = rng.choice(list(step.instances))
        vals = [
            _value(rng, step.instances[k], i, f, positive)
            for i, f in enumerate(step.takes)
        ]
        try:
            step(k, *vals)
        except Exception:  # noqa: BLE001, S112 — a row the twin rejects
            continue
        ids.append(k)
        drawn.append(vals)
        if len(ids) == ROWS:
            break
    else:
        raise SystemExit(f"{step.name}: no {ROWS} rows the twin answers")
    cols: dict[str, pa.Array] = {"__iid": pa.array(ids, pa.int64())}
    for i, f in enumerate(step.takes):
        vals = [r[i] for r in drawn]
        if f.type == pa.int64():
            vals = [None if v is None else int(v) for v in vals]
        elif f.type == pa.bool_():
            vals = [None if v is None else bool(v) for v in vals]
        elif f.type != pa.string():
            vals = [None if v is None else float(v) for v in vals]
        cols[f.name] = pa.array(vals, f.type)
    return pa.table(cols)


def _median_call(fn: DuckDBInferFn, table: pa.Table, calls: int) -> float:
    fn.infer_arrow(table)  # warm
    times = []
    for _ in range(calls):
        t = time.perf_counter()
        fn.infer_arrow(table)
        times.append(time.perf_counter() - t)
    return statistics.median(times)


def _curve_step(shape: str, n: int) -> PythonTransform:
    """A one-step build-curve subject over n DOUBLE features: `polynomial`
    (degree 2, one instance), `l1`/`l2` (a Normalizer, one instance) or
    `standard` (a StandardScaler, three instances)."""
    x = np.random.default_rng(0).normal(size=(10, n))
    if shape == "polynomial":
        ests = [PolynomialFeatures(degree=2).fit(x)]
    elif shape in ("l1", "l2"):
        ests = [Normalizer(norm=shape).fit(x)]
    elif shape == "standard":
        ests = [StandardScaler().fit(x + k) for k in range(3)]
    else:
        raise SystemExit(f"no build-curve shape {shape!r}")
    lanes = np.asarray(ests[0].transform(x[:1])).shape[1]
    return PythonTransform(
        "tf",
        dict(enumerate(ests)),
        pa.schema([(f"x{i}", pa.float64()) for i in range(n)]),
        pa.struct([(f"f{i}", pa.float64()) for i in range(lanes)]),
    )


def build_curve(shape: str, widths: list[int]) -> None:
    """Build time against width: `to_native` on `_curve_step(shape, n)`,
    whose trial build reads every lane (a refusal is timed too)."""
    print(f"| {shape}: features | lanes | build (s) |")
    print("|---|---|---|")
    for n in widths:
        step = _curve_step(shape, n)
        t = time.perf_counter()
        try:
            to_native(step, strict=True)
            what = ""
        except NotNative:
            what = ", refused"
        print(
            f"| {n} | {_lanes(step):,} | {time.perf_counter() - t:.2f}{what} |",
            flush=True,
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", type=int, default=30)
    ap.add_argument(
        "--build-curve",
        type=int,
        nargs="*",
        metavar="N",
        help="time builds of one --shape over N features instead",
    )
    ap.add_argument(
        "--shape",
        default="polynomial",
        choices=["polynomial", "l1", "l2", "standard"],
        help="the --build-curve subject (default: polynomial)",
    )
    args = ap.parse_args()
    if confit.BUILD_PROFILE != "release":
        raise SystemExit(f"confit is a {confit.BUILD_PROFILE} build: time a release")
    warnings.simplefilter("ignore")
    threadpool_limits(1)
    if args.build_curve is not None:
        build_curve(args.shape, args.build_curve or [20, 30, 45, 60])
        return
    print(
        "| class | config, seed | features | lanes | instances | build (s)"
        " | native (µs/call) | twin (µs/call) | speedup |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    capped = {}
    for cls in fixtures.FIXTURES:
        step, native, j, seed, declined = widest(cls)
        if declined:
            capped[cls.__name__] = declined
        table = rows(step, seed, getattr(fixtures.FIXTURES[cls][j], "positive", False))
        sql = query(step)
        schema = {"__THIS__": table.schema}
        t = time.perf_counter()
        fn = DuckDBInferFn(sql, row_tables=schema, static_tables={}, udfs=[native])
        build = time.perf_counter() - t
        nat = _median_call(fn, table, args.calls)
        twin_fn = DuckDBInferFn(sql, row_tables=schema, static_tables={}, udfs=[step])
        twin = _median_call(twin_fn, table, args.calls)
        print(
            f"| `{cls.__name__}` | {j}, {seed} | {len(step.takes)} | {_lanes(step)}"
            f" | {len(step.instances)} | {build:.2f} | {nat * 1e6:,.0f}"
            f" | {twin * 1e6:,.0f} | {twin / nat:,.0f}x |",
            flush=True,
        )
    for name, why in capped.items():
        print(f"\n{name}: {len(why)} wider fixture steps stay Python, e.g. {why[0]}")


if __name__ == "__main__":
    main()
