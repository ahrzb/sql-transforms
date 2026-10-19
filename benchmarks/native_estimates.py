"""Check build estimates, or propose coefficients without changing product code.

Run with a release build: python -m benchmarks.native_estimates --check [FAMILY]
or --fit FAMILY. Both commands time the whole warm to_native, median of three,
including translation, probes and the trial build. Fit data and return schemas
are made outside the timer. One untimed call warms each return before measuring.

CALIBRATION holds 3--5 fixed shapes per estimate, from its documented grids.
Each shape measures list and struct returns: admission prices the slower return,
so --check applies [0.5, 2] to that maximum, not to the deliberately faster list.
Every return and its measured/estimate ratio is printed. A refusal fails the
check; it never substitutes a smaller shape. Only the seconds cap is lifted:
structural caps and numerical/platform refusals remain active.

--fit adds FIT_EXTRA shapes to identify every declared term. It fits the slower
return of each shape, using column-scaled least squares, and prints coefficients
and a measured/predicted band for the estimate's docstring. It never edits a
coefficient. This is calibration, not approval of a regression or a new budget.
"""

from __future__ import annotations

import argparse
import dataclasses
import math
import platform
import statistics
import subprocess
import time
import warnings
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from unittest.mock import patch

import confit
import numpy as np
import pyarrow as pa
from confit import sql as S
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import FactorAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.isotonic import IsotonicRegression
from sklearn.kernel_approximation import (
    AdditiveChi2Sampler,
    Nystroem,
    RBFSampler,
    SkewedChi2Sampler,
)
from sklearn.preprocessing import (
    PowerTransformer,
    QuantileTransformer,
    SplineTransformer,
)
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from sql_transform._udf import PythonTransform
from sql_transform.native import _registry as R
from sql_transform.native import (
    distance,
    isotonic,
    kernel,
    power,
    project,
    quantile,
    spline,
    to_native,
    trees,
)
from threadpoolctl import threadpool_limits


@dataclass
class Shape:
    n: int
    k: int = 1
    options: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return f"n={self.n},k={self.k}" + "".join(
            f",{key}={value}" for key, value in self.options.items()
        )


ESTIMATES = {
    "isotonic": isotonic._build_estimate,
    "quantile": quantile._build_estimate,
    "spline": spline._build_estimate,
    "trees": trees._build_estimate,
    "project": project._build_estimate,
    "centered": project._centered_build_estimate,
    "sparse": project._sparse_build_estimate,
    "factor": project._factor_build_estimate,
    "distance": distance._build_estimate,
    "additive_chi2": kernel._additive_chi2_build_estimate,
    "rbf": kernel._rbf_build_estimate,
    "skewed": kernel._skewed_build_estimate,
    "nystroem": kernel._nystroem_build_estimate,
    "power": power._build_estimate,
}

# n = input features; k = thresholds/quantiles/knots/trees/components/steps.
# These are fixed workloads, never "the widest shape that still translates".
CALIBRATION = {
    "isotonic": [
        Shape(1, 18000),
        Shape(1, 24703, {"out_of_bounds": "clip"}),
        Shape(1, 28524, {"out_of_bounds": "raise"}),
    ],
    "quantile": [Shape(1, 8000), Shape(16, 750), Shape(512, 10, {"integers": True})],
    "spline": [
        Shape(160, 8, {"degree": 3, "extrapolation": "constant"}),
        Shape(160, 8, {"degree": 3, "extrapolation": "error"}),
        Shape(112, 8, {"degree": 5, "extrapolation": "continue"}),
    ],
    "trees": [
        Shape(8, 400, {"max_depth": 5}),
        Shape(128, 200, {"max_depth": 5}),
        Shape(8, 1, {"max_depth": None, "rows": 8000}),
    ],
    "project": [Shape(512, 96), Shape(1024, 48), Shape(4096, 2)],
    "centered": [Shape(512, 96), Shape(1024, 32), Shape(4096, 1)],
    "sparse": [Shape(4096, 32), Shape(1024, 480), Shape(16, 1024, {"repeat": 512})],
    "factor": [Shape(256, 128), Shape(1024, 24), Shape(4096, 1)],
    "distance": [Shape(2, 800), Shape(512, 80), Shape(2048, 4)],
    "additive_chi2": [Shape(896, 1), Shape(128, 3), Shape(3, 400)],
    "rbf": [
        Shape(512, 64, {"apart": True}),
        Shape(1024, 24, {"apart": True}),
        Shape(2048, 8, {"apart": False}),
    ],
    "skewed": [Shape(1, 768), Shape(768, 2), Shape(2048, 1)],
    "nystroem": [Shape(8, 250), Shape(256, 150), Shape(2048, 20)],
    "power": [
        Shape(640, options={"method": "box-cox", "standardize": False}),
        Shape(320, options={"method": "yeo-johnson", "standardize": False}),
        Shape(640, options={"method": "box-cox", "standardize": True}),
        Shape(288, options={"method": "yeo-johnson", "standardize": True}),
    ],
}

# More independent rows than coefficients, including each conditional branch.
# These smaller shapes come from the same documented ranges as CALIBRATION.
FIT_EXTRA = {
    "isotonic": [Shape(1, 3418), Shape(1, 12000)],
    "quantile": [Shape(8, 500), Shape(32, 500, {"integers": True})],
    "spline": [
        Shape(32, 5, {"degree": 1, "extrapolation": "linear"}),
        Shape(64, 8, {"degree": 3, "extrapolation": "periodic"}),
    ],
    "trees": [Shape(8, 100, {"max_depth": 5}), Shape(32, 250, {"max_depth": 5})],
    "project": [Shape(n, k) for n, k in [(64, 16), (256, 128), (2048, 1), (1024, 8)]],
    "centered": [
        Shape(n, k)
        for n, k in [(64, 16), (256, 128), (2048, 1), (1024, 8), (2048, 4), (128, 64)]
    ],
    "sparse": [
        Shape(n, k, {"density": d})
        for n, k, d in [
            (64, 128, 0.1),
            (256, 64, 1 / 3),
            (1024, 32, "auto"),
            (2048, 16, "auto"),
            (16, 512, 0.1),
        ]
    ]
    + [Shape(64, 512, {"repeat": 128}), Shape(16, 512, {"repeat": 256})],
    "factor": [
        Shape(n, k) for n, k in [(64, 16), (128, 64), (512, 16), (2048, 1), (1024, 4)]
    ],
    "distance": [
        Shape(n, k)
        for n, k in [(64, 200), (256, 32), (1024, 1), (512, 16), (128, 128), (8, 400)]
    ],
    "additive_chi2": [Shape(n, k) for n, k in [(128, 1), (64, 3), (32, 24), (8, 100)]],
    "rbf": [
        Shape(n, k, {"apart": a})
        for a in [False, True]
        for n, k in [(128, 64), (512, 32), (1024, 8), (2048, 1), (256, 128), (64, 256)]
    ],
    "skewed": [Shape(n, k) for n, k in [(2, 256), (32, 32), (512, 1), (128, 8)]],
    "nystroem": [Shape(n, k) for n, k in [(64, 100), (512, 32), (1024, 1), (128, 64)]],
    "power": [
        Shape(n, options={"method": method, "standardize": standardize})
        for method, n in [
            ("box-cox", 256),
            ("box-cox", 512),
            ("yeo-johnson", 128),
            ("yeo-johnson", 256),
        ]
        for standardize in [False, True]
    ],
}


def fitted(family: str, shape: Shape):
    """Deterministic fitted state; no calibration imports a pytest fixture."""
    n, k, options = shape.n, shape.k, dict(shape.options)
    rng = np.random.default_rng(0)
    if family == "isotonic":
        x = np.arange(float(k))
        return IsotonicRegression(**options).fit(x, x + 0.1 * np.sin(x))
    if family == "quantile":
        x = (
            rng.integers(0, 5, size=(max(k, 32), n)).astype(float)
            if options.pop("integers", False)
            else rng.normal(size=(max(k, 32), n))
        )
        return QuantileTransformer(n_quantiles=k, subsample=None).fit(x)
    if family == "spline":
        return SplineTransformer(n_knots=k, **options).fit(rng.normal(size=(64, n)))
    if family == "trees":
        rows = options.pop("rows", 2000)
        return RandomTreesEmbedding(
            n_estimators=k, random_state=0, sparse_output=False, **options
        ).fit(rng.normal(size=(rows, n)))
    if family in ("project", "centered"):
        return GaussianRandomProjection(n_components=k, random_state=0).fit(
            rng.normal(size=(10, n))
        )
    if family == "sparse":
        repeat = options.pop("repeat", None)
        est = SparseRandomProjection(n_components=k, random_state=0, **options).fit(
            rng.normal(size=(10, n))
        )
        if repeat is not None:
            indices = [rng.permutation(n)[:2] for _ in range(repeat)]
            est.components_ = sparse.csr_matrix(
                (
                    np.concatenate(
                        [[1.0, -1.0 - (j % repeat) * 2.0**-20] for j in range(k)]
                    ),
                    np.concatenate([indices[j % repeat] for j in range(k)]),
                    np.arange(0, 2 * k + 1, 2),
                ),
                shape=(k, n),
            )
        return est
    if family == "factor":
        return FactorAnalysis(n_components=k, max_iter=5, random_state=0).fit(
            rng.normal(size=(max(k + 4, 32), n))
        )
    if family == "distance":
        return KMeans(n_clusters=k, n_init=1, max_iter=1, random_state=0).fit(
            rng.normal(size=(max(k, 32), n))
        )
    if family == "power":
        return PowerTransformer(**options).fit(rng.lognormal(size=(64, n)))
    if family == "additive_chi2":
        return AdditiveChi2Sampler(sample_steps=k, sample_interval=0.4).fit(
            rng.random((10, n))
        )
    if family == "rbf":
        return RBFSampler(n_components=k, random_state=0).fit(rng.normal(size=(10, n)))
    if family == "skewed":
        return SkewedChi2Sampler(n_components=k, random_state=0).fit(
            rng.random((10, n))
        )
    if family == "nystroem":
        return Nystroem(n_components=k, random_state=0).fit(
            rng.normal(size=(max(k, 32), n))
        )
    raise ValueError(family)


@contextmanager
def admitted(family: str, shape: Shape, est):
    """Lift seconds caps only; hold the read spelling whose formula is timed."""
    with ExitStack() as stack:
        for module in (
            distance,
            isotonic,
            kernel,
            power,
            project,
            quantile,
            spline,
            trees,
        ):
            stack.enter_context(patch.object(module, "MAX_BUILD_S", math.inf))
        if family in ("project", "centered", "rbf"):
            apart = family == "centered" or (family == "rbf" and shape.options["apart"])
            entry = dataclasses.replace(R._CATALOG[type(est)], apart_of=lambda _: apart)
            stack.enter_context(patch.dict(R._CATALOG, {type(est): entry}))
        yield


def arguments(family: str, shape: Shape, est) -> tuple:
    n, k = shape.n, shape.k
    if family == "isotonic":
        return (len(est.X_thresholds_) - 1,)
    if family == "trees":
        return (est,)
    if family == "quantile":
        leaves = [quantile._leaves_of(q, est.references_) for q in est.quantiles_.T]
        return (
            sum(map(len, leaves)),
            sum(at is not None for tree in leaves for _, _, at in tree),
            n,
        )
    if family == "spline":
        x = [R._feature(S.col(f"x{i}"), pa.float64()) for i in range(n)]
        token = R._GUARD.set(R._Guard("list", None))
        try:
            with admitted(family, shape, est):
                fields = spline._spline(
                    est, R._apart(x, [pa.float64()] * n), [pa.float64()] * n
                )
        finally:
            R._GUARD.reset(token)
        return (fields, est)
    if family == "sparse":
        return (est.components_,)
    if family == "power":
        return (n, est.method, est.standardize)
    if family == "rbf":
        return (n, k, shape.options["apart"])
    return (n, k)


def step_for(est, returns: str) -> PythonTransform:
    n = est.n_features_in_ if hasattr(est, "n_features_in_") else 1
    # Positive, inside the isotonic/spline fitted domain and valid for Box-Cox.
    out = est.transform(np.ones((1, n)))
    width = out.shape[-1] if out.ndim > 1 else 1
    returns_type = (
        (pa.float64() if width == 1 else pa.list_(pa.float64(), width))
        if returns == "list"
        else pa.struct([(f"f{i}", pa.float64()) for i in range(width)])
    )
    return PythonTransform(
        "tf",
        {0: est},
        pa.schema([(f"x{i}", pa.float64()) for i in range(n)]),
        returns_type,
    )


def measure(step: PythonTransform) -> tuple[float, float]:
    """Median whole warm to_native and its trial-build portion, in seconds."""
    to_native(step, strict=True, allow_bound=True)
    whole, trial = [], []
    builds = R._builds
    for _ in range(3):
        elapsed = []

        def timed_build(*args, elapsed=elapsed, **kwargs):
            start = time.perf_counter()
            try:
                return builds(*args, **kwargs)
            finally:
                elapsed.append(time.perf_counter() - start)

        with patch.object(R, "_builds", timed_build):
            start = time.perf_counter()
            to_native(step, strict=True, allow_bound=True)
            whole.append(time.perf_counter() - start)
        trial.append(sum(elapsed))
    return statistics.median(whole), statistics.median(trial)


def fit_coefficients(estimate: R.BuildEstimate, rows: list[tuple[dict, float]]):
    names = list(estimate.coefficients)
    a = np.array([[terms.get(name, 0.0) for name in names] for terms, _ in rows])
    y = np.array([seconds for _, seconds in rows])
    scale = np.linalg.norm(a, axis=0)
    if np.any(scale == 0):
        raise ValueError("fit shapes do not exercise every declared term")
    coefficients, _, rank, _ = np.linalg.lstsq(a / scale, y, rcond=None)
    if rank != len(names):
        raise ValueError(
            f"fit grid has rank {rank}, needs {len(names)} independent terms"
        )
    coefficients /= scale
    predicted = a @ coefficients
    if np.any(predicted <= 0):
        raise ValueError(
            "fit predicts a nonpositive time; change the model or its grid"
        )
    ratio = y / predicted
    return dict(zip(names, coefficients, strict=True)), (
        float(ratio.min()),
        float(ratio.max()),
    )


def run(family: str, fit: bool = False) -> bool:
    estimate = ESTIMATES[family]
    rows = []
    ok = True
    shapes = CALIBRATION[family] + (FIT_EXTRA[family] if fit else [])
    for shape in shapes:
        try:
            est = fitted(family, shape)
            args = arguments(family, shape, est)
            predicted = estimate(*args)
            times = []
            for returns in ("list", "struct"):
                step = step_for(est, returns)
                with admitted(family, shape, est):
                    whole, trial = measure(step)
                times.append(whole)
                returns = "scalar" if pa.types.is_float64(step.returns) else returns
                print(
                    f"| {family} | {shape.label} | {returns} | {predicted:.4f}"
                    f" | {whole:.4f} | {whole / predicted:.3f} | {trial:.4f}"
                    f" | {whole / trial:.3f} |",
                    flush=True,
                )
            ratio = max(times) / predicted
            ok &= 0.5 <= ratio <= 2.0
            rows.append((estimate.terms(*args), max(times)))
        except (R.NotNative, ValueError, RecursionError) as exc:
            print(f"{family} {shape.label}: REFUSED: {exc}", flush=True)
            ok = False
    if fit:
        if len(rows) != len(shapes):
            print(f"{family}: no fit: a fixed workload refused", flush=True)
            return False
        coefficients, band = fit_coefficients(estimate, rows)
        print(
            f"\n{family}: proposed coefficients (warm_factor={estimate.warm_factor:g})"
        )
        for name, coefficient in coefficients.items():
            print(f"    {name}={coefficient:.8g},")
        print(
            "Docstring: whole warm to_native, release build, median of three; "
            f"{len(rows)} shapes, slower of list and struct; each took "
            f"{band[0]:.2f} to {band[1]:.2f} times the estimate. {provenance()}"
        )
        return True
    return ok


def provenance() -> str:
    root = Path(__file__).resolve().parents[1]
    sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],  # noqa: S607 — local checkout's Git
        cwd=root,
        text=True,
    ).strip()
    return (
        f"{date.today()}, {platform.platform()}, {platform.machine()}, "
        f"Python {platform.python_version()}, confit {confit.BUILD_PROFILE}, HEAD {sha}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", nargs="?", const="all", choices=["all", *ESTIMATES])
    mode.add_argument("--fit", choices=ESTIMATES)
    args = parser.parse_args()
    if confit.BUILD_PROFILE != "release":
        parser.error(f"confit is a {confit.BUILD_PROFILE} build; use release")
    print(provenance(), flush=True)
    print(
        "| family | shape | return | estimate s | warm to_native s"
        " | measured/estimate | trial s | whole/trial |"
    )
    print("|---|---|---|---:|---:|---:|---:|---:|")
    families = list(ESTIMATES) if args.check == "all" else [args.fit or args.check]
    with warnings.catch_warnings(), threadpool_limits(1):
        warnings.simplefilter("ignore")
        results = [run(family, fit=args.fit is not None) for family in families]
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
