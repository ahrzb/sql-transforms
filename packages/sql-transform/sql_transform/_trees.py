"""Fitted sklearn ensembles -> the two Arrow tables confit scores natively.

Only numbers cross: no estimator object, no pickle, no live model reference
reaches the engine. The engine stays sklearn-free; this module is the one
place that knows what a `tree_` looks like.

Bit-exactness is the whole point, so the packing mirrors each family's own
summation order rather than a tidier one:

- a forest sums its trees and divides — `agg="mean"`, `base=0`;
- a boosted model SEEDS the accumulator with its init prediction and adds
  `learning_rate * value` per stage — `agg="sum"`, base = init, and the
  scaling is folded into the leaf values here so the engine performs the same
  multiply-then-add sequence sklearn's `predict_stages` does.

Measured 2026-08-07: applying the base after the sum instead of seeding with
it diverges from sklearn on up to 1365 of 2000 rows (632 ULP), and summing
pairwise (`arr.sum(axis=1)`) on up to 1853 of 2000. Neither is a rounding
detail you can wave off.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pyarrow as pa
from sklearn.base import is_regressor

__all__ = ["TreePackError", "pack_trees"]


class TreePackError(ValueError):
    """A fitted estimator this packer refuses to describe."""


def _stages(est: Any) -> tuple[list[Any], float, str, float]:
    """(trees, base, agg, leaf scale) for one fitted estimator."""
    cls = type(est).__name__
    # A classifier's tree stores per-class scores in `values`, not the number
    # `predict` returns — and it has a `tree_` just like a regressor does, so
    # without this check it would pack cleanly and score class-0 fractions.
    if not is_regressor(est):
        raise TreePackError(
            f"{cls}: only regressors pack — a classifier's leaf values are"
            " class scores, not predictions"
        )
    # `values[:, 0, 0]` takes target 0, so a 2-D y would pack cleanly and
    # serve `predict(X)[:, 0]` with the other targets silently gone — the
    # same shape of failure BaggingRegressor is refused for, one axis over.
    n_out = getattr(est, "n_outputs_", 1)
    if n_out != 1:
        raise TreePackError(
            f"{cls}: multi-output regression is not supported (n_outputs_"
            f"={n_out}) — one model set scores one number per row"
        )
    if hasattr(est, "tree_"):
        return [est], 0.0, "sum", 1.0
    if cls in ("RandomForestRegressor", "ExtraTreesRegressor"):
        return list(est.estimators_), 0.0, "mean", 1.0
    if cls == "GradientBoostingRegressor":
        if getattr(est, "n_classes_", 1) != 1:
            raise TreePackError(f"{cls}: multi-output boosting is not supported")
        init = est.init_
        # A data-dependent init makes `base` a per-ROW value, which is not
        # something a per-model header can carry. `init="zero"` stays a bare
        # string on the fitted estimator.
        if (
            init is not None
            and init != "zero"
            and type(init).__name__ != "DummyRegressor"
        ):
            raise TreePackError(
                f"{cls}: init={type(init).__name__} is not a constant — only"
                " 'zero' and the default DummyRegressor pack to a base term"
            )
        zeros = np.zeros((1, est.n_features_in_))
        base = float(est._raw_predict_init(zeros)[0, 0])
        return list(est.estimators_[:, 0]), base, "sum", float(est.learning_rate)
    # BaggingRegressor is deliberately absent. Its `estimators_features_`
    # gives each tree a feature SUBSET, so that tree's `feature` ids are
    # subset-local — reading them as global indices scores the wrong columns
    # and answers plausibly. Its base estimator need not be a tree at all.
    raise TreePackError(f"{cls} is not a tree ensemble this packer knows")


def pack_trees(estimators: Sequence[Any], features: Sequence[str]) -> dict[str, Any]:
    """Fitted regressors in, a confit `models=` entry out.

    Model ids are dense and follow the sequence order, so the params table's
    instance-id column indexes straight into it.
    """
    estimators = list(estimators)
    if not estimators:
        raise TreePackError("no estimators to pack")
    features = list(features)

    cols: dict[str, list[np.ndarray]] = {
        k: []
        for k in (
            "model_id",
            "tree_id",
            "node_id",
            "feature",
            "threshold",
            "left",
            "right",
            "missing_left",
            "value",
        )
    }
    headers = {"model_id": [], "base": [], "agg": [], "link": []}
    for mid, est in enumerate(estimators):
        if getattr(est, "n_features_in_", len(features)) != len(features):
            raise TreePackError(
                f"model {mid} was fitted on {est.n_features_in_} features,"
                f" {len(features)} names given"
            )
        trees, base, agg, scale = _stages(est)
        for tid, t in enumerate(trees):
            state = t.tree_.__getstate__()
            nodes, values = state["nodes"], state["values"]
            n = len(nodes)
            left = nodes["left_child"].astype(np.int32)
            # sklearn marks a leaf with feature = TREE_UNDEFINED (-2) and both
            # children -1; the engine's leaf marker is feature = -1.
            feature = np.where(left == -1, -1, nodes["feature"]).astype(np.int32)
            cols["model_id"].append(np.full(n, mid, np.int64))
            cols["tree_id"].append(np.full(n, tid, np.int64))
            cols["node_id"].append(np.arange(n, dtype=np.int64))
            cols["feature"].append(feature)
            cols["threshold"].append(nodes["threshold"].astype(np.float64))
            cols["left"].append(left)
            cols["right"].append(nodes["right_child"].astype(np.int32))
            cols["missing_left"].append(nodes["missing_go_to_left"].astype(bool))
            # `scale * value` is folded in here so the engine's per-tree add is
            # the same IEEE operation on the same double sklearn accumulates.
            cols["value"].append(scale * values[:, 0, 0].astype(np.float64))
        headers["model_id"].append(mid)
        headers["base"].append(base)
        headers["agg"].append(agg)
        headers["link"].append("identity")

    arrow_ty = {
        "model_id": pa.int64(),
        "tree_id": pa.int64(),
        "node_id": pa.int64(),
        "feature": pa.int32(),
        "threshold": pa.float64(),
        "left": pa.int32(),
        "right": pa.int32(),
        "missing_left": pa.bool_(),
        "value": pa.float64(),
    }
    return {
        "nodes": pa.table(
            {k: pa.array(np.concatenate(v), type=arrow_ty[k]) for k, v in cols.items()}
        ),
        "models": pa.table(
            {
                "model_id": pa.array(headers["model_id"], pa.int64()),
                "base": pa.array(headers["base"], pa.float64()),
                "agg": pa.array(headers["agg"], pa.string()),
                "link": pa.array(headers["link"], pa.string()),
            }
        ),
        "features": features,
    }
