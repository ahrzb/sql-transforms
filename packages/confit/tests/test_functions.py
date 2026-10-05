"""The function classes (`confit.functions`): each is served at parity with
its own definition, and refuses a bad declaration while it is being made.

`ExternFunction` and `Ensemble` are checked twice against DuckDB: through
the campaign verdict, which registers them by the oracle's own recipe from
the documented protocol, and through their own `register`, which is the
spelling a user gets.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from confit import DuckDBInferFn, Ensemble, ExternFunction, FunctionError
from confit import compare
from confit.functions import _int_to_f32
from confit.oracle import Oracle

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import gen as G  # noqa: E402
from fuzz import trees  # noqa: E402
from fuzz.parity import assert_parity, table  # noqa: E402

F = pa.schema([("x", pa.float64())])


def _double(x):
    return None if x is None else (2.0 * x,)


DOUBLE = ExternFunction("dbl", F, pa.float64(), _double)
SPLIT = ExternFunction(
    "split",
    pa.schema([("a", pa.int64()), ("s", pa.string())]),
    pa.struct([("n", pa.int64()), ("u", pa.string())]),
    lambda a, s: None if a is None else (a + 1, None if s is None else s.upper()),
)
PAIR = ExternFunction(
    "pair", F, pa.list_(pa.float64(), 2), lambda x: None if x is None else (x, -x)
)

ROWS = table(
    {"a": "int?", "s": "str?", "x": "float?"},
    [
        {"a": 1, "s": "ab", "x": 1.5},
        {"a": None, "s": None, "x": None},
        {"a": -7, "s": "Zz", "x": -0.0},
    ],
)


@pytest.mark.parametrize(
    "sql, fn",
    [
        ("SELECT dbl(x) AS d FROM __THIS__", DOUBLE),
        ("SELECT dbl(dbl(x)) + 1.0 AS d FROM __THIS__ WHERE dbl(x) > 0", DOUBLE),
        ("SELECT split(a, s).n AS n, split(a, s).u AS u FROM __THIS__", SPLIT),
        ("SELECT pair(x) AS p FROM __THIS__", PAIR),
    ],
)
def test_an_extern_function_agrees_with_the_oracle(sql, fn):
    assert_parity(sql, ROWS, udfs=[fn])


def _own_register(sql, rows, fns, statics=None):
    """Engine rows against DuckDB with the functions registered by their own
    `register`."""
    fn = DuckDBInferFn(
        sql, row_tables={"__THIS__": rows.schema}, static_tables=statics or {}, udfs=fns
    )
    got = fn.infer_arrow(rows).to_pylist()
    with Oracle() as o:
        for f in fns:
            f.register(o)
        for name, t in (statics or {}).items():
            o.load(name, t)
        o.load("__THIS__", rows)
        want = o.answer(sql).to_pylist()
    compare.assert_rows(got, want, ctx=sql)


@pytest.mark.parametrize(
    "sql, fn",
    [
        ("SELECT dbl(x) AS d FROM __THIS__", DOUBLE),
        ("SELECT split(a, s).n AS n, split(a, s).u AS u FROM __THIS__", SPLIT),
        ("SELECT pair(x) AS p FROM __THIS__", PAIR),
    ],
)
def test_register_is_the_definition_duckdb_runs(sql, fn):
    _own_register(sql, ROWS, [fn])


def test_a_structural_protocol_object_is_still_accepted():
    class Plain:
        name = "dbl"
        takes = F
        returns = pa.float64()

        def __call__(self, x):
            return _double(x)

    assert_parity("SELECT dbl(x) AS d FROM __THIS__", ROWS, udfs=[Plain()])


# ------------------------------------------------------------------ ensemble


def _ensemble(seed: int, kind: str) -> tuple[Ensemble, trees.TreeModel]:
    rng = random.Random(seed)  # noqa: S311
    n = rng.randint(1, 3)
    spec = G.TreeSpec(
        kind=kind,
        n_features=n,
        int_features=tuple(i for i in range(n) if rng.random() < 0.5),
        depth=rng.randint(1, 4),
        instances=rng.randint(1, 3),
    )
    ref = trees.make_tree(spec, seed)
    nodes, models, grid = ref.tree_tables()
    return Ensemble(ref.name, ref.takes, nodes, models, grid), ref


def _features(rng: random.Random, takes: pa.Schema) -> list:
    out = []
    for t in takes.types:
        if rng.random() < 0.15:
            out.append(None)
        elif t == pa.int64():
            out.append(rng.choice(trees._INT_POOL) + rng.choice([0, 1, -1]))
        else:
            out.append(rng.choice(G.QUANT) + rng.choice([0.0, 0.005, -0.005]))
    return out


@pytest.mark.parametrize("seed", range(40))
def test_the_ensemble_walk_is_the_fuzzers_reference_walk(seed):
    # Two independent spellings of the protocol's reference semantics.
    ens, ref = _ensemble(seed, ("dtr", "rf", "gbr")[seed % 3])
    rng = random.Random(seed)  # noqa: S311
    for _ in range(50):
        iid = rng.choice([None, *ens.instances])
        feats = _features(rng, ens.takes)
        assert ens(iid, *feats) == ref(iid, *feats)


def _score_rows(ens: Ensemble, seed: int) -> pa.Table:
    rng = random.Random(seed)  # noqa: S311
    rows = []
    for _ in range(12):
        feats = _features(rng, ens.takes)
        rows.append(
            {"id": rng.choice([None, *ens.instances])}
            | {f"f{i}": v for i, v in enumerate(feats)}
        )
    spec = {"id": "int?"} | {
        f.name: ("int?" if f.type == pa.int64() else "float?") for f in ens.takes
    }
    return table(spec, rows)


def _score_sql(ens: Ensemble) -> str:
    args = ", ".join(f.name for f in ens.takes)
    return f"SELECT {ens.name}(id, {args}) AS p FROM __THIS__"


@pytest.mark.parametrize("seed", range(12))
def test_an_ensemble_agrees_with_the_oracle(seed):
    ens, _ = _ensemble(seed, ("dtr", "rf", "gbr")[seed % 3])
    rows = _score_rows(ens, seed)
    assert_parity(_score_sql(ens), rows, udfs=[ens])
    _own_register(_score_sql(ens), rows, [ens])


def _stump(value: float, link: str) -> Ensemble:
    nodes = pa.table(
        {
            "model_id": pa.array([0], pa.int64()),
            "tree_id": pa.array([0], pa.int64()),
            "node_id": pa.array([0], pa.int64()),
            "feature": pa.array([-1], pa.int32()),
            "threshold": pa.array([0.0]),
            "left": pa.array([-1], pa.int32()),
            "right": pa.array([-1], pa.int32()),
            "missing_left": pa.array([True]),
            "value": pa.array([value]),
        }
    )
    models = pa.table(
        {
            "model_id": pa.array([0], pa.int64()),
            "base": pa.array([0.0]),
            "agg": pa.array(["sum"]),
            "link": pa.array([link]),
        }
    )
    return Ensemble("stump", F, nodes, models, "float64")


def test_a_sigmoid_past_exps_range_is_zero_as_the_kernel_says():
    ens = _stump(-1000.0, "sigmoid")
    assert ens(0, 1.0) == (0.0,)
    assert_parity(
        "SELECT stump(id, x) AS p FROM __THIS__",
        table({"id": "int", "x": "float"}, [{"id": 0, "x": 1.0}]),
        udfs=[ens],
    )


def test_an_id_with_no_model_traps_on_both_engines():
    assert_parity(
        "SELECT stump(id, x) AS p FROM __THIS__",
        table({"id": "int", "x": "float"}, [{"id": 3, "x": 1.0}]),
        udfs=[_stump(1.0, "identity")],
        trap="no model with id 3",
    )


@pytest.mark.parametrize(
    "n",
    [0, 1, -1, 2**24 + 1, 2**24 + 3, 2**53 + 1, -(2**53) - 1, 2**63 - 1, -(2**63)]
    + [random.Random(i).randrange(-(2**63), 2**63) for i in range(200)],  # noqa: S311
)
def test_an_integer_narrows_to_float32_in_one_rounding(n):
    assert _int_to_f32(n) == float(np.array([n], np.int64).astype(np.float32)[0])


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: ExternFunction("f", pa.schema([("x", pa.int32())]), pa.int64(), _double),
         "takes type int32"),
        (lambda: ExternFunction("f", F, pa.list_(pa.float64(), 1), _double),
         "width-1 list"),
        (lambda: ExternFunction("f", F, pa.list_(pa.float64()), _double),
         "must declare its width"),
        (lambda: ExternFunction(
            "f", F, pa.struct([("a", pa.int64()), ("A", pa.int64())]), _double),
         "collide case-insensitively"),
        (lambda: ExternFunction("f", F, pa.float64(), 3), "callable"),
        (lambda: ExternFunction("", F, pa.float64(), _double), "name"),
    ],
)
def test_a_bad_extern_declaration_refuses_at_construction(build, match):
    with pytest.raises(FunctionError, match=match):
        build()


def test_a_bad_ensemble_declaration_refuses_at_construction():
    good = _stump(1.0, "identity")
    n, m = good.nodes, good.models
    with pytest.raises(FunctionError, match="compare_grid"):
        Ensemble("e", F, n, m, "float16")
    with pytest.raises(FunctionError, match="a number"):
        Ensemble("e", pa.schema([("s", pa.string())]), n, m, "float64")
    with pytest.raises(FunctionError, match="dense from 0"):
        Ensemble("e", F, n, m.set_column(0, "model_id", pa.array([1], pa.int64())),
                 "float64")
    with pytest.raises(FunctionError, match="column 'value'"):
        Ensemble("e", F, n.drop_columns(["value"]), m, "float64")
