"""A CASE over `x = <integer constant>` arms lowers to a binary search over
the constants (`lower.rs`, `case_dispatch`) once it has 8 or more arms: what
a native transform's per-instance selection compiles to. It must answer
exactly what the arm-by-arm CASE answers; every case here runs against the
DuckDB oracle on both backends (the campaign verdict).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pyarrow as pa
import pytest
from confit import SqlFunction
from confit import sql as S

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz.parity import assert_parity, table  # noqa: E402

KEYS = [None, -7, -1, 0, 1, 2, 3, 5, 8, 13, 21, 34, 1000, -(2**63), 2**63 - 1]
ROWS = table(
    {"g": "int?", "t": "int8?", "x": "float?", "s": "str?"},
    [
        {
            "g": g,
            "t": None if g is None else max(-128, min(127, g)),
            "x": 0.5 * i,
            "s": f"r{i}",
        }
        for i, g in enumerate(KEYS)
    ],
)


def case(col: str, consts: list[int], arm, default: str | None) -> str:
    arms = " ".join(f"WHEN {col} = {c} THEN {arm(i, c)}" for i, c in enumerate(consts))
    tail = f" ELSE {default}" if default is not None else ""
    return f"CASE {arms}{tail} END"


@pytest.mark.parametrize("seed", range(12))
def test_a_dispatch_agrees_with_the_oracle(seed):
    rng = random.Random(seed)  # noqa: S311
    n = rng.randint(8, 40)
    pool = [k for k in KEYS if k is not None] + [
        rng.randint(-50, 50) for _ in range(30)
    ]
    consts = [rng.choice(pool) for _ in range(n)]  # repeats: the first arm wins
    col = rng.choice(["g", "t"])
    if col == "t":
        consts = [max(-128, min(127, c)) for c in consts]
    arm = rng.choice(
        [
            lambda i, c: f"x * {i + 1}",
            lambda i, c: f"'{i}' || s",
            lambda i, c: f"CASE WHEN x > {i} THEN NULL ELSE x END",
        ]
    )
    default = rng.choice([None, "NULL", {"x": "x", "s": "s"}])
    if isinstance(default, dict):
        default = "x" if "x *" in arm(0, 0) or "CASE" in arm(0, 0) else "s"
    sql = f"SELECT {case(col, consts, arm, default)} AS o FROM __THIS__"
    assert_parity(sql, ROWS)


def test_the_constant_may_stand_on_either_side():
    arms = " ".join(f"WHEN {c} = g THEN {c * 2}" for c in range(10))
    assert_parity(f"SELECT CASE {arms} ELSE -1 END AS o FROM __THIS__", ROWS)


def test_an_instance_dispatch_in_a_sql_function():
    k = 50
    f = SqlFunction(
        "tf",
        pa.schema([("iid", pa.int64()), ("v", pa.float64())]),
        pa.float64(),
        lambda iid, v: _ladder(iid, v, k),
    )
    rows = table(
        {"g": "int?", "x": "float?"}, [{"g": g, "x": 1.5} for g in (None, 0, 7, 49)]
    )
    assert_parity("SELECT tf(g, x) AS o FROM __THIS__", rows, udfs=[f])
    bad = table({"g": "int?", "x": "float?"}, [{"g": 50, "x": 1.5}])
    assert_parity(
        "SELECT tf(g, x) AS o FROM __THIS__", bad, udfs=[f], trap="not fitted"
    )


def _ladder(iid, v, k):
    e = S.case(iid == S.lit(0), v * S.lit(1.0))
    for i in range(1, k):
        e = e.when(iid == S.lit(i), (v - S.lit(float(i))) / S.lit(2.0 + i))
    return e.when(iid.isnull(), S.lit(None, pa.float64())).otherwise(
        S.fn("error", S.lit("instance id not fitted"))
    )
