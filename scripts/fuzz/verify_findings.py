"""Independent verification of 6 candidate findings (fresh constructions)."""

import os
import sys

import duckdb
from pydantic import create_model

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "packages", "confit", "tests"),
)
from confit import DuckDBInferFn  # noqa: E402


def make_model(**fields):
    return create_model("Row", **fields)


def duck(sql, schema_sql, rows):
    con = duckdb.connect()
    con.execute(f"CREATE TABLE __THIS__ ({schema_sql})")
    for r in rows:
        con.execute(
            f"INSERT INTO __THIS__ VALUES ({', '.join(['?'] * len(r))})", list(r)
        )
    try:
        return ("rows", con.execute(sql).fetchall())
    except BaseException as e:
        return ("err", str(e).splitlines()[0])
    finally:
        con.close()


def engine(sql, model, rows, interp=False):
    if interp:
        os.environ["SPECIALIZER_FORCE_INTERP"] = "1"
    else:
        os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
    try:
        fn = DuckDBInferFn(
            sql, row_tables={"__THIS__": model}, static_tables={}, output="dict"
        )
    except ValueError as e:
        return ("build_err", str(e))
    try:
        out = [
            tuple(r.values())
            for r in fn.infer({"__THIS__": [model(**r) for r in rows]})
        ]
        return ("rows", out)
    except BaseException as e:
        return ("infer_err", type(e).__name__ + ": " + str(e))


def check(name, sql, schema_sql, model_fields, rows):
    model = make_model(**model_fields)
    w = duck(sql, schema_sql, [tuple(r.values()) for r in rows])
    g = engine(sql, model, rows)
    gi = engine(sql, model, rows, interp=True)
    ok = w == g
    print(f"{'OK ' if ok else 'DIV'} {name}: duck={w} confit={g} interp={gi}")
    print(f"    sql: {sql}")


R = [{"k": 1}, {"k": 2}]
check(
    "F1 nan-sign",
    "SELECT CAST((0.0 / 0.0) AS VARCHAR) AS r FROM __THIS__",
    "k BIGINT",
    {"k": (int, ...)},
    R,
)

V = [{"f": 0.0}]
check(
    "F2 pow-neg-zero",
    "SELECT CAST(pow(-f, -3.0) AS VARCHAR) AS r FROM __THIS__",
    "f DOUBLE",
    {"f": (float, ...)},
    V,
)

S = [{"s": "12.9"}, {"s": "13"}, {"s": "abc"}, {"s": "-3.7"}, {"s": "0.5"}]
check(
    "F3 trycast-frac",
    "SELECT TRY_CAST(s AS BIGINT) AS r FROM __THIS__",
    "s VARCHAR",
    {"s": (str, ...)},
    S,
)

T = [{"s": "1"}, {"s": "ab"}]
check(
    "F4 substr-undershoot",
    "SELECT substr(s, -3, 1) AS r FROM __THIS__",
    "s VARCHAR",
    {"s": (str, ...)},
    T,
)

U = [{"s": "abcd"}]
check(
    "F4b substr-fold",
    "SELECT substr(substr(s, 2, 2), -3, 1) AS r FROM __THIS__",
    "s VARCHAR",
    {"s": (str, ...)},
    U,
)

N = [{"k": -9223372036854775808}, {"k": 1}]
check(
    "F5 neg-overflow", "SELECT -k AS r FROM __THIS__", "k BIGINT", {"k": (int, ...)}, N
)

L = [{"k": -1}, {"k": 2}]
check(
    "F6 and-eager",
    "SELECT (k > 0 AND log2(k) > 0) AS r FROM __THIS__",
    "k BIGINT",
    {"k": (int, ...)},
    L,
)

C = [{"k": 7, "k2": 0}, {"k": 1, "k2": 0}]
check(
    "F5b neg-bewteen",
    "SELECT NOT (1 BETWEEN k2 AND ceil(-k)) AS r FROM __THIS__",
    "k BIGINT, k2 BIGINT",
    {"k": (int, ...), "k2": (int, ...)},
    C,
)

P = [{"k": 2, "f": 1.0}]
check(
    "F2b pow-odd-two",
    "SELECT CAST(pow(-f, -3.0) AS VARCHAR) AS r FROM __THIS__",
    "f DOUBLE",
    {"f": (float, ...)},
    V,
)
