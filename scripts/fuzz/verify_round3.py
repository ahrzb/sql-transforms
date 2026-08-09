"""Round 3: root-cause probes for F1/F2 sign bits and F4 single-row kernel."""

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


def engine(sql, model, rows):
    try:
        fn = DuckDBInferFn(
            sql, row_tables={"__THIS__": model}, static_tables={}, output="dict"
        )
    except ValueError as e:
        return ("build_err", str(e))
    try:
        return (
            "rows",
            [
                tuple(r.values())
                for r in fn.infer({"__THIS__": [model(**r) for r in rows]})
            ],
        )
    except BaseException as e:
        return ("infer_err", type(e).__name__ + ": " + str(e))


def check(name, sql, schema_sql, model_fields, rows):
    model = make_model(**model_fields)
    w = duck(sql, schema_sql, [tuple(r.values()) for r in rows])
    g = engine(sql, model, rows)
    print(
        f"{'OK ' if w == g else 'DIV'} {name}: duck={w}\n    confit={g}\n    sql: {sql}\n"
    )


F = [{"f": 0.0}]
check(
    "P1 runtime 0/0 via col",
    "SELECT CAST((f / f) AS VARCHAR) AS r FROM __THIS__",
    "f DOUBLE",
    {"f": (float, ...)},
    F,
)
check(
    "P2 runtime f-f",
    "SELECT CAST((f - f) AS VARCHAR) AS r FROM __THIS__",
    "f DOUBLE",
    {"f": (float, ...)},
    F,
)
check(
    "P3 neg col sign",
    "SELECT CAST((-f) AS VARCHAR) AS r FROM __THIS__",
    "f DOUBLE",
    {"f": (float, ...)},
    F,
)
check(
    "P4 mul by -1",
    "SELECT CAST((f * -1.0) AS VARCHAR) AS r FROM __THIS__",
    "f DOUBLE",
    {"f": (float, ...)},
    F,
)
check(
    "P5 pow runtime",
    "SELECT CAST(pow(0.0 - f, -3.0) AS VARCHAR) AS r FROM __THIS__",
    "f DOUBLE",
    {"f": (float, ...)},
    F,
)
check(
    "P6 pow neg-col",
    "SELECT CAST(pow(-f, -3.0) AS VARCHAR) AS r, CAST(-f AS VARCHAR) AS n FROM __THIS__",
    "f DOUBLE",
    {"f": (float, ...)},
    F,
)

S = [{"s": "ab"}]
check(
    "P7 substr -5,2 single",
    "SELECT substr(s, -5, 2) AS r FROM __THIS__",
    "s VARCHAR",
    {"s": (str, ...)},
    S,
)
S2 = [{"s": "abcde"}]
check(
    "P8 substr -7,2 single",
    "SELECT substr(s, -7, 2) AS r FROM __THIS__",
    "s VARCHAR",
    {"s": (str, ...)},
    S2,
)
check(
    "P9 substr -7,2 len5 exp",
    "SELECT substr(s, -7, 2) AS r FROM __THIS__",
    "s VARCHAR",
    {"s": (str, ...)},
    S,
)

K = [{"k": 3}, {"k": 4}]
check(
    "P10 fold pow -7",
    "SELECT CAST(pow(2, -3) AS VARCHAR) AS r FROM __THIS__",
    "k BIGINT",
    {"k": (int, ...)},
    K,
)
check(
    "P11 fold div",
    "SELECT CAST((1.0 / 0.0) AS VARCHAR) AS r FROM __THIS__",
    "k BIGINT",
    {"k": (int, ...)},
    K,
)
check(
    "P12 fold nan-2",
    "SELECT CAST((0.0 * 0.0 / 0.0) AS VARCHAR) AS r FROM __THIS__",
    "k BIGINT",
    {"k": (int, ...)},
    K,
)
