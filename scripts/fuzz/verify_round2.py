"""Round 2: exact agent reproductions + duckdb self-consistency probes."""

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


B = [{"s": "1"}, {"s": "12.9"}, {"s": "abc"}, {"s": "héllo"}, {"s": "x"}]
check(
    "F4a substr-undershoot-multibyte-batch",
    "SELECT substr(s, -3, 1) AS r FROM __THIS__",
    "s VARCHAR",
    {"s": (str, ...)},
    B,
)

check(
    "F4c substr-bc-direct",
    "SELECT substr(s, -3, 1) AS r FROM __THIS__",
    "s VARCHAR",
    {"s": (str, ...)},
    [{"s": "bc"}],
)

check(
    "F4b substr-fold-again",
    "SELECT substr(substr(s, 2, 2), -3, 1) AS r FROM __THIS__",
    "s VARCHAR",
    {"s": (str, ...)},
    [{"s": "abcd"}],
)

print("--- duckdb self-consistency: nested vs direct ---")
print(
    "direct  substr('bc', -3, 1)      :",
    duck("SELECT substr(s, -3, 1) FROM __THIS__", "s VARCHAR", [("bc",)]),
)
print(
    "nested  substr(substr('abcd',2,2),-3,1):",
    duck(
        "SELECT substr(substr(s, 2, 2), -3, 1) FROM __THIS__", "s VARCHAR", [("abcd",)]
    ),
)
print(
    "nested2 substr(substr('abcd',2,2),-3,1) 3 rows:",
    duck(
        "SELECT substr(substr(s, 2, 2), -3, 1) FROM __THIS__",
        "s VARCHAR",
        [("abcd",), ("xy",), ("wxyz",)],
    ),
)
print(
    "direct  substr('ab', -3, 1)      :",
    duck("SELECT substr(s, -3, 1) FROM __THIS__", "s VARCHAR", [("ab",)]),
)
print()

E = [{"k": -9223372036854775808, "k2": 7}, {"k": 1, "k2": 0}]
check(
    "F5b neg-in-between",
    "SELECT NOT (1 BETWEEN k2 AND ceil(-k)) AS r FROM __THIS__",
    "k BIGINT, k2 BIGINT",
    {"k": (int, ...), "k2": (int, ...)},
    E,
)

check(
    "F5c neg-in-case",
    "SELECT CASE WHEN k2 > 0 THEN -k ELSE 0 END AS r FROM __THIS__",
    "k BIGINT, k2 BIGINT",
    {"k": (int, ...), "k2": (int, ...)},
    E,
)

check(
    "F5d neg-in-pow",
    "SELECT (pow(k, 4) BETWEEN (-f + -f) AND -2 AND (k BETWEEN -k AND k OR k < -f)) AS r FROM __THIS__",
    "k BIGINT, f DOUBLE",
    {"k": (int, ...), "f": (float, ...)},
    [{"k": -9223372036854775808, "f": 1.0}, {"k": 3, "f": 2.0}],
)
