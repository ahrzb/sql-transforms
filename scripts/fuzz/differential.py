"""Differential fuzz harness: confit (DuckDBInferFn) vs duckdb 1.5.5.

Contract under test (docs/known-limitations.md): for any SQL the engine
either serves bit-for-bit identical to DuckDB, or refuses loudly at build
time. There is no third mode. A candidate = a contract breach, classes:

  A  duck ok, engine serves, rows differ          (silent wrong answer)
  C  duck ok, engine traps at inference            (false execution error)
  D  duck ok, engine hangs / would never emit      (non-termination)
  E  duck errors, engine serves rows               (should have refused)
  G  duck ok, engine panics / kills the process    (crash)

Known, documented divergences are NOT candidates (filtered here):
  * decimal literal -> f64 mapping (duck types literals DECIMAL)
  * int-literal output schema width in arrow (TASK-79, values agree)
  * NULL||NULL typed VARCHAR (values agree)
  * row order for shape='many' (multiset is the contract)
  * duplicate output names renamed (duck .df() renames identically)
  * trap/error TEXT wording (class matters, not prose)

Usage:
  python scripts/fuzz/differential.py --surface arith --seed 42 --n 6000
      --out scripts/fuzz/output/candidates.jsonl
  Surfaces: arith cast strings struct cond joins arrow
  Env: SPECIALIZER_FORCE_INTERP=1 forces the interpreter backend.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb
import pyarrow as pa
from pydantic import create_model

sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "packages", "confit", "tests"),
)

from confit import DuckDBInferFn  # noqa: E402

_PY = {"int": int, "float": float, "str": str, "bool": bool}
_ARROW = {
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
    "bool": pa.bool_(),
}


def row_model(schema: dict[str, str | pa.StructType | tuple]):
    fields = {}
    for name, spec in schema.items():
        if isinstance(spec, tuple):
            t, nullable = spec
            fields[name] = (
                (struct_model("Struct", t) | None)
                if nullable
                else struct_model("Struct", t),
                None if nullable else ...,
            )
        elif isinstance(spec, pa.StructType):
            fields[name] = (struct_model("Struct", spec), None)
        elif spec.endswith("?"):
            fields[name] = (_PY[spec[:-1]] | None, None)
        else:
            fields[name] = (_PY[spec], ...)
    return create_model("Row", **fields)


def static(
    schema: dict[str, str | pa.StructType | tuple], rows: list[dict[str, Any]]
) -> pa.Table:
    def field_spec(n, s):
        if isinstance(s, tuple):
            t, nullable = s
            return pa.field(n, t, nullable=nullable)
        if isinstance(s, pa.StructType):
            return pa.field(n, s, nullable=True)
        return pa.field(n, _ARROW[s.rstrip("?")], nullable=s.endswith("?"))

    arrow = pa.schema(field_spec(n, s) for n, s in schema.items())
    return pa.Table.from_pylist(rows, schema=arrow)


@dataclass
class Case:
    surface: str
    sql: str
    schema: dict[str, str]
    rows: list[dict[str, Any]]
    statics: dict[str, pa.Table] = field(default_factory=dict)
    shape: str = "filter"


@dataclass
class Result:
    duck_ok: bool
    duck_rows: list | None
    duck_err: str | None
    confit_build_ok: bool
    confit_err: str | None
    confit_rows: list | None
    backend: str | None
    classifier: str
    detail: str = ""


def _rows_key(rows: list) -> list:
    return sorted(sorted((k, repr(v)) for k, v in r.items()) for r in rows)


def run_case(c: Case) -> Result:
    model = row_model(c.schema)
    con = duckdb.connect()
    con.register("__arrow_this", static(c.schema, c.rows))
    con.execute("CREATE TABLE __THIS__ AS SELECT * FROM __arrow_this")
    for name, table in c.statics.items():
        con.register(f"__arrow_{name}", table)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')
    try:
        want = con.execute(c.sql).to_arrow_table().to_pylist()
        duck_ok, duck_rows, duck_err = True, want, None
    except BaseException as e:
        duck_ok, duck_rows, duck_err = False, None, str(e)
    finally:
        con.close()

    try:
        shape_arg = {} if c.shape == "filter" else {"shape": c.shape}
        fn = DuckDBInferFn(
            c.sql,
            row_tables={"__THIS__": model},
            static_tables=c.statics,
            output="dict",
            **shape_arg,
        )
        got = fn.infer({"__THIS__": [model(**r) for r in c.rows]})
        confit_build_ok, confit_err, confit_rows = True, None, got
    except BaseException as e:
        confit_build_ok, confit_err, confit_rows = False, str(e), None

    # classifier
    if not confit_build_ok:
        cls = "reject"  # build-time refusal: fine by contract
    elif not duck_ok:
        cls = "E_engine_serves_duck_errors"
    else:
        try:
            match = _rows_key(confit_rows) == _rows_key(duck_rows)
        except BaseException:
            match = False
        cls = "match" if match else "A_rows_differ"

    return Result(
        duck_ok,
        duck_rows,
        duck_err,
        confit_build_ok,
        confit_err,
        confit_rows,
        getattr(fn, "backend", None) if confit_build_ok else None,
        cls,
    )


def dump_case(c: Case) -> dict:
    return {
        "surface": c.surface,
        "sql": c.sql,
        "schema": c.schema,
        "rows": c.rows,
        "statics": {name: table.to_pylist() for name, table in c.statics.items()},
        "shape": c.shape,
    }


# ----------------------------------------------------------------- generators

INT_EDGES = [
    0,
    1,
    -1,
    2,
    7,
    9223372036854775807,
    -9223372036854775808,
    2147483648,
    -2147483649,
    10**18,
    2**53,
    2**63 - 1,
    3,
    -7,
    64,
]
FLOAT_EDGES = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    2.5,
    3.141592653589793,
    0.1,
    1e-9,
    float("inf"),
    float("-inf"),
    1e308,
    5e-324,
    123456789012345678.0,
    1e16,
    2.5e-10,
]
STR_EDGES = [
    "",
    "a",
    "abc",
    "héllo",
    "٣٤",
    "ß",
    "K",
    "HELLO",
    "x&&y--z[a]",
    "a\nb\tc",
    "   ",
    "12.9",
    "123",
    "true",
    "TRUE",
    "1",
    "abcABC",
]
BOOL_EDGES = [True, False]


def rng_pick(rng, seq):
    return seq[rng.randrange(len(seq))]


def gen_rows(rng, schema, n=8):
    rows = []
    for _ in range(n):
        row = {}
        for name, spec in schema.items():
            t = spec.rstrip("?")
            if spec.endswith("?") and rng.random() < 0.3:
                row[name] = None
                continue
            if t == "int":
                row[name] = (
                    rng_pick(rng, INT_EDGES)
                    if rng.random() < 0.8
                    else rng.randint(-100, 100)
                )
            elif t == "float":
                row[name] = (
                    rng_pick(rng, FLOAT_EDGES)
                    if rng.random() < 0.7
                    else rng.uniform(-1e6, 1e6)
                )
            elif t == "str":
                row[name] = rng_pick(rng, STR_EDGES)
            else:
                row[name] = rng_pick(rng, BOOL_EDGES)
        rows.append(row)
    return rows


def gen_arith_expr(rng, depth=0):
    r = rng.random()
    if depth > 2:
        return rng_pick(rng, ["k", "f", "-k", "-f"])
    if r < 0.45:
        op = rng_pick(rng, ["+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "pow"])
        if op == "pow":
            return f"pow({gen_arith_expr(rng, depth + 1)}, {rng.randint(-3, 4) or 2})"
        return (
            f"({gen_arith_expr(rng, depth + 1)} {op} {gen_arith_expr(rng, depth + 1)})"
        )
    if r < 0.6:
        f = rng_pick(
            rng,
            [
                "abs",
                "sqrt",
                "floor",
                "ceil",
                "round",
                "sign",
                "exp",
                "log10",
                "log2",
                "trunc",
                "cbrt",
            ],
        )
        if f == "round" and rng.random() < 0.5:
            return f"round({gen_arith_expr(rng, depth + 1)}, {rng.randint(-3, 3)})"
        return f"{f}({gen_arith_expr(rng, depth + 1)})"
    if r < 0.7:
        return rng_pick(rng, ["k", "f", "k2", "d"])
    if r < 0.85:
        return f"CASE WHEN {gen_bool_expr(rng, depth + 1)} THEN {gen_arith_expr(rng, depth + 1)} ELSE {gen_arith_expr(rng, depth + 1)} END"
    return str(rng.randint(-5, 5))


def gen_bool_expr(rng, depth=0):
    r = rng.random()
    if depth > 2:
        return rng_pick(rng, ["k > 0", "k = 0", "f < 1.5", "b", "k IS NULL"])
    if r < 0.35:
        op = rng_pick(rng, ["AND", "OR"])
        return f"({gen_bool_expr(rng, depth + 1)} {op} {gen_bool_expr(rng, depth + 1)})"
    if r < 0.5:
        return f"NOT ({gen_bool_expr(rng, depth + 1)})"
    if r < 0.62:
        return f"{gen_arith_expr(rng, depth + 1)} {rng_pick(rng, ['=', '<>', '<', '<=', '>', '>='])} {gen_arith_expr(rng, depth + 1)}"
    if r < 0.72:
        return f"{rng_pick(rng, ['k', 'f', 'd', 's'])} IS {'NOT ' if rng.random() < 0.5 else ''}NULL"
    if r < 0.85:
        return f"{gen_arith_expr(rng, depth + 1)} BETWEEN {gen_arith_expr(rng, depth + 1)} AND {gen_arith_expr(rng, depth + 1)}"
    return rng_pick(rng, ["TRUE", "FALSE", "b"])


def case_arith(rng, sql_pool, _) -> Case:
    schema = {"k": "int?", "k2": "int?", "f": "float?", "d": "float?", "b": "bool?"}
    shape = "filter"
    r = rng.random()
    if r < 0.35:
        expr = gen_arith_expr(rng)
        where = f" WHERE {gen_bool_expr(rng)}" if rng.random() < 0.5 else ""
        sql = f"SELECT {expr} AS r FROM __THIS__{where}"
    elif r < 0.6:
        sql = f"SELECT {gen_bool_expr(rng)} AS r, k FROM __THIS__"
    elif r < 0.8:
        sql = f"SELECT {gen_arith_expr(rng)} AS r, {gen_arith_expr(rng)} AS s FROM __THIS__ WHERE {gen_bool_expr(rng)}"
    else:
        expr = f"CAST({gen_arith_expr(rng)} AS {rng_pick(rng, ['BIGINT', 'DOUBLE', 'VARCHAR', 'BOOLEAN'])}) AS r"
        t = "TRY_CAST" if rng.random() < 0.4 else "CAST"
        expr = f"{t}({gen_arith_expr(rng)} AS {rng_pick(rng, ['BIGINT', 'DOUBLE', 'VARCHAR', 'BOOLEAN'])}) AS r"
        sql = f"SELECT {expr} FROM __THIS__"
    return Case("arith", sql, schema, gen_rows(rng, schema), shape=shape)


def case_cast(rng, _p, _) -> Case:
    schema = {"k": "int?", "f": "float?", "s": "str?", "b": "bool?"}
    src = rng_pick(rng, ["k", "f", "s", "b"])
    tgt = rng_pick(rng, ["BIGINT", "DOUBLE", "VARCHAR", "BOOLEAN"])
    form = rng_pick(rng, ["CAST", "TRY_CAST", "::"])
    if form == "::":
        expr = f"{src}::{tgt}"
    else:
        expr = f"{form}({src} AS {tgt})"
    sql = f"SELECT {expr} AS r FROM __THIS__"
    return Case("cast", sql, schema, gen_rows(rng, schema))


def gen_str_expr(rng, depth=0):
    r = rng.random()
    if depth > 2:
        return rng_pick(rng, ["s", "s2"])
    if r < 0.4:
        f = rng_pick(
            rng,
            [
                "upper",
                "lower",
                "reverse",
                "trim",
                "ltrim",
                "rtrim",
                "length",
                "char_length",
            ],
        )
        args = f"({gen_str_expr(rng, depth + 1)})"
        if f in ("trim", "ltrim", "rtrim") and rng.random() < 0.5:
            args = f"({gen_str_expr(rng, depth + 1)}, '{rng_pick(rng, [' ', 'h', 'é', 'x'])}')"
        return f"{f}{args}"
    if r < 0.55:
        return (
            f"substr({gen_str_expr(rng, depth + 1)}, {rng.randint(-3, 6) or 1}, {rng.randint(0, 4)})"
            if rng.random() < 0.5
            else f"substr({gen_str_expr(rng, depth + 1)}, {rng.randint(-3, 6) or 1})"
        )
    if r < 0.65:
        return f"{gen_str_expr(rng, depth + 1)} || {rng_pick(rng, ['k', "'xy'", "'é'", 's2'])}"
    if r < 0.75:
        return f"replace({gen_str_expr(rng, depth + 1)}, {rng_pick(rng, ["'a'", "'x&&y--z'", "''", "'é'"])}, {rng_pick(rng, ["'b'", "''"])})"
    if r < 0.85:
        return f"split_part({gen_str_expr(rng, depth + 1)}, {rng_pick(rng, ["'a'", "','", "'&&'"])}, {rng.randint(0, 4)})"
    return f"concat({gen_str_expr(rng, depth + 1)}, {rng_pick(rng, ["'::'", 's2', "'٣٤'"])}, {rng_pick(rng, ['k', "'Z'"])})"


def gen_str_pred(rng, depth=0):
    r = rng.random()
    if r < 0.35:
        op = rng_pick(rng, ["LIKE", "NOT LIKE", "ILIKE", "GLOB"])
        if op == "GLOB":
            pat = rng_pick(rng, ["*ell*", "h?llo", "*", "?", "[a-c]*", "\\*"])
        else:
            pat = rng_pick(
                rng,
                ["%ell%", "h_llo", "%", "_", "a%c", "\\%", "%o%w%", "%ß%", "hé%llo"],
            )
        return f"{gen_str_expr(rng, depth + 1)} {op} '{pat}'"
    if r < 0.6:
        f = rng_pick(rng, ["starts_with", "ends_with", "contains"])
        return f"{f}({gen_str_expr(rng, depth + 1)}, {rng_pick(rng, ["'ell'", "'h'", "''", "'٣٤'"])})"
    if r < 0.8:
        return f"regexp_matches({gen_str_expr(rng, depth + 1)}, {rng_pick(rng, ["'a.c'", "'^h.*o$'", "'[0-9]+'", "'(?:x|y)&&'", "'(?i)HELLO'", "'.*[a-z]\\d.*'", "'^$'", "'é+'", "'\\\\d{2,3}'", "'.'"])})"
    return f"{gen_str_expr(rng, depth + 1)} {rng_pick(rng, ['=', '<>', 'LIKE'])} {rng_pick(rng, ["'abc'", "'héllo'", "''", 's'])}"


def case_strings(rng, _p, _) -> Case:
    schema = {"s": "str?", "s2": "str?", "k": "int?"}
    sql = (
        f"SELECT {gen_str_expr(rng)} AS r FROM __THIS__ WHERE {gen_str_pred(rng)}"
        if rng.random() < 0.6
        else f"SELECT {gen_str_expr(rng)} AS r, {gen_str_pred(rng)} AS w FROM __THIS__"
    )
    return Case("strings", sql, schema, gen_rows(rng, schema))


_SCALAR_PA = {
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
    "bool": pa.bool_(),
}


def struct_rows(rng, schema):
    rows = []
    for _ in range(8):
        row = {}
        for name, spec in schema.items():
            if isinstance(spec, tuple):
                row[name] = _gen_struct_val(rng, spec[0])
            elif isinstance(spec, pa.StructType):
                row[name] = _gen_struct_val(rng, spec)
            else:
                row[name] = rng_pick(rng, INT_EDGES) if rng.random() < 0.8 else None
        rows.append(row)
    return rows


def _gen_struct_val(rng, t):
    if rng.random() < 0.15:
        return None
    return _gen_struct_val_forced(rng, t)


def _gen_struct_val_forced(rng, t):
    if isinstance(t, pa.StructType):
        val = {}
        for f in t:
            if pa.types.is_struct(f.type):
                if rng.random() < 0.15:
                    val[f.name] = None
                else:
                    val[f.name] = _gen_struct_val_forced(rng, f.type)
            else:
                if f.nullable and rng.random() < 0.25:
                    val[f.name] = None
                else:
                    base = (
                        "str"
                        if pa.types.is_string(f.type)
                        else (
                            "float"
                            if pa.types.is_floating(f.type)
                            else ("bool" if pa.types.is_boolean(f.type) else "int")
                        )
                    )
                    val[f.name] = {
                        "int": lambda: rng_pick(rng, INT_EDGES),
                        "float": lambda: rng_pick(rng, FLOAT_EDGES),
                        "str": lambda: rng_pick(rng, STR_EDGES),
                        "bool": lambda: rng_pick(rng, BOOL_EDGES),
                    }[base]()
        return val
    return None


def struct_model(name, t: pa.StructType):
    fields = {}
    for f in t:
        if pa.types.is_struct(f.type):
            fields[f.name] = (
                struct_model(f"{name}_{f.name}", f.type),
                None if f.nullable else ...,
            )
        else:
            base = (
                "str"
                if pa.types.is_string(f.type)
                else (
                    "float"
                    if pa.types.is_floating(f.type)
                    else ("bool" if pa.types.is_boolean(f.type) else "int")
                )
            )
            py_t = _PY[base]
            fields[f.name] = (
                (py_t | None) if f.nullable else py_t,
                None if f.nullable else ...,
            )
    return create_model(name, **fields)


def _mk_struct(fields_spec: str, nullable=False) -> pa.StructType:
    """'i:int?, b:struct{c:int}, f:float?' -> pa.StructType (depth-safe split)."""
    pa_fields = []
    for name, spec in _split_fields(fields_spec):
        spec = spec.strip()
        if spec.startswith("struct"):
            m = spec.endswith("}?")
            body = spec[7:-2] if m else spec[7:-1]
            pa_fields.append(pa.field(name, _mk_struct(body), nullable=m))
        else:
            nullable = spec.endswith("?")
            pa_fields.append(
                pa.field(name, _SCALAR_PA[spec.rstrip("?")], nullable=nullable)
            )
    return pa.struct(pa_fields)


def _split_fields(s: str) -> list[tuple[str, str]]:
    out = []
    depth = 0
    cur = ""
    for ch in s:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur)
    return [tuple(p.split(":", 1)) for p in out]


STRUCT_SCHEMAS = [
    {"t": _mk_struct("i:int?, f:float?, s:str?")},
    {"t": (_mk_struct("a:int?, b:struct{c:int?, d:str?}"), True), "k": "int?"},
    {"t": _mk_struct("i:int, f:float, s:str?"), "k": "int?"},
    {"t": _mk_struct("n:struct{x:float?}"), "k": "int?"},
    {"t": _mk_struct("s:str?, g:struct{h:int?}"), "k": "int?"},
]


def gen_struct_expr(rng, depth=0):
    r = rng.random()
    paths = [
        "t.i",
        "t.f",
        "t.s",
        "t.a",
        "t.b.c",
        "t.b.d",
        "t.n.x",
        "t.i + 1",
        "COALESCE(t.i, 0)",
        "CASE WHEN t.f IS NULL THEN 0 ELSE 1 END",
        "t.s || 'x'",
        "k",
        "t.i - t.i",
        "LENGTH(t.s)",
    ]
    if r < 0.7:
        return rng_pick(rng, paths)
    if r < 0.85:
        return f"{rng_pick(rng, ['t.i', 't.f', 't.s', 't.a', 't.b.c', 'k'])} IS {'NOT ' if rng.random() < 0.5 else ''}NULL"
    return rng_pick(rng, ["t.i > 0 AND t.f < 1.5", "t.s LIKE '%a%'", "t.b.c = 1"])


def case_struct(rng, _p, _) -> Case:
    schema = rng_pick(rng, STRUCT_SCHEMAS)
    r = rng.random()
    if r < 0.3:
        sql = "SELECT t.* FROM __THIS__"
    elif r < 0.45:
        exc = rng_pick(rng, ["t.i", "t.f", "t.s", "t.b.c"])
        sql = f"SELECT t.* EXCLUDE ({exc}) FROM __THIS__"
    elif r < 0.6:
        sql = "SELECT t.* REPLACE (t.i + 1 AS i) FROM __THIS__"
    elif r < 0.8:
        sql = "SELECT *, t.* FROM __THIS__"
    else:
        sql = f"SELECT {gen_struct_expr(rng)} AS r FROM __THIS__ WHERE {gen_struct_expr(rng)}"
    return Case("struct", sql, schema, struct_rows(rng, schema))


def gen_cond_expr(rng, depth=0):
    r = rng.random()
    if depth > 2:
        return rng_pick(rng, ["k > 0", "s IS NULL", "b", "f >= 1.0", "s = 'abc'"])
    if r < 0.25:
        return f"CASE WHEN {gen_cond_expr(rng, depth + 1)} THEN {gen_cond_expr(rng, depth + 1)} ELSE {gen_cond_expr(rng, depth + 1)} END"
    if r < 0.4:
        return f"COALESCE({gen_cond_expr(rng, depth + 1)}, {gen_cond_expr(rng, depth + 1)})"
    if r < 0.5:
        return (
            f"NULLIF({gen_cond_expr(rng, depth + 1)}, {gen_cond_expr(rng, depth + 1)})"
        )
    if r < 0.65:
        return f"IF({gen_bool_expr(rng, depth + 1)}, {rng_pick(rng, ['1', '0', 'k', "'x'", 's'])}, {rng_pick(rng, ['1', '0', 'k', "'x'", 's'])})"
    if r < 0.8:
        return f"GREATEST({gen_cond_expr(rng, depth + 1)}, {gen_cond_expr(rng, depth + 1)}, {rng_pick(rng, ['k', 'f', '0'])})"
    return rng_pick(rng, ["k", "f", "s", "b", "1.5", "NULL"])


def case_cond(rng, _p, _) -> Case:
    schema = {"k": "int?", "f": "float?", "s": "str?", "b": "bool?"}
    sql = (
        f"SELECT {gen_cond_expr(rng)} AS r FROM __THIS__ WHERE {gen_bool_expr(rng)}"
        if rng.random() < 0.5
        else f"SELECT {gen_cond_expr(rng)} AS r, k FROM __THIS__"
    )
    return Case("cond", sql, schema, gen_rows(rng, schema))


DIMS = {
    "dim": static(
        {"id": "int", "name": "str?", "v": "float?"},
        [
            {"id": 1, "name": "one", "v": 1.5},
            {"id": 3, "name": None, "v": None},
            {"id": 7, "name": "seven", "v": -2.0},
            {"id": -1, "name": "neg", "v": 0.0},
            {"id": 100, "name": "", "v": 1e16},
        ],
    ),
    "dimnull": static(
        {"id": "int?", "name": "str?"},
        [
            {"id": 1, "name": "one"},
            {"id": None, "name": "nul"},
            {"id": 3, "name": None},
            {"id": -4, "name": "m4"},
        ],
    ),
    "dimstr": static(
        {"code": "str?", "tag": "str?"},
        [
            {"code": "a", "tag": "A"},
            {"code": "é", "tag": "E"},
            {"code": "", "tag": "EMPTY"},
            {"code": "٣٤", "tag": "AR"},
            {"code": None, "tag": "N"},
        ],
    ),
    "dim2": static(
        {"a": "int?", "b": "int?", "lab": "str?"},
        [
            {"a": 1, "b": 2, "lab": "x"},
            {"a": 1, "b": 9, "lab": "y"},
            {"a": None, "b": 2, "lab": "n"},
            {"a": 5, "b": 5, "lab": "z"},
        ],
    ),
}


def case_joins(rng, _p, _) -> Case:
    schema = {"k": "int?", "code": "str?", "f": "float?"}
    combos = [
        ("INNER", "dim"),
        ("LEFT", "dim"),
        ("INNER", "dimnull"),
        ("LEFT", "dimnull"),
        ("INNER", "dimstr"),
        ("LEFT", "dimstr"),
        ("RIGHT", "dimnull"),
    ]
    join_kind, dim_name = rng_pick(rng, combos)
    key = "k" if dim_name in ("dim", "dimnull") else "code"
    dkey = "id" if dim_name in ("dim", "dimnull") else "code"
    on = f"__THIS__.{key} = {dkey}"
    if rng.random() < 0.3:
        extra = rng_pick(
            rng,
            [
                f"{dkey} > 0",
                "v IS NOT NULL",
                "tag IS NULL",
                f"__THIS__.{key} = 1",
                f"{dkey} <> -1",
                "TRUE",
            ],
        )
        on = f"{on} AND {extra}"
    cols = rng_pick(
        rng,
        [
            "SELECT k, code, name, v, tag",
            "SELECT *",
            "SELECT k, name, v",
            "SELECT code, tag, f",
        ],
    )
    if rng.random() < 0.3:
        on = f"{dkey} = __THIS__.{key} + {rng.randint(0, 2)}"
    sql = f"{cols} FROM __THIS__ {join_kind} JOIN {dim_name} AS d ON {on}"
    return Case(
        "joins",
        sql,
        schema,
        gen_rows(rng, schema),
        statics={
            "dim": DIMS["dim"],
            "dimnull": DIMS["dimnull"],
            "dimstr": DIMS["dimstr"],
            "dim2": DIMS["dim2"],
        },
    )


def case_arrow(rng, _p, _) -> Case:
    gen = rng_pick(
        rng, [case_arith, case_cast, case_strings, case_struct, case_cond, case_joins]
    )
    c = gen(rng, None, None)
    c.surface = "arrow"
    return c


def run_arrow(c: Case, res: Result) -> str:
    """Value + schema parity on infer_arrow vs duckdb arrow, plus infer/infer_arrow
    cross-equality. Returns classifier override."""
    if not res.confit_build_ok or not res.duck_ok:
        return res.classifier
    model = row_model(c.schema)
    try:
        fn = DuckDBInferFn(
            c.sql,
            row_tables={"__THIS__": model},
            static_tables=c.statics,
            output="dict",
        )
        input_table = static(c.schema, c.rows)
        got = fn.infer_arrow(input_table)
        con = duckdb.connect()
        con.register("__arrow_this", static(c.schema, c.rows))
        con.execute("CREATE TABLE __THIS__ AS SELECT * FROM __arrow_this")
        for name, table in c.statics.items():
            con.register(f"__arrow_{name}", table)
            con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')
        want = con.execute(c.sql).to_arrow_table()
    except BaseException as e:
        return f"arrow_error:{e}"
    got_rows, want_rows = got.to_pylist(), want.to_pylist()
    if _rows_key(got_rows) != _rows_key(want_rows):
        return "A_arrow_rows_differ"
    if got.schema != want.schema:
        return "S_arrow_schema_diff"
    return "match"


SURFACES = {
    "arith": case_arith,
    "cast": case_cast,
    "strings": case_strings,
    "struct": case_struct,
    "cond": case_cond,
    "joins": case_joins,
    "arrow": case_arrow,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--surface", required=True, choices=sorted(SURFACES))
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--out", default="scripts/fuzz/output/candidates.jsonl")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    gen = SURFACES[args.surface]
    stats: Counter = Counter()
    candidates = []
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    for i in range(args.n):
        c = gen(rng, None, None)
        res = run_case(c)
        if args.surface == "arrow" and res.classifier in (
            "match",
            "E_engine_serves_duck_errors",
        ):
            res.classifier = run_arrow(c, res)
        stats[res.classifier] += 1
        if res.classifier not in ("match", "reject"):
            rec = {
                "case": dump_case(c),
                "result": asdict(res),
                "seed": args.seed,
                "i": i,
            }
            candidates.append(rec)
            print(json.dumps(rec, ensure_ascii=False), file=sys.stderr)

    print(
        f"[{args.surface}] seed={args.seed} n={args.n}: {dict(stats)}", file=sys.stderr
    )
    with open(args.out, "a", encoding="utf-8") as f:
        for rec in candidates:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
