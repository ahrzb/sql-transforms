"""Verify round4 candidate families with a clean driver (repr, no json).

Re-runs the exact recorded cases for the non-documented families and prints
per-case duck vs confit in Python repr so no serializer masks the diff.
Select families via --family (E, trycast, sign, arrowstruct).

Usage:
  uv run python scripts/fuzz/verify_round4.py --family E --max 10
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "output"))
sys.path.insert(
    0,
    os.path.join(os.path.dirname(__file__), "..", "..", "packages", "confit", "tests"),
)

import duckdb  # noqa: E402
import pyarrow as pa  # noqa: E402
from confit import DuckDBInferFn  # noqa: E402
from differential import Case, row_model, static  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "output")


def duck_run(c: Case):
    con = duckdb.connect()
    con.register("__arrow_this", static(c.schema, c.rows))
    con.execute("CREATE TABLE __THIS__ AS SELECT * FROM __arrow_this")
    for name, table in c.statics.items():
        con.register(f"__arrow_{name}", table)
        con.execute(f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"')
    try:
        return ("rows", con.execute(c.sql).to_arrow_table().to_pylist())
    except BaseException as e:  # noqa: BLE001
        return ("err", str(e).splitlines()[0])
    finally:
        con.close()


def confit_run(c: Case, interp: bool):
    if interp:
        os.environ["SPECIALIZER_FORCE_INTERP"] = "1"
    else:
        os.environ.pop("SPECIALIZER_FORCE_INTERP", None)
    model = row_model(c.schema)
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
        return ("rows", got, getattr(fn, "backend", None))
    except BaseException as e:  # noqa: BLE001
        return ("err", str(e).splitlines()[0], None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--family", required=True, choices=["E", "trycast", "sign", "arrowstruct"]
    )
    ap.add_argument("--max", type=int, default=10)
    args = ap.parse_args()

    seen = set()
    n = 0
    for fn in sorted(os.listdir(OUT)):
        if not (
            fn.startswith("round4-") and "cranelift" in fn and fn.endswith(".jsonl")
        ):
            continue
        with open(os.path.join(OUT, fn), encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                res = rec["result"]
                cls = res["classifier"]
                c = rec["case"]
                key = json.dumps(c, sort_keys=True)
                if key in seen:
                    continue
                seen.add(key)
                if not pick(args.family, cls, c):
                    continue
                cobj = Case(
                    c["surface"],
                    c["sql"],
                    c["schema"],
                    c["rows"],
                    {
                        name: pa.Table.from_pylist(rows)
                        for name, rows in c["statics"].items()
                    },
                    c.get("shape", "filter"),
                )
                duck = duck_run(cobj)
                g = confit_run(cobj, False)
                gi = confit_run(cobj, True)
                print(f"--- {c['surface']} {cls} sql: {c['sql']}")
                print(
                    f"    duck:   {duck[0]}: {repr(duck[1]) if duck[0] == 'rows' else duck[1]}"
                )
                print(
                    f"    confit: {g[0]}: {repr(g[1]) if g[0] == 'rows' else g[1]}  backend={g[2]}"
                )
                print(
                    f"    interp: {gi[0]}: {repr(gi[1]) if gi[0] == 'rows' else gi[1]}  backend={gi[2]}"
                )
                n += 1
                if n >= args.max:
                    return


def pick(fam: str, cls: str, c: dict) -> bool:
    sql = c["sql"]
    if fam == "E":
        return cls == "E_engine_serves_duck_errors"
    if fam == "trycast":
        return cls == "A_rows_differ" and "TRY_CAST" in sql
    if fam == "sign":
        return (
            cls == "A_rows_differ"
            and "TRY_CAST" not in sql
            and (
                ("nan" in repr(c.get("rows", ""))) or c["surface"] in ("arith", "cond")
            )
        )
    if fam == "arrowstruct":
        return cls.startswith("arrow_error")
    return False


if __name__ == "__main__":
    main()
