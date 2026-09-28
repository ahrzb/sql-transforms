"""Probe queries against DuckDB from the command line, with the campaign's verdict.

This replaces throwaway comparison scripts. Each query gets the verdict that
`fuzz.parity` gives a test, plus what each side answered:

    uv run python -m fuzz.probe "SELECT a + 1 AS b FROM __THIS__" \\
        --table '__THIS__={"a": [1, null, 9223372036854775807]}'
    uv run python -m fuzz.probe -f queries.sql --table ... --table 'd=...'

`--table NAME=JSON` takes a column dict. Types are inferred by Arrow: an
integer column is int64, a float column is double, and so on. To pin a type,
append `::type` to the name, with one type per column in order and the
storage names of the campaign vocabulary:
`--table '__THIS__::int32,string={"a": [1], "s": ["x"]}'`. `__THIS__` is the
request table; every other name is a static table. `-f` reads queries
separated by `;` at a line end. The exit status is 1 if any verdict is
outside `--expect` (default: agreement).
"""

from __future__ import annotations

import argparse
import json
import sys

import pyarrow as pa

from fuzz import oracle as O
from fuzz import parity


def _table(arg: str) -> tuple[str, pa.Table]:
    name, _, body = arg.partition("=")
    name, _, types = name.partition("::")
    cols = json.loads(body)
    if types:
        tys = [O._ARROW[t.strip()] for t in types.split(",")]
        if len(tys) != len(cols):
            raise SystemExit(f"{name}: {len(tys)} types for {len(cols)} columns")
        t = pa.table(
            {c: pa.array(v, ty) for (c, v), ty in zip(cols.items(), tys, strict=True)}
        )
    else:
        t = pa.table(cols)
    return name.strip(), t


def _answers(c) -> tuple[str, str]:
    """What confit and DuckDB (optimizer off) each did, one line apiece."""
    table = O._arrow_table(c.row_schema, c.rows)
    statics = {n: O._arrow_table(s, r) for n, (s, r) in c.statics.items()}
    try:
        fn = O._build(c.sql, table.schema, statics, [], c.shape, False)
        try:
            out = fn.infer_arrow(table)
            ours = f"{out.schema.types} {out.to_pylist()}"
        except Exception as e:  # noqa: BLE001
            ours = f"traps: {type(e).__name__}: {e}"
    except Exception as e:  # noqa: BLE001
        ours = f"refuses: {type(e).__name__}: {e}"
    got, phase, err = O._duck_run(c.sql, c, [])[0]
    if got is None:
        duck = f"{'rejects' if phase == 'build' else 'traps'}: {err}"
    else:
        duck = f"{got.schema.types} {got.to_pylist()}"
    return ours, duck


def main(argv=None) -> int:
    # The campaign watchdog's phase markers are noise here.
    phase, O._phase = O._phase, lambda _name: None
    try:
        return _main(argv)
    finally:
        O._phase = phase


def _main(argv) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("sql", nargs="*")
    ap.add_argument("-f", "--file", help="queries, separated by `;` at line end")
    ap.add_argument("--table", action="append", default=[], metavar="NAME=JSON")
    ap.add_argument("--shape", choices=["map", "filter", "many"])
    ap.add_argument("--expect", default=",".join(parity.AGREEMENT))
    a = ap.parse_args(argv)
    queries = list(a.sql)
    if a.file:
        text = open(a.file).read()
        queries += [q.strip() for q in text.split(";\n") if q.strip()]
    tables = dict(_table(t) for t in a.table)
    rows = tables.pop("__THIS__", pa.table({"__unused": pa.array([0], pa.int64())}))
    expect = tuple(a.expect.split(","))
    bad = 0
    for sql in queries:
        sql = sql.rstrip(";")
        c = parity.case(sql, rows, statics=tables, shape=a.shape)
        v = O.run_case(c)
        ours, duck = _answers(c)
        mark = "ok " if v.kind in expect else "!! "
        bad += v.kind not in expect
        print(f"{mark}{v.kind} {v.klass}".rstrip() + f"\n   {sql}")
        print(f"   confit: {ours[:300]}\n   duckdb: {duck[:300]}")
        if v.kind not in expect and v.detail:
            print(f"   detail: {v.detail[:300]}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
