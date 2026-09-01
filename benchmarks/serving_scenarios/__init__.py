"""Serving-path benchmark scenarios — famous tabular-ML inference paths.

Each module reproduces the SERVING shape of a well-known Kaggle-style
solution: a wide input row -> scalar feature expressions + LEFT JOINs to
fitted static tables (lookup dims and pre-materialized target/frequency
encodings — that is what a mean encoding IS at serve time).

Module contract (see any scenario module):
    NAME, KAGGLE, N_INPUT_COLS, N_OUTPUT_COLS, ROW_SCHEMA, SQL,
    make_statics(seed) -> dict[str, pa.Table],
    make_rows(seed, n) -> list[dict],
    handcrafted(statics) -> Callable[[dict], dict]

The standing trust anchor is three-way parity, enforced by
tests/test_serving_scenarios.py and re-checked by the bench harness:
specializer output == DuckDB itself == the handcrafted Python twin.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable

import duckdb
import pyarrow as pa
from confit.compare import multiset, sequence

NAMES = ["titanic", "house_prices", "fraud_txn", "store_sales", "feature_bundle"]

SEED = 20260726
_ARROW = {
    "int": pa.int64(),
    "float": pa.float64(),
    "str": pa.string(),
    "bool": pa.bool_(),
}


def load(name: str):
    return importlib.import_module(f"benchmarks.serving_scenarios.{name}")


def all_scenarios():
    return [load(n) for n in NAMES]


def arrow_schema(schema: dict[str, str] | pa.Schema) -> pa.Schema:
    # A scenario may declare ROW_SCHEMA as a pa.Schema outright: the
    # dict[str, str] shorthand cannot spell a struct column (TASK-114).
    if isinstance(schema, pa.Schema):
        return schema
    return pa.schema(
        pa.field(n, _ARROW[s.rstrip("?")], nullable=s.endswith("?"))
        for n, s in schema.items()
    )


def rows_table(mod, rows: list[dict]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=arrow_schema(mod.ROW_SCHEMA))


def build_spec_fn(mod, statics: dict[str, pa.Table]):
    """The specializer serving fn (env knobs select backend/boundary)."""
    from confit import DuckDBInferFn

    return DuckDBInferFn(
        mod.SQL,
        row_tables={"__THIS__": arrow_schema(mod.ROW_SCHEMA)},
        static_tables=statics,
    )


def duckdb_server(
    mod, statics: dict[str, pa.Table]
) -> Callable[[list[dict]], list[dict]]:
    """The 'just run DuckDB' baseline: statics materialized ONCE as native
    tables (DuckDB gets the same prepare-time courtesy as everyone else);
    each call pays what a DuckDB-backed service would — build an Arrow
    batch from the row dicts, register, execute, fetch."""
    con = duckdb.connect()
    for name, table in statics.items():
        con.register(f"__arrow_{name}", table)
        # S608: table names come from our own scenario modules, not input.
        ddl = f'CREATE TABLE "{name}" AS SELECT * FROM "__arrow_{name}"'  # noqa: S608
        con.execute(ddl)
        con.unregister(f"__arrow_{name}")
    schema = arrow_schema(mod.ROW_SCHEMA)

    def call(rows: list[dict]) -> list[dict]:
        con.register("__THIS__", pa.Table.from_pylist(rows, schema=schema))
        out = con.execute(mod.SQL).to_arrow_table().to_pylist()
        con.unregister("__THIS__")
        return out

    return call


def verify_parity(mod, n: int = 300) -> list[str]:
    """specializer == DuckDB == handcrafted, exact (repr-level floats).
    Returns human-readable mismatches; empty list = trusted."""
    statics = mod.make_statics(SEED)
    rows = mod.make_rows(SEED + 1, n)

    fn = build_spec_fn(mod, statics)
    got_spec = fn.infer_rows(rows)
    got_duck = duckdb_server(mod, statics)(rows)
    hand = mod.handcrafted(statics)
    got_hand = [hand(r) for r in rows]

    problems: list[str] = []
    if fn.backend != "cranelift":
        problems.append(f"{mod.NAME}: backend is {fn.backend}, not cranelift")

    # The columnar boundary is the same engine behind a second entry point,
    # so it agrees POSITIONALLY with infer_rows or it is a defect (TASK-114).
    got_arrow = fn.infer_arrow(rows_table(mod, rows)).to_pylist()
    if got_arrow != got_spec:
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(got_arrow, got_spec, strict=False))
                if a != b
            ),
            min(len(got_arrow), len(got_spec)),
        )
        problems.append(
            f"{mod.NAME} infer_arrow vs infer_rows: {len(got_arrow)} vs "
            f"{len(got_spec)} rows, first difference at {first}: "
            f"{got_arrow[first : first + 1]} != {got_spec[first : first + 1]}"
        )

    # No more output='dict' vs typed-mode differential: dict-out is the only
    # mode the arrow schema surface has (output= was deleted).

    # DuckDB reorders rows after hash joins (the specializer preserves input
    # order — a contract, not an accident): multiset equality, like the
    # corpus replay. The handcrafted twin is positional.
    if multiset(got_spec) != multiset(got_duck):
        spec_only = multiset(got_spec)
        duck_only = multiset(got_duck)
        first = next(
            (
                i
                for i, (a, b) in enumerate(zip(spec_only, duck_only, strict=False))
                if a != b
            ),
            min(len(spec_only), len(duck_only)),
        )
        problems.append(
            f"{mod.NAME} vs duckdb: multiset mismatch at sorted index {first}: "
            f"{spec_only[first] if first < len(spec_only) else '<missing>'} != "
            f"{duck_only[first] if first < len(duck_only) else '<missing>'}"
        )
    if len(got_hand) != len(got_spec):
        problems.append(
            f"{mod.NAME} vs handcrafted: {len(got_spec)} vs {len(got_hand)} rows"
        )
    else:
        spec_seq = sequence(got_spec)
        hand_seq = sequence(got_hand)
        for i, (a, b) in enumerate(zip(got_spec, got_hand, strict=True)):
            if spec_seq[i] != hand_seq[i]:
                problems.append(f"{mod.NAME} vs handcrafted row {i}: {a!r} != {b!r}")
                if len(problems) > 5:
                    return problems
    return problems
