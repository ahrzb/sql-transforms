# ruff: noqa: E501  -- pin claims and queries are single-line by format (query lines map 1:1 to observed segments)
"""Phase-0 DuckDB probes for the dialect logical plan
(2026-08-13-dialect-logical-plan-design.md).

Emits the pins-dialect/*.json fixtures in the house pins format. Claims are
authored against the OBSERVED output of this script — rerun after a DuckDB
upgrade and diff; a changed observation is a changed pin, which is a design
event, not a test flake.

Run from the repo root:  uv run python docs/superpowers/specs/pins-dialect/probe_duckdb.py
"""

from __future__ import annotations

import json
import pathlib

import duckdb

HERE = pathlib.Path(__file__).parent
VER = f"v{duckdb.__version__}"


def run(con: duckdb.DuckDBPyConnection, q: str) -> str:
    try:
        return repr(con.execute(q).fetchall())
    except Exception as e:  # pinned errors are pins too (options-errors.json precedent)
        return f"ERROR: {str(e).splitlines()[0]}"


def observe(con: duckdb.DuckDBPyConnection, pin: dict) -> dict:
    segs = [run(con, q) for q in pin["query"].split("\n")]
    return {**pin, "observed": " | ".join(segs)}


def arrow_type(con: duckdb.DuckDBPyConnection, expr: str) -> str:
    try:
        f = con.sql(f"SELECT {expr} AS v").arrow().schema.field("v")
        return str(f.type)
    except Exception as e:
        return f"ERROR: {str(e).splitlines()[0]}"


def write(name: str, doc: dict) -> None:
    path = HERE / name
    path.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path.name}")


# --- 1. DuckDB -> Arrow export, the full lattice (design table D2) ----------

ARROW_EXPORTS = [
    ("BOOLEAN", "true::BOOLEAN"),
    ("TINYINT", "1::TINYINT"),
    ("SMALLINT", "1::SMALLINT"),
    ("INTEGER", "1::INTEGER"),
    ("BIGINT", "1::BIGINT"),
    ("HUGEINT", "1::HUGEINT"),
    ("UTINYINT", "1::UTINYINT"),
    ("USMALLINT", "1::USMALLINT"),
    ("UINTEGER", "1::UINTEGER"),
    ("UBIGINT", "1::UBIGINT"),
    ("UHUGEINT", "1::UHUGEINT"),
    ("FLOAT", "1.5::FLOAT"),
    ("DOUBLE", "1.5::DOUBLE"),
    ("DECIMAL(18,3)", "1.234::DECIMAL(18,3)"),
    ("DECIMAL(38,9)", "1.234::DECIMAL(38,9)"),
    ("VARCHAR", "'x'::VARCHAR"),
    ("BLOB", "'\\xAA'::BLOB"),
    ("DATE", "DATE '2026-08-13'"),
    ("TIME", "TIME '11:30:00'"),
    ("TIMETZ", "TIMETZ '11:30:00+02'"),
    ("TIMESTAMP", "TIMESTAMP '2026-08-13 11:30:00'"),
    ("TIMESTAMP_S", "'2026-08-13 11:30:00'::TIMESTAMP_S"),
    ("TIMESTAMP_MS", "'2026-08-13 11:30:00'::TIMESTAMP_MS"),
    ("TIMESTAMP_NS", "'2026-08-13 11:30:00'::TIMESTAMP_NS"),
    ("TIMESTAMPTZ", "TIMESTAMPTZ '2026-08-13 11:30:00+00'"),
    ("INTERVAL", "INTERVAL 3 MONTH + INTERVAL 4 DAY + INTERVAL 5 SECOND"),
    ("UUID", "uuid()"),
    ("ENUM", "'happy'::mood"),
    ("BIT", "'101'::BIT"),
    ("LIST(INTEGER)", "[1, 2]"),
    ("STRUCT(a INTEGER)", "{'a': 1}"),
    ("MAP(INTEGER,INTEGER)", "map([1], [2])"),
    ("UNION(num INTEGER, s VARCHAR)", "union_value(num := 2)"),
    # aggregate result types that cross the fit boundary (lattice-spec cross-check)
    ("sum(BIGINT)", "(SELECT sum(x) FROM (VALUES (1::BIGINT)) t(x))"),
    ("sum(DOUBLE)", "(SELECT sum(x) FROM (VALUES (1.5::DOUBLE)) t(x))"),
    ("avg(BIGINT)", "(SELECT avg(x) FROM (VALUES (1::BIGINT)) t(x))"),
    ("count(*)", "(SELECT count(*) FROM (VALUES (1)) t(x))"),
]


def probe_arrow_export(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE TYPE mood AS ENUM ('sad', 'happy')")
    pins = [
        {
            "claim": f"DuckDB {duck} exports to Arrow as recorded",
            "query": f"SELECT {expr} AS v  -- observed = str(rel.arrow().schema.field('v').type)",
            "observed": arrow_type(con, expr),
        }
        for duck, expr in ARROW_EXPORTS
    ]
    write(
        "arrow-export.json",
        {
            "area": "duckdb-arrow-export",
            "duckdb_version": VER,
            "setup": "CREATE TYPE mood AS ENUM ('sad','happy')  -- observed segments are str(pyarrow type) of the result column, not fetchall()",
            "pins": pins,
        },
    )


# --- 2. Sort defaults + default window frame (design table D3) --------------


def probe_sort_window_defaults(con: duckdb.DuckDBPyConnection) -> None:
    pins = [
        {
            "claim": "Bare ORDER BY x is ASC with NULLS LAST",
            "query": "SELECT x FROM (VALUES (2),(NULL),(1)) t(x) ORDER BY x",
        },
        {
            "claim": "ORDER BY x DESC also puts NULLS LAST — DuckDB's default null order is a setting (nulls_last), not tied to direction like PostgreSQL",
            "query": "SELECT x FROM (VALUES (2),(NULL),(1)) t(x) ORDER BY x DESC",
        },
        {
            "claim": "Explicit NULLS FIRST/LAST spellings are accepted in both directions",
            "query": "SELECT x FROM (VALUES (2),(NULL),(1)) t(x) ORDER BY x ASC NULLS FIRST\nSELECT x FROM (VALUES (2),(NULL),(1)) t(x) ORDER BY x DESC NULLS FIRST",
        },
        {
            "claim": "Window ORDER BY has the same default null placement as top-level ORDER BY",
            "query": "SELECT x, row_number() OVER (ORDER BY x) AS rn FROM (VALUES (2),(NULL),(1)) t(x) ORDER BY rn",
        },
        {
            "claim": "Default window frame with ORDER BY is RANGE UNBOUNDED PRECEDING .. CURRENT ROW: peer rows (tied order key) share the running value",
            "query": "SELECT k, v, sum(v) OVER (ORDER BY k) AS s FROM (VALUES (1,10),(1,20),(2,5)) t(k,v) ORDER BY k, v",
        },
        {
            "claim": "Explicit ROWS frame distinguishes the peers — proves the previous pin measured RANGE, not ROWS",
            "query": "SELECT k, v, sum(v) OVER (ORDER BY k, v ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS s FROM (VALUES (1,10),(1,20),(2,5)) t(k,v) ORDER BY k, v",
        },
        {
            "claim": "Window aggregate with PARTITION BY and no ORDER BY spans the whole partition",
            "query": "SELECT k, sum(v) OVER (PARTITION BY k) AS s FROM (VALUES (1,10),(1,20),(2,5)) t(k,v) ORDER BY k, v",
        },
    ]
    write(
        "sort-window-defaults.json",
        {
            "area": "sort-and-window-defaults",
            "duckdb_version": VER,
            "setup": "none  -- multi-statement pins: query lines map 1:1 to ' | '-separated observed segments",
            "pins": [observe(con, p) for p in pins],
        },
    )


# --- 3. Aggregate tiers: order sensitivity, measured -------------------------


def probe_aggregate_tiers(con: duckdb.DuckDBPyConnection) -> None:
    pins = [
        {
            "claim": "sum(DOUBLE) is order-sensitive: permuting the same four values changes the answer (1e16 absorbs 1.0) — the epsilon tier exists",
            "query": "SELECT sum(x) FROM (VALUES (1e16),(1.0),(1.0),(-1e16)) t(x)\nSELECT sum(x) FROM (VALUES (1e16),(-1e16),(1.0),(1.0)) t(x)",
        },
        {
            "claim": "avg(DOUBLE) inherits the same order sensitivity",
            "query": "SELECT avg(x) FROM (VALUES (1e16),(1.0),(1.0),(-1e16)) t(x)\nSELECT avg(x) FROM (VALUES (1e16),(-1e16),(1.0),(1.0)) t(x)",
        },
        {
            "claim": "sum(BIGINT) is exact and order-insensitive (integer algebra; result is decimal128(38,0) at the Arrow boundary per arrow-export.json)",
            "query": "SELECT sum(x) FROM (VALUES (9007199254740993::BIGINT),(1::BIGINT),(-9007199254740993::BIGINT)) t(x)\nSELECT sum(x) FROM (VALUES (1::BIGINT),(9007199254740993::BIGINT),(-9007199254740993::BIGINT)) t(x)",
        },
        {
            "claim": "min/max/count over DOUBLE are order-insensitive (selection, not accumulation) — exact tier",
            "query": "SELECT min(x), max(x), count(x) FROM (VALUES (1e16),(1.0),(-1e16)) t(x)\nSELECT min(x), max(x), count(x) FROM (VALUES (-1e16),(1e16),(1.0)) t(x)",
        },
        {
            "claim": "stddev_pop(DOUBLE) permutation probe: this permutation did NOT expose order sensitivity — insufficient to move stddev out of the epsilon tier (accumulation is still floating-point; absence of a counterexample here is not a proof)",
            "query": "SELECT stddev_pop(x) FROM (VALUES (1e16),(1.0),(1.0),(-1e16)) t(x)\nSELECT stddev_pop(x) FROM (VALUES (1e16),(-1e16),(1.0),(1.0)) t(x)",
        },
        {
            "claim": "Bare decimal literals make 0.1 + 0.2 = 0.3 TRUE in DuckDB (DECIMAL arithmetic); the same comparison over DOUBLE is FALSE — the classic float identity needs explicit casts to observe",
            "query": "SELECT 0.1 + 0.2 = 0.3, typeof(0.1 + 0.2)\nSELECT 0.1::DOUBLE + 0.2::DOUBLE = 0.3::DOUBLE",
        },
        {
            "claim": "quantize via the f32 grid: CAST(CAST(x AS FLOAT) AS DOUBLE) collapses epsilon-different DOUBLE inputs to one value (the pack_trees trick as a plan node)",
            "query": "SELECT CAST(CAST(0.1::DOUBLE + 0.2::DOUBLE AS FLOAT) AS DOUBLE) = CAST(CAST(0.3::DOUBLE AS FLOAT) AS DOUBLE)",
        },
    ]
    write(
        "aggregate-tiers.json",
        {
            "area": "aggregate-order-sensitivity",
            "duckdb_version": VER,
            "setup": "none  -- small VALUES lists execute single-threaded in literal order, so permuting literals permutes accumulation order; multi-statement pins as elsewhere",
            "pins": [observe(con, p) for p in pins],
        },
    )


# --- 4. Strings and operator signatures (printer mapping inputs) ------------


def probe_strings_operators(con: duckdb.DuckDBPyConnection) -> None:
    pins = [
        {
            "claim": "Default string ORDER BY is binary UTF-8 byte order: all ASCII uppercase < lowercase; multibyte (ß U+00DF, é U+00E9, snowman U+2603) sort after ASCII by leading byte",
            "query": "SELECT s FROM (VALUES ('B'),('Z'),('a'),('e'),('ss'),('ß'),('é'),('☃')) t(s) ORDER BY s",
        },
        {
            "claim": "IS NOT DISTINCT FROM is the null-safe equality: NULL~NULL is true, value~NULL is false — the plan's JoinKey null_safe=true spelling in DuckDB",
            "query": "SELECT NULL IS NOT DISTINCT FROM NULL, 1 IS NOT DISTINCT FROM NULL, 1 IS NOT DISTINCT FROM 1",
        },
        {
            "claim": "Integer / is float division (DOUBLE), // is integer division — two distinct signatures for the registry",
            "query": "SELECT 1/2, 1//2, typeof(1/2), typeof(1//2)",
        },
        {
            "claim": "Negative integer // and % semantics, recorded for the Spark/BigQuery signature mapping",
            "query": "SELECT (-7)//2, (-7)%2, 7//(-2), 7%(-2)",
        },
        {
            "claim": "CAST is strict (errors on bad input), TRY_CAST is the try form (NULL on bad input) — the plan's Cast{strict|try} split",
            "query": "SELECT TRY_CAST('x' AS INTEGER)\nSELECT CAST('x' AS INTEGER)",
        },
        {
            "claim": "Strict CAST overflow errors by name (INT32 bound), TRY_CAST nulls it",
            "query": "SELECT CAST(3000000000 AS INTEGER)\nSELECT TRY_CAST(3000000000 AS INTEGER)",
        },
    ]
    write(
        "strings-operators.json",
        {
            "area": "strings-and-operator-signatures",
            "duckdb_version": VER,
            "setup": "none  -- multi-statement pins as elsewhere",
            "pins": [observe(con, p) for p in pins],
        },
    )


# --- 5. Division, cast, and NaN edges (2026-08-13 adversarial review) --------


def probe_division_cast_edges(con: duckdb.DuckDBPyConnection) -> None:
    pins = [
        {
            "claim": "/ is IEEE on zero divisors: 1/0 = inf, 0/0 = NaN, 1/-0.0 = -inf (Spark ANSI and BigQuery's bare / both ERROR here - forced in the printers)",
            "query": "SELECT 1/0, 0/0\nSELECT 1.0::DOUBLE / (-0.0)::DOUBLE",
        },
        {
            "claim": "// and % return NULL on a zero divisor (Spark ANSI and BigQuery DIV/MOD error - guarded in the printers)",
            "query": "SELECT 1//0, 1%0",
        },
        {
            "claim": "INT_MIN % -1 is an overflow trap at every width (Spark and BigQuery MOD return 0 - forced back into an error in the printers)",
            "query": "SELECT (-9223372036854775807 - 1) % (-1)\nSELECT (-2147483648)::INTEGER % (-1)::INTEGER",
        },
        {
            "claim": "CAST DOUBLE -> INTEGER rounds half-even, never truncates (Spark truncates - forced via rint)",
            "query": "SELECT CAST(2.7::DOUBLE AS INTEGER), CAST(2.5::DOUBLE AS INTEGER), CAST(1.5::DOUBLE AS INTEGER), CAST((-2.5)::DOUBLE AS INTEGER)",
        },
        {
            "claim": "CAST DECIMAL -> INTEGER rounds half-AWAY-from-zero (Spark truncates - forced via round)",
            "query": "SELECT CAST(2.5::DECIMAL(3,1) AS INTEGER), CAST((-2.5)::DECIMAL(3,1) AS INTEGER), CAST(2.4::DECIMAL(3,1) AS INTEGER)",
        },
        {
            "claim": "CAST '1.5' (string) -> INTEGER parses and ROUNDS; Spark treats it as malformed (string-source casts refuse in the printers)",
            "query": "SELECT CAST('1.5' AS INTEGER), TRY_CAST('1.5' AS INTEGER)",
        },
        {
            "claim": "Floats order TOTALLY: NaN equals NaN and exceeds everything (BigQuery comparisons are IEEE - forced with IS_NAN cases)",
            "query": "SELECT 'NaN'::DOUBLE = 'NaN'::DOUBLE, 5.0 < 'NaN'::DOUBLE, 'NaN'::DOUBLE < 5.0, 'NaN'::DOUBLE IS NOT DISTINCT FROM 'NaN'::DOUBLE",
        },
        {
            "claim": "Decimal literal typing counts a bare-zero integer part as one digit: typeof(0.5) = DECIMAL(2,1)",
            "query": "SELECT typeof(0.5), typeof(1.50), typeof(10.5)",
        },
        {
            "claim": "FLOAT / FLOAT computes at FLOAT, not DOUBLE (the plan refuses f32 division rather than derive a wrong width)",
            "query": "SELECT typeof(1.5::FLOAT / 2.0::FLOAT)",
        },
    ]
    write(
        "division-cast-edges.json",
        {
            "area": "division-cast-nan-edges",
            "duckdb_version": VER,
            "setup": "none  -- multi-statement pins as elsewhere; measured 2026-08-13 while verifying the adversarial review's findings",
            "pins": [observe(con, p) for p in pins],
        },
    )


def main() -> None:
    con = duckdb.connect()
    probe_arrow_export(con)
    probe_sort_window_defaults(con)
    probe_aggregate_tiers(con)
    probe_strings_operators(con)
    probe_division_cast_edges(con)


if __name__ == "__main__":
    main()
