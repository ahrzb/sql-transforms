# ruff: noqa: E501  -- pin claims and queries are single-line by format (query lines map 1:1 to observed segments)
"""Phase-0 Spark probes for the dialect logical plan
(2026-08-13-dialect-logical-plan-design.md).

Emits spark-ansi.json. The dialect under measurement is Spark PLUS the pinned
configuration below (design decision D6) — a probe run under different flags
measures a different dialect and must not overwrite these pins.

Run from the repo root:  uv run python packages/confit/docs/specs/pins-dialect/probe_spark.py
(needs `uv pip install pyspark`; a JVM must be on PATH)
"""

from __future__ import annotations

import json
import pathlib

from pyspark.sql import SparkSession

HERE = pathlib.Path(__file__).parent

PINNED_CONFIG = {
    "spark.sql.ansi.enabled": "true",
    "spark.sql.session.timeZone": "UTC",
    # local[1]: single-threaded accumulation so permutation probes measure the
    # operation's order sensitivity, not the scheduler's nondeterminism.
    "master": "local[1]",
}


def run(spark: SparkSession, q: str) -> str:
    try:
        return repr([tuple(r) for r in spark.sql(q).collect()])
    except Exception as e:
        first = str(e).strip().splitlines()[0]
        return f"ERROR: {first}"


def observe(spark: SparkSession, pin: dict) -> dict:
    segs = [run(spark, q) for q in pin["query"].split("\n")]
    return {**pin, "observed": " | ".join(segs)}


PINS = [
    {
        "claim": "Bare ORDER BY x is ASC with NULLS FIRST — the OPPOSITE default null order to DuckDB (nulls_last); the plan's mandatory explicit null order exists because of exactly this",
        "query": "SELECT x FROM VALUES (2),(NULL),(1) t(x) ORDER BY x",
    },
    {
        "claim": "ORDER BY x DESC puts NULLS LAST — Spark ties null order to direction (PostgreSQL-style), unlike DuckDB's direction-independent setting",
        "query": "SELECT x FROM VALUES (2),(NULL),(1) t(x) ORDER BY x DESC",
    },
    {
        "claim": "Explicit NULLS FIRST/LAST spellings are accepted in both directions — the printer can always force",
        "query": "SELECT x FROM VALUES (2),(NULL),(1) t(x) ORDER BY x ASC NULLS LAST\nSELECT x FROM VALUES (2),(NULL),(1) t(x) ORDER BY x DESC NULLS FIRST",
    },
    {
        "claim": "Default window frame with ORDER BY is RANGE UNBOUNDED PRECEDING .. CURRENT ROW — agrees with DuckDB (peer rows share the running value)",
        "query": "SELECT k, v, sum(v) OVER (ORDER BY k) AS s FROM VALUES (1,10),(1,20),(2,5) t(k,v) ORDER BY k, v",
    },
    {
        "claim": "Default string ordering agrees with DuckDB's binary UTF-8 order on the same probe values",
        "query": "SELECT s FROM VALUES ('B'),('Z'),('a'),('e'),('ss'),('ß'),('é'),('☃') t(s) ORDER BY s",
    },
    {
        "claim": "<=> is the null-safe equality (the plan's JoinKey null_safe=true spelling in Spark); IS NOT DISTINCT FROM is also accepted",
        "query": "SELECT NULL <=> NULL, 1 <=> NULL, 1 <=> 1\nSELECT NULL IS NOT DISTINCT FROM NULL, 1 IS NOT DISTINCT FROM NULL",
    },
    {
        "claim": "Integer / is float division (DOUBLE) as in DuckDB; the integer-division spelling is `div` (DuckDB spells it //)",
        "query": "SELECT 1/2, typeof(1/2), 1 div 2, typeof(1 div 2)",
    },
    {
        "claim": "Negative integer div and % semantics, recorded against DuckDB's ((-7)//2, (-7)%2, 7//(-2), 7%(-2)) for the signature mapping",
        "query": "SELECT (-7) div 2, (-7)%2, 7 div (-2), 7%(-2)",
    },
    {
        "claim": "ANSI mode: integer overflow errors by name instead of wrapping — matches DuckDB's trap class",
        "query": "SELECT CAST(2147483647 AS INT) + CAST(1 AS INT)",
    },
    {
        "claim": "ANSI mode: strict CAST errors on bad input, try_cast is the try form (NULL) — same strict|try split as DuckDB's CAST/TRY_CAST",
        "query": "SELECT CAST('x' AS INT)\nSELECT try_cast('x' AS INT)",
    },
    {
        "claim": "ANSI mode: strict CAST overflow errors by name, try_cast nulls it",
        "query": "SELECT CAST(3000000000 AS INT)\nSELECT try_cast(3000000000 AS INT)",
    },
    {
        "claim": "Bare decimal literals: 0.1 + 0.2 = 0.3 is TRUE (DECIMAL arithmetic) — agrees with DuckDB's literal typing, disagrees over DOUBLE as everywhere",
        "query": "SELECT 0.1 + 0.2 = 0.3, typeof(0.1 + 0.2)\nSELECT 0.1D + 0.2D = 0.3D",
    },
    {
        "claim": "sum(DOUBLE) is order-sensitive under local[1]: same four values, permuted, different answer — Spark shares DuckDB's epsilon tier",
        "query": "SELECT sum(x) FROM VALUES (1e16),(1.0D),(1.0D),(-1e16) t(x)\nSELECT sum(x) FROM VALUES (1e16),(-1e16),(1.0D),(1.0D) t(x)",
    },
    {
        "claim": "sum(BIGINT) result type is BIGINT (errors on overflow under ANSI) — DuckDB's is DECIMAL(38,0); a printer forcing DuckDB result types must cast the argument",
        "query": "SELECT typeof(sum(x)) FROM VALUES (1L) t(x)\nSELECT sum(x) FROM VALUES (9223372036854775807L),(1L) t(x)",
    },
    {
        "claim": "ANSI: zero divisors ERROR for /, div and % (DuckDB: / is IEEE inf/NaN, // and % give NULL) - the printer guards every division",
        "query": "SELECT 1/0\nSELECT 1 div 0\nSELECT 1 % 0",
    },
    {
        "claim": "INT_MIN % -1 returns 0 (DuckDB traps) - the printer forces the error through the checked div spelling",
        "query": "SELECT CAST(-9223372036854775808 AS BIGINT) % CAST(-1 AS BIGINT)",
    },
    {
        "claim": "CAST DOUBLE -> INT truncates toward zero (DuckDB rounds half-even); rint is Spark's half-even rounding - the printer's forcing",
        "query": "SELECT CAST(CAST(2.7 AS DOUBLE) AS INT), CAST(CAST(2.5 AS DOUBLE) AS INT)\nSELECT rint(CAST(2.7 AS DOUBLE)), rint(CAST(2.5 AS DOUBLE)), rint(CAST(1.5 AS DOUBLE))",
    },
    {
        "claim": "round(dec, 0) is HALF_UP = half-away-from-zero, matching DuckDB's DECIMAL -> int rule - the printer's forcing",
        "query": "SELECT round(CAST(2.5 AS DECIMAL(3,1)), 0), round(CAST(-2.5 AS DECIMAL(3,1)), 0)",
    },
    {
        "claim": "Spark's NaN semantics are ALREADY DuckDB's total order: NaN = NaN is true and NaN exceeds everything - no forcing needed on comparisons",
        "query": "SELECT double('NaN') = double('NaN'), 5.0D < double('NaN'), double('NaN') < 5.0D, double('NaN') <=> double('NaN')",
    },
    {
        "claim": "Bought scalar functions agree with DuckDB, NULL propagation included; starts_with spells startswith; concat propagates NULL (DuckDB skips - forced via coalesce); levenshtein counts CHARACTERS (DuckDB: bytes - refused)",
        "query": "SELECT upper('héllo'), length('é☃'), contains('abc',''), startswith('abc','ab'), instr('abcb','b'), repeat('ab',-1), translate('abcba','abc','xy'), concat_ws('-','a',CAST(NULL AS STRING),'c')\nSELECT concat('a',CAST(NULL AS STRING),'c'), concat(coalesce('a',''),coalesce(CAST(NULL AS STRING),''),coalesce('c',''))\nSELECT levenshtein('é','e')",
    },
    {
        "claim": "TIMESTAMP_NTZ is reachable from SQL by cast — the landing zone for DuckDB's wall-clock TIMESTAMP; plain TIMESTAMP is the session-tz instant type (landing zone for TIMESTAMPTZ under pinned UTC)",
        "query": "SELECT typeof(CAST('2026-08-13 11:30:00' AS TIMESTAMP_NTZ)), typeof(TIMESTAMP '2026-08-13 11:30:00')",
    },
]


def main() -> None:
    builder = SparkSession.builder.appName("dialect-phase0-pins")
    for k, v in PINNED_CONFIG.items():
        if k == "master":
            builder = builder.master(v)
        else:
            builder = builder.config(k, v)
    spark = builder.getOrCreate()
    doc = {
        "area": "spark-ansi-baseline",
        "spark_version": spark.version,
        "pinned_config": PINNED_CONFIG,
        "setup": "none  -- multi-statement pins: query lines map 1:1 to ' | '-separated observed segments; observed is [tuple(row) for row in collect()]",
        "pins": [observe(spark, p) for p in PINS],
    }
    out = HERE / "spark-ansi.json"
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {out.name}")
    spark.stop()


if __name__ == "__main__":
    main()
