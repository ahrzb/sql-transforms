"""Transformer serving-path bench — what the opaque-UDF trampoline costs.

`bench_serving.py` measures pure-SQL serving. This one measures the piece
that had no number: a fitted sklearn transformer served row-at-a-time
through the UDF extern slot (GIL crossing + `est.transform` on a 1xk
block), against the same query with the transformer removed.

Rows are the same wide shape in every variant, so the delta between rows
IS the transformer cost, not the query's.

  sql_only    marginalized aggregates only — the ceiling (no UDF at all)
  udf_plain   + one author PythonUDF (trampoline, no sklearn)
  tf_width1   + one fitted StandardScaler (one extern call per row)
  tf_fields2  two field accesses on ONE width-2 PCA

`tf_fields2` pins DRAFT-24 loop 4 (TASK-63): k accessed fields share ONE
evaluation per row on both paths — the row path (confit: lane reads off
one ecall) and the batch column (DuckDB `transform`: one struct-returning
call, CSE merges the identical mentions). The former `tf_bare2` scenario
(a bare width-2 item) stopped being legal SQL with struct-valued calls —
bare transformer items refuse until DRAFT-25's nested outputs.

Run: uv run python -m benchmarks.bench_transforms [--json out.json]
"""

from __future__ import annotations

import json
import random
import statistics
import sys
import time

import pyarrow as pa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sql_transform import PythonUDF, SQLProjection

SEED = 20260804
N_TRAIN = 4000
GROUPS = ["us", "de", "fr", "jp", "br", None]


def make_table(n: int, seed: int = SEED) -> pa.Table:
    rng = random.Random(seed)
    cols: dict[str, list] = {
        "grp": [rng.choice(GROUPS) for _ in range(n)],
        "name": [f"r{i}" for i in range(n)],
    }
    for c in "abcdef":  # NOT 'g' — that would shadow the group column
        cols[c] = [rng.uniform(-50, 50) for _ in range(n)]
    return pa.table(cols)


TRAIN = make_table(N_TRAIN)
BUNDLE = "struct_pack(a := a, b := b, c := c, d := d)"

# Every variant projects the same 6 SQL columns; only the transformer part
# differs, so row-to-row deltas are the transformer's cost alone.
BASE = (
    "a - avg(a) OVER (PARTITION BY grp) AS z1,"
    " b / (1 + abs(c)) AS z2,"
    " d - min(d) OVER (PARTITION BY grp) AS z3,"
    " e * 2 + f AS z4,"
    " greatest(f, 0) AS z5,"
    " name"
)
QUERIES = {
    "sql_only": f"SELECT {BASE} FROM __THIS__",
    "udf_plain": f"SELECT half(a) AS t, {BASE} FROM __THIS__",
    "tf_width1": f"SELECT sc({BUNDLE}) OVER (PARTITION BY grp).a AS t, {BASE}"
    " FROM __THIS__",
    "tf_fields2": f"SELECT pca({BUNDLE}) OVER (PARTITION BY grp).pca0 AS t0,"
    f" pca({BUNDLE}) OVER (PARTITION BY grp).pca1 AS t1, {BASE} FROM __THIS__",
}


def registry() -> dict:
    return {
        "half": PythonUDF(
            "half", lambda x: None if x is None else x * 0.5, ("f64",), ("f64",)
        ),
        "sc": StandardScaler(),
        "pca": PCA(n_components=2),
    }


def _prepared(sql: str):
    p = SQLProjection(sql, transformers=registry()).fit(TRAIN)
    p.infer(TRAIN.to_pylist()[0])  # force the lazy Confit prepare
    return p, TRAIN


def bench(sql: str, sizes=(1, 64), repeats: int = 200) -> dict[int, float]:
    p, train = _prepared(sql)
    rows = train.to_pylist()
    out: dict[int, float] = {}
    for n in sizes:
        batch = rows[:n]
        for _ in range(20):  # warmup
            p.infer_batch(batch)
        samples = []
        for i in range(repeats):
            chunk = rows[(i * n) % (len(rows) - n) :][:n]
            t0 = time.perf_counter_ns()
            p.infer_batch(chunk)
            samples.append((time.perf_counter_ns() - t0) / n)
        out[n] = statistics.median(samples)
    return out


def bench_batch(sql: str, repeats: int = 5) -> float:
    """The DuckDB batch path (`transform` over the full table), ns/row."""
    p, train = _prepared(sql)
    p.transform(train)  # warmup
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        p.transform(train)
        samples.append((time.perf_counter_ns() - t0) / train.num_rows)
    return statistics.median(samples)


def main() -> int:
    results: dict[str, dict[int, float]] = {}
    batch: dict[str, float] = {}
    for label, sql in QUERIES.items():
        results[label] = bench(sql)
        batch[label] = bench_batch(sql)
    base = results["sql_only"]
    print(
        f"{'variant':12} {'n=1 ns/row':>12} {'n=64 ns/row':>12}"
        f" {'batch ns/row':>13}  delta vs sql_only (n=1)"
    )
    print("-" * 76)
    for label, r in results.items():
        d1 = r[1] - base[1]
        print(
            f"{label:12} {r[1]:12,.0f} {r[64]:12,.0f} {batch[label]:13,.0f}"
            + ("" if label == "sql_only" else f"   +{d1:,.0f}ns/row")
        )
    if "--json" in sys.argv:
        path = sys.argv[sys.argv.index("--json") + 1]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    k: {str(n): v for n, v in r.items()} | {"batch": batch[k]}
                    for k, r in results.items()
                },
                f,
            )
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
