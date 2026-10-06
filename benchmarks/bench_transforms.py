"""Serving timings for SQL, scalar UDFs, and fitted sklearn transforms.

All four scenarios use the same wide input and base SQL projections:

  sql_only    window-derived params, without a UDF
  udf_plain   one author PythonUDF
  tf_width1   one fitted StandardScaler output field
  tf_fields2  two output fields from one width-2 PCA call

Run: uv run --no-sync python -m benchmarks.bench_transforms [--json out.json]
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
    """Build a reproducible mixed-key fixture with six independent numeric columns."""
    rng = random.Random(seed)
    cols: dict[str, list] = {
        "grp": [rng.choice(GROUPS) for _ in range(n)],
        "name": [f"r{i}" for i in range(n)],
    }
    for c in "abcdef":
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
    "tf_width1": f"SELECT sc_transform(sc_fit({BUNDLE}) OVER (PARTITION BY grp),"
    f" {BUNDLE}).a AS t, {BASE} FROM __THIS__",
    "tf_fields2": f"SELECT pca_transform(pca_fit({BUNDLE}) OVER (PARTITION BY grp),"
    f" {BUNDLE}).pca0 AS t0,"
    f" pca_transform(pca_fit({BUNDLE}) OVER (PARTITION BY grp),"
    f" {BUNDLE}).pca1 AS t1, {BASE} FROM __THIS__",
}


def registry() -> dict:
    """Return fresh scalar-UDF and estimator captures for each fitted benchmark."""
    return {
        "half": PythonUDF(
            "half",
            lambda x: None if x is None else x * 0.5,
            pa.schema([("x", pa.float64())]),
        ),
        "sc": StandardScaler(),
        "pca": PCA(n_components=2),
    }


def _prepared(sql: str):
    fitted = SQLProjection.marginalize(sql, captured=registry()).fit(TRAIN)
    function = fitted.compile()
    function.infer_rows(TRAIN.slice(0, 1).to_pylist())
    return fitted, function, TRAIN


def bench(sql: str, sizes=(1, 64), repeats: int = 200) -> dict[int, float]:
    """Median Confit infer_rows duration per row, in ns, keyed by batch size."""
    _, function, train = _prepared(sql)
    rows = train.to_pylist()
    out: dict[int, float] = {}
    for n in sizes:
        batch = rows[:n]
        for _ in range(20):  # exclude warmup calls from the timed samples
            function.infer_rows(batch)
        samples = []
        for i in range(repeats):
            chunk = rows[(i * n) % (len(rows) - n) :][:n]
            t0 = time.perf_counter_ns()
            function.infer_rows(chunk)
            samples.append((time.perf_counter_ns() - t0) / n)
        out[n] = statistics.median(samples)
    return out


def bench_batch(sql: str, repeats: int = 5) -> float:
    """The DuckDB batch path (`transform` over the full table), ns/row."""
    p, _, train = _prepared(sql)
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
