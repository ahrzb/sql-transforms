"""A fitted feature bundle arriving as ONE struct column (TASK-114).

The other scenarios are wide at the TOP level: forty named scalar columns.
Real fitted pipelines rarely serve that shape. A feature store, a Kaggle
"features_v3" blob or a protobuf request message hands the serving path one
nested record -- sixteen scalar leaves under a single nullable parent -- plus
a couple of routing scalars beside it.

That shape is what this scenario benches, and it is the shape where the
validity fold matters: `tenure_months` is a NOT NULL leaf, so under a NULL
bundle it has no validity buffer of its own and its data buffer still holds
a live value. Every parity leg here reads it directly, so a fold-free ingest
fails the standing gate rather than the bench merely being slower.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pyarrow as pa

NAME = "feature_bundle"
KAGGLE = (
    "The feature-store serving shape: one fitted 'features_v3' bundle as a "
    "nested record (16 mixed-type leaves under a nullable parent, including a "
    "NOT NULL tenure leaf) plus routing scalars, feature-engineered into 21 "
    "model inputs -- missingness flags, winsorized ratios, log1p skew fixes, "
    "interaction flags, a concatenated segment key -- with the region target "
    "encoding served as a fitted LEFT JOIN carrying the unseen-'offshore' miss."
)

N_INPUT_COLS = 3
N_OUTPUT_COLS = 21

# The bundle: 16 scalar leaves of mixed type under one NULLABLE parent.
# `tenure_months` is deliberately NOT NULL -- the fold hazard, in the bench.
_BUNDLE = pa.struct(
    [
        pa.field("age", pa.float64()),
        pa.field("income", pa.float64()),
        pa.field("tenure_months", pa.int64(), nullable=False),
        pa.field("n_txn", pa.int64()),
        pa.field("avg_amt", pa.float64()),
        pa.field("max_amt", pa.float64()),
        pa.field("ratio", pa.float64()),
        pa.field("score", pa.float64()),
        pa.field("segment", pa.string()),
        pa.field("region", pa.string()),
        pa.field("device", pa.string()),
        pa.field("is_premium", pa.bool_()),
        pa.field("is_verified", pa.bool_()),
        pa.field("risk_band", pa.string()),
        pa.field("credit", pa.int64()),
        pa.field("utilization", pa.float64()),
    ]
)

ROW_SCHEMA = pa.schema(
    [
        pa.field("customer_id", pa.int64(), nullable=False),
        pa.field("channel", pa.string(), nullable=False),
        pa.field("feat", _BUNDLE, nullable=True),
    ]
)

# Fitted constants a pipeline bakes into the serving query as literals.
AGE_MEDIAN = 41.0
INCOME_MEDIAN = 52000.0
GLOBAL_REGION_RATE = 0.0512

_REGIONS = ["north", "south", "east", "west", "central"]


def make_statics(seed: int) -> dict[str, pa.Table]:
    rnd = random.Random(seed)
    base = {
        "north": (0.0413, 18422),
        "south": (0.0671, 15118),
        "east": (0.0522, 20907),
        "west": (0.0348, 12640),
        "central": (0.0794, 9033),
    }
    region_enc = pa.Table.from_pylist(
        [
            {
                "region": r,
                "region_rate": round(v + rnd.uniform(-0.002, 0.002), 6),
                "region_n": n,
            }
            for r, (v, n) in base.items()
        ],
        schema=pa.schema(
            [
                ("region", pa.string()),
                ("region_rate", pa.float64()),
                ("region_n", pa.int64()),
            ]
        ),
    )
    return {"region_enc": region_enc}


def make_rows(seed: int, n: int) -> list[dict]:
    rnd = random.Random(seed)
    rows: list[dict] = []
    for i in range(n):
        channel = rnd.choices(["web", "app", "call"], [60, 33, 7])[0]
        # ~6% of requests arrive with the bundle entirely absent -- the
        # feature store missed, and every leaf is NULL through the parent.
        if rnd.random() < 0.06:
            rows.append({"customer_id": 900000 + i, "channel": channel, "feat": None})
            continue
        avg_amt = round(math.exp(rnd.gauss(3.4, 0.8)), 4)
        rows.append(
            {
                "customer_id": 900000 + i,
                "channel": channel,
                "feat": {
                    "age": (
                        None
                        if rnd.random() < 0.11
                        else round(min(94.0, max(18.0, rnd.gauss(41.0, 14.0))), 1)
                    ),
                    "income": (
                        None
                        if rnd.random() < 0.08
                        else round(math.exp(rnd.gauss(10.8, 0.55)), 2)
                    ),
                    "tenure_months": rnd.randint(0, 240),
                    "n_txn": rnd.choices([0, 1, 2, 5, 11, 40], [8, 19, 25, 28, 15, 5])[
                        0
                    ],
                    "avg_amt": avg_amt,
                    "max_amt": round(avg_amt * rnd.uniform(1.0, 6.5), 4),
                    "ratio": (
                        None
                        if rnd.random() < 0.05
                        else round(rnd.uniform(-0.2, 1.4), 6)
                    ),
                    "score": round(rnd.uniform(0.0, 1.0), 6),
                    "segment": rnd.choices(
                        ["mass", "affluent", "private", "student"], [58, 27, 6, 9]
                    )[0],
                    # 'offshore' is unseen by region_enc -- the LEFT JOIN miss.
                    "region": rnd.choices(
                        _REGIONS + ["offshore"], [24, 21, 26, 17, 9, 3]
                    )[0],
                    "device": rnd.choices(["mobile", "desktop", "tablet"], [63, 30, 7])[
                        0
                    ],
                    "is_premium": rnd.random() < 0.22,
                    "is_verified": (
                        None if rnd.random() < 0.04 else rnd.random() < 0.81
                    ),
                    "risk_band": rnd.choices(["A", "B", "C", "D"], [31, 39, 22, 8])[0],
                    "credit": rnd.randint(300, 850),
                    "utilization": (
                        None if rnd.random() < 0.07 else round(rnd.uniform(0.0, 1.3), 6)
                    ),
                },
            }
        )
    return rows


SQL = """
SELECT
  customer_id AS customer_id,
  channel AS channel,
  coalesce(feat.age, 41.0) AS age_filled,
  CASE WHEN feat.age IS NULL THEN 1 ELSE 0 END AS age_missing,
  coalesce(feat.income, 52000.0) / 12.0 AS monthly_income,
  feat.tenure_months AS tenure_months,
  CASE WHEN feat.tenure_months >= 24 THEN 1 ELSE 0 END AS is_tenured,
  feat.n_txn * 1.0e0 AS n_txn,
  coalesce(feat.avg_amt, 0.0) AS avg_amt,
  greatest(coalesce(feat.max_amt, 0.0), coalesce(feat.avg_amt, 0.0)) AS peak_amt,
  least(greatest(feat.ratio, 0.0), 1.0) AS ratio_clipped,
  ln(1.0 + feat.score) AS score_log1p,
  feat.segment || '/' || channel AS segment_channel,
  CASE WHEN feat.is_premium THEN 1 ELSE 0 END AS premium_flag,
  CASE WHEN feat.is_verified AND feat.is_premium THEN 1 ELSE 0 END AS trusted_flag,
  CASE WHEN feat.device = 'mobile' THEN 1 ELSE 0 END AS is_mobile,
  feat.risk_band AS risk_band,
  feat.credit - 300 AS credit_offset,
  coalesce(feat.utilization, 0.0) * 100.0 AS utilization_pct,
  coalesce(re.region_rate, 0.0512) AS region_rate,
  re.region_n AS region_n
FROM __THIS__
LEFT JOIN region_enc AS re ON feat.region = re.region
"""


def handcrafted(statics: dict[str, pa.Table]) -> Callable[[dict], dict]:
    """What a competent engineer hand-writes: the bundle unpacked once per
    row, the encoding table as a plain dict prepared at startup."""
    enc = {r["region"]: r for r in statics["region_enc"].to_pylist()}
    empty = dict.fromkeys(
        [
            "age",
            "income",
            "tenure_months",
            "n_txn",
            "avg_amt",
            "max_amt",
            "ratio",
            "score",
            "segment",
            "region",
            "device",
            "is_premium",
            "is_verified",
            "risk_band",
            "credit",
            "utilization",
        ]
    )

    def infer(row: dict) -> dict:
        channel = row["channel"]
        # A NULL bundle nulls every leaf -- the OR of both validity levels.
        f = row["feat"] or empty
        age = f["age"]
        ratio = f["ratio"]
        score = f["score"]
        tenure = f["tenure_months"]
        credit = f["credit"]
        segment = f["segment"]
        premium = f["is_premium"]
        verified = f["is_verified"]
        avg_amt = f["avg_amt"] if f["avg_amt"] is not None else 0.0
        max_amt = f["max_amt"] if f["max_amt"] is not None else 0.0
        e = enc.get(f["region"])
        n_txn = f["n_txn"]
        return {
            "customer_id": row["customer_id"],
            "channel": channel,
            "age_filled": age if age is not None else 41.0,
            "age_missing": 1 if age is None else 0,
            "monthly_income": (f["income"] if f["income"] is not None else 52000.0)
            / 12.0,
            "tenure_months": tenure,
            "is_tenured": 1 if tenure is not None and tenure >= 24 else 0,
            "n_txn": None if n_txn is None else n_txn * 1.0,
            "avg_amt": avg_amt,
            "peak_amt": max(max_amt, avg_amt),
            # greatest()/least() SKIP nulls in DuckDB, so a NULL ratio
            # clamps to the floor rather than propagating.
            "ratio_clipped": 0.0 if ratio is None else min(max(ratio, 0.0), 1.0),
            "score_log1p": None if score is None else math.log(1.0 + score),
            "segment_channel": None if segment is None else segment + "/" + channel,
            "premium_flag": 1 if premium else 0,
            "trusted_flag": 1 if (verified and premium) else 0,
            "is_mobile": 1 if f["device"] == "mobile" else 0,
            "risk_band": f["risk_band"],
            "credit_offset": None if credit is None else credit - 300,
            "utilization_pct": (
                f["utilization"] if f["utilization"] is not None else 0.0
            )
            * 100.0,
            "region_rate": e["region_rate"] if e is not None else 0.0512,
            "region_n": e["region_n"] if e is not None else None,
        }

    return infer
