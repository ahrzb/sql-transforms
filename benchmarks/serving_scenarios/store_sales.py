"""store_sales — Rossmann/Walmart-style store-sales feature serving.

Serving shape reproduced here:
  * the Rossmann ``store.csv`` dim (store_type, assortment, competition_*,
    promo2_since_*) LEFT-JOINed on store_id — with ~10% of serving store_ids
    absent from the dim (new stores), so join-miss NULL semantics are real;
  * per-store mean-sales / mean-customers / sales-per-customer target
    encodings (what a fitted mean encoding IS at serve time), with partial
    coverage and literal global priors as fallback;
  * day-of-week / month / region seasonal-factor encodings;
  * competition-open-months and promo2-active-weeks arithmetic clamped with
    NULL-ignoring greatest(0, ...), promo interactions, day-of-week one-hots,
    state-holiday IN-set CASEs, a store_type x promo cross, ratios vs store
    means, and Walmart-style econ covariates (fuel/CPI/unemployment/markdowns);
  * the log/cyclical features the famous solutions actually ship (previously
    compromised away when the SQL surface lacked math builtins, now real):
    ln(1+CompetitionDistance) and ln(1+markdown)/ln(1+mean_sales) log1p
    transforms, competition-age-in-years via floor(), Rossmann week-of-month
    via trunc((day-1)/7), sin/cos week-of-year + day-of-week cyclical
    seasonality augmenting the seasonal-factor joins, the canonical
    Jan/Apr/Jul/Oct PromoInterval renewal-month flag and a Nov/Dec peak-season
    flag via IN lists, and a promo2-weeks saturation cap via least().
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pyarrow as pa

NAME = "store_sales"
KAGGLE = (
    "Rossmann Store Sales (+ Walmart Recruiting) serving path: store.csv dim join, "
    "competition-open-months / promo2-weeks arithmetic, per-store mean-target encodings "
    "with global priors, dow/month seasonal factors, store_type x promo cross, "
    "holiday CASEs, Walmart econ covariates, log1p distance/markdown/sales "
    "transforms, week-of-month, sin/cos cyclical seasonality, PromoInterval flag."
)

N_INPUT_COLS = 21
N_OUTPUT_COLS = 56

ROW_SCHEMA: dict[str, str] = {
    "record_id": "int",  # the Rossmann test.csv `Id`: unique serving-row key
    "store_id": "int",
    "day_of_week": "int",  # 1=Mon .. 7=Sun
    "day_of_month": "int",
    "month": "int",
    "year": "int",
    "week_of_year": "int",
    "is_open": "bool",
    "promo": "bool",
    "promo2": "bool",  # store enrolled in Promo2, known to the caller
    "school_holiday": "bool",
    "state_holiday": "str",  # '0', 'a' public, 'b' easter, 'c' christmas
    "customers_expected": "float?",  # upstream forecast, sometimes missing
    "temperature": "float?",
    "fuel_price": "float?",
    "cpi": "float?",
    "unemployment": "float?",
    "markdown_total": "float?",  # Walmart-style markdown spend, often absent
    "days_since_prev_promo": "int?",
    "region": "str",
    "competitor_promo": "bool",
}

# Fitted constants a real pipeline would bake into the generated SQL at fit
# time: global-prior fallbacks for encoding misses and imputation fills.
PRIOR_MEAN_SALES = 5773.0
PRIOR_MEAN_CUSTOMERS = 633.0
PRIOR_SPC = 9.12
COMP_DIST_FILL = 75000.0
PRIOR_FUEL = 3.45
PRIOR_UNEMP = 8.0
DAYS_SINCE_PROMO_FILL = 30

N_STORES = 60  # serving traffic hits 1..60
DIM_COVERED = 54  # store.csv knows 1..54 (55..60 are new stores -> join miss)
STATS_COVERED = 50  # encodings fitted for 1..50 (51..60 lack history)

_COMP_MONTHS = (
    "(__THIS__.year - store_dim.competition_open_since_year) * 12"
    " + (__THIS__.month - store_dim.competition_open_since_month)"
)
_PROMO2_WEEKS = (
    "(__THIS__.year - store_dim.promo2_since_year) * 52"
    " + (__THIS__.week_of_year - store_dim.promo2_since_week)"
)

SQL = f"""
SELECT
    __THIS__.record_id AS record_id,
    __THIS__.store_id AS store_id,
    CASE WHEN __THIS__.is_open THEN 1 ELSE 0 END AS is_open_flag,
    CASE WHEN __THIS__.promo THEN 1 ELSE 0 END AS promo_flag,
    CASE WHEN __THIS__.promo2 THEN 1 ELSE 0 END AS promo2_flag,
    CASE WHEN __THIS__.school_holiday THEN 1 ELSE 0 END AS school_holiday_flag,
    CASE WHEN __THIS__.state_holiday = 'a' THEN 1 ELSE 0 END AS state_hol_public,
    CASE WHEN __THIS__.state_holiday = 'b' THEN 1 ELSE 0 END AS state_hol_easter,
    CASE WHEN __THIS__.state_holiday = 'c' THEN 1 ELSE 0 END AS state_hol_christmas,
    CASE WHEN __THIS__.state_holiday IN ('a', 'b', 'c') THEN 1 ELSE 0 END
        AS any_state_holiday,
    CASE WHEN __THIS__.state_holiday <> '0' AND __THIS__.school_holiday
         THEN 1 ELSE 0 END AS holiday_x_school,
    CASE WHEN __THIS__.day_of_week = 1 THEN 1 ELSE 0 END AS dow_mon,
    CASE WHEN __THIS__.day_of_week = 2 THEN 1 ELSE 0 END AS dow_tue,
    CASE WHEN __THIS__.day_of_week = 3 THEN 1 ELSE 0 END AS dow_wed,
    CASE WHEN __THIS__.day_of_week = 4 THEN 1 ELSE 0 END AS dow_thu,
    CASE WHEN __THIS__.day_of_week = 5 THEN 1 ELSE 0 END AS dow_fri,
    CASE WHEN __THIS__.day_of_week = 6 THEN 1 ELSE 0 END AS dow_sat,
    CASE WHEN __THIS__.day_of_week = 7 THEN 1 ELSE 0 END AS dow_sun,
    CASE WHEN __THIS__.day_of_week >= 6 THEN 1 ELSE 0 END AS is_weekend,
    CASE WHEN __THIS__.promo AND __THIS__.day_of_week <= 5
         THEN 1 ELSE 0 END AS promo_weekday,
    CASE WHEN __THIS__.competitor_promo AND NOT __THIS__.promo
         THEN 1 ELSE 0 END AS comp_promo_pressure,
    CASE WHEN store_dim.store_type = 'a' THEN 1
         WHEN store_dim.store_type = 'b' THEN 2
         WHEN store_dim.store_type = 'c' THEN 3
         WHEN store_dim.store_type = 'd' THEN 4
         ELSE 0 END AS store_type_ord,
    CASE WHEN store_dim.assortment = 'a' THEN 1
         WHEN store_dim.assortment = 'b' THEN 2
         WHEN store_dim.assortment = 'c' THEN 3
         ELSE 0 END AS assortment_ord,
    CASE WHEN NOT __THIS__.promo THEN 0
         WHEN store_dim.store_type = 'a' THEN 1
         WHEN store_dim.store_type = 'b' THEN 2
         WHEN store_dim.store_type = 'c' THEN 3
         WHEN store_dim.store_type = 'd' THEN 4
         ELSE 0 END AS promo_x_store_type,
    COALESCE(store_dim.competition_distance, {COMP_DIST_FILL}) AS comp_distance,
    1.0 / (1.0 + COALESCE(store_dim.competition_distance, {COMP_DIST_FILL}))
        AS comp_distance_inv,
    greatest(0, {_COMP_MONTHS}) AS comp_open_months,
    CASE WHEN NOT __THIS__.promo2 THEN 0
         ELSE greatest(0, {_PROMO2_WEEKS}) END AS promo2_active_weeks,
    COALESCE(store_stats.mean_sales, {PRIOR_MEAN_SALES}) AS store_mean_sales,
    COALESCE(store_stats.mean_customers, {PRIOR_MEAN_CUSTOMERS})
        AS store_mean_customers,
    COALESCE(store_stats.sales_per_customer, {PRIOR_SPC}) AS store_spc,
    COALESCE(region_stats.region_factor, 1.0) AS region_factor,
    COALESCE(store_stats.mean_sales, {PRIOR_MEAN_SALES})
        * COALESCE(dow_stats.sales_factor, 1.0)
        * COALESCE(month_stats.sales_factor, 1.0)
        * COALESCE(region_stats.region_factor, 1.0)
        * CASE WHEN __THIS__.promo THEN 1.22 ELSE 1.0 END AS expected_sales,
    COALESCE(store_stats.mean_customers, {PRIOR_MEAN_CUSTOMERS})
        * COALESCE(dow_stats.customers_factor, 1.0)
        * COALESCE(month_stats.customers_factor, 1.0) AS expected_customers,
    __THIS__.customers_expected
        / NULLIF(COALESCE(store_stats.mean_customers, {PRIOR_MEAN_CUSTOMERS}), 0.0)
        AS customers_ratio,
    __THIS__.customers_expected
        * COALESCE(store_stats.sales_per_customer, {PRIOR_SPC}) AS forecast_sales,
    COALESCE(__THIS__.markdown_total, 0.0) AS markdown_filled,
    CASE WHEN __THIS__.markdown_total IS NULL THEN 0 ELSE 1 END AS has_markdown,
    COALESCE(__THIS__.markdown_total, 0.0)
        / NULLIF(__THIS__.customers_expected, 0.0) AS markdown_per_customer,
    (__THIS__.temperature - 15.0) / 10.0 AS temp_norm,
    __THIS__.cpi / NULLIF(__THIS__.unemployment, 0.0) AS cpi_unemployment_ratio,
    COALESCE(__THIS__.fuel_price, {PRIOR_FUEL})
        * COALESCE(__THIS__.unemployment, {PRIOR_UNEMP}) AS econ_pressure,
    COALESCE(__THIS__.days_since_prev_promo, {DAYS_SINCE_PROMO_FILL})
        AS days_since_promo_filled,
    CASE WHEN __THIS__.promo
              AND COALESCE(__THIS__.days_since_prev_promo, {DAYS_SINCE_PROMO_FILL}) < 7
         THEN 1 ELSE 0 END AS promo_fatigue,
    ln(1.0 + COALESCE(store_dim.competition_distance, {COMP_DIST_FILL}))
        AS comp_distance_log,
    floor(greatest(0, {_COMP_MONTHS}) / 12.0) AS comp_open_years,
    trunc((__THIS__.day_of_month - 1) / 7.0) + 1.0 AS week_of_month,
    sin(2.0 * pi() * __THIS__.week_of_year / 52.0) AS woy_sin,
    cos(2.0 * pi() * __THIS__.week_of_year / 52.0) AS woy_cos,
    sin(2.0 * pi() * __THIS__.day_of_week / 7.0) AS dow_sin,
    cos(2.0 * pi() * __THIS__.day_of_week / 7.0) AS dow_cos,
    CASE WHEN __THIS__.promo2 AND __THIS__.month IN (1, 4, 7, 10)
         THEN 1 ELSE 0 END AS promo2_interval_month,
    CASE WHEN __THIS__.month IN (11, 12) THEN 1 ELSE 0 END AS peak_season,
    ln(1.0 + COALESCE(__THIS__.markdown_total, 0.0)) AS markdown_log,
    ln(1.0 + COALESCE(store_stats.mean_sales, {PRIOR_MEAN_SALES}))
        AS store_mean_sales_log,
    least(CASE WHEN NOT __THIS__.promo2 THEN 0
               ELSE greatest(0, {_PROMO2_WEEKS}) END, 156) AS promo2_weeks_capped
FROM __THIS__
LEFT JOIN store_dim ON __THIS__.store_id = store_dim.store_id
LEFT JOIN store_stats ON __THIS__.store_id = store_stats.store_id
LEFT JOIN dow_stats ON __THIS__.day_of_week = dow_stats.day_of_week
LEFT JOIN month_stats ON __THIS__.month = month_stats.month
LEFT JOIN region_stats ON __THIS__.region = region_stats.region
"""

_REGIONS = ["north", "south", "east", "west"]


def make_statics(seed: int) -> dict[str, pa.Table]:
    rng = random.Random(seed)

    dim_rows = []
    for sid in range(1, DIM_COVERED + 1):
        enrolled = rng.random() < 0.5
        dim_rows.append(
            {
                "store_id": sid,
                "store_type": rng.choices("abcd", weights=[54, 2, 13, 31])[0],
                "assortment": rng.choices("abc", weights=[53, 1, 46])[0],
                # prepare-time imputation keeps the static NULL-free, like the
                # classic fill of missing CompetitionDistance with a far value
                "competition_distance": round(
                    math.exp(rng.uniform(math.log(30.0), math.log(75000.0))), 1
                ),
                "competition_open_since_month": rng.randint(1, 12),
                "competition_open_since_year": rng.randint(2000, 2015),
                # sentinel far-future start for stores not enrolled in Promo2
                # (clamps to 0 active weeks)
                "promo2_since_week": rng.randint(1, 52) if enrolled else 1,
                "promo2_since_year": rng.randint(2009, 2015) if enrolled else 2099,
            }
        )

    stats_rows = []
    for sid in range(1, STATS_COVERED + 1):
        mean_customers = round(rng.uniform(300.0, 1400.0), 2)
        spc = round(rng.uniform(6.5, 11.5), 4)
        mean_sales = round(mean_customers * spc, 2)
        stats_rows.append(
            {
                "store_id": sid,
                "mean_sales": mean_sales,
                "mean_customers": mean_customers,
                "sales_per_customer": round(mean_sales / mean_customers, 4),
            }
        )

    dow_base = [1.15, 0.98, 0.95, 0.96, 1.02, 1.08, 0.45]
    dow_rows = [
        {
            "day_of_week": d + 1,
            "sales_factor": round(dow_base[d] + rng.uniform(-0.02, 0.02), 4),
            "customers_factor": round(dow_base[d] + rng.uniform(-0.03, 0.03), 4),
        }
        for d in range(7)
    ]

    month_base = [0.96, 0.94, 0.99, 1.0, 1.01, 0.98, 1.02, 0.99, 0.97, 1.0, 1.05, 1.35]
    month_rows = [
        {
            "month": m + 1,
            "sales_factor": round(month_base[m] + rng.uniform(-0.02, 0.02), 4),
            "customers_factor": round(month_base[m] + rng.uniform(-0.03, 0.03), 4),
        }
        for m in range(12)
    ]

    region_base = {"north": 1.04, "south": 0.97, "east": 0.92, "west": 1.07}
    region_rows = [
        {
            "region": r,
            "region_factor": round(region_base[r] + rng.uniform(-0.02, 0.02), 4),
        }
        for r in _REGIONS
    ]

    return {
        "store_dim": pa.Table.from_pylist(
            dim_rows,
            schema=pa.schema(
                [
                    ("store_id", pa.int64()),
                    ("store_type", pa.string()),
                    ("assortment", pa.string()),
                    ("competition_distance", pa.float64()),
                    ("competition_open_since_month", pa.int64()),
                    ("competition_open_since_year", pa.int64()),
                    ("promo2_since_week", pa.int64()),
                    ("promo2_since_year", pa.int64()),
                ]
            ),
        ),
        "store_stats": pa.Table.from_pylist(
            stats_rows,
            schema=pa.schema(
                [
                    ("store_id", pa.int64()),
                    ("mean_sales", pa.float64()),
                    ("mean_customers", pa.float64()),
                    ("sales_per_customer", pa.float64()),
                ]
            ),
        ),
        "dow_stats": pa.Table.from_pylist(
            dow_rows,
            schema=pa.schema(
                [
                    ("day_of_week", pa.int64()),
                    ("sales_factor", pa.float64()),
                    ("customers_factor", pa.float64()),
                ]
            ),
        ),
        "month_stats": pa.Table.from_pylist(
            month_rows,
            schema=pa.schema(
                [
                    ("month", pa.int64()),
                    ("sales_factor", pa.float64()),
                    ("customers_factor", pa.float64()),
                ]
            ),
        ),
        "region_stats": pa.Table.from_pylist(
            region_rows,
            schema=pa.schema(
                [("region", pa.string()), ("region_factor", pa.float64())]
            ),
        ),
    }


def make_rows(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        dow = rng.randint(1, 7)
        month = rng.randint(1, 12)
        state_holiday = rng.choices(["0", "a", "b", "c"], weights=[91, 5, 2, 2])[0]
        open_p = 0.1 if state_holiday != "0" else (0.6 if dow == 7 else 0.97)
        rows.append(
            {
                "record_id": 9_000_000 + i,
                "store_id": rng.randint(1, N_STORES),
                "day_of_week": dow,
                "day_of_month": rng.randint(1, 28),
                "month": month,
                "year": rng.choice([2014, 2015]),
                "week_of_year": (month - 1) * 4 + rng.randint(1, 4),
                "is_open": rng.random() < open_p,
                "promo": dow <= 5 and rng.random() < 0.45,
                "promo2": rng.random() < 0.5,
                "school_holiday": rng.random() < 0.18,
                "state_holiday": state_holiday,
                "customers_expected": (
                    None
                    if rng.random() < 0.12
                    else round(rng.uniform(200.0, 1600.0), 1)
                ),
                "temperature": (
                    None if rng.random() < 0.05 else round(rng.uniform(-5.0, 35.0), 1)
                ),
                "fuel_price": (
                    None if rng.random() < 0.05 else round(rng.uniform(2.4, 4.5), 3)
                ),
                "cpi": None
                if rng.random() < 0.08
                else round(rng.uniform(126.0, 228.0), 2),
                "unemployment": (
                    None if rng.random() < 0.08 else round(rng.uniform(3.8, 14.3), 3)
                ),
                "markdown_total": (
                    None
                    if rng.random() < 0.55
                    else round(rng.uniform(50.0, 20000.0), 2)
                ),
                "days_since_prev_promo": (
                    None if rng.random() < 0.10 else rng.randint(0, 60)
                ),
                "region": rng.choice(_REGIONS),
                "competitor_promo": rng.random() < 0.3,
            }
        )
    return rows


def handcrafted(statics: dict[str, pa.Table]) -> Callable[[dict], dict]:
    """What a competent engineer hand-writes for a Python microservice:
    plain dict lookups prepared once, then a per-row closure computing the
    identical features with SQL NULL semantics (join miss => None
    propagation through arithmetic, CASE falls through on unknown)."""
    dim = {r["store_id"]: r for r in statics["store_dim"].to_pylist()}
    stats = {r["store_id"]: r for r in statics["store_stats"].to_pylist()}
    dow_f = {r["day_of_week"]: r for r in statics["dow_stats"].to_pylist()}
    month_f = {r["month"]: r for r in statics["month_stats"].to_pylist()}
    region_f = {r["region"]: r for r in statics["region_stats"].to_pylist()}

    def fn(row: dict) -> dict:
        d = dim.get(row["store_id"])
        s = stats.get(row["store_id"])
        w = dow_f[row["day_of_week"]]  # full coverage by construction
        m = month_f[row["month"]]
        g = region_f[row["region"]]

        dow = row["day_of_week"]
        promo = row["promo"]
        hol = row["state_holiday"]

        st = d["store_type"] if d is not None else None
        sort = d["assortment"] if d is not None else None
        comp_dist = d["competition_distance"] if d is not None else COMP_DIST_FILL

        # greatest(0, x) with NULL-ignoring greatest: join-miss NULL months
        # collapse to the 0 floor, negatives clamp to 0.
        if d is None:
            comp_open_months = 0
        else:
            comp_open_months = max(
                0,
                (row["year"] - d["competition_open_since_year"]) * 12
                + (row["month"] - d["competition_open_since_month"]),
            )

        if not row["promo2"] or d is None:
            promo2_active_weeks = 0
        else:
            promo2_active_weeks = max(
                0,
                (row["year"] - d["promo2_since_year"]) * 52
                + (row["week_of_year"] - d["promo2_since_week"]),
            )

        mean_sales = s["mean_sales"] if s is not None else PRIOR_MEAN_SALES
        mean_customers = s["mean_customers"] if s is not None else PRIOR_MEAN_CUSTOMERS
        spc = s["sales_per_customer"] if s is not None else PRIOR_SPC

        ce = row["customers_expected"]
        md = row["markdown_total"]
        md_filled = md if md is not None else 0.0
        t = row["temperature"]
        cpi = row["cpi"]
        unemp = row["unemployment"]
        fuel = row["fuel_price"]
        ds = row["days_since_prev_promo"]
        ds_filled = ds if ds is not None else DAYS_SINCE_PROMO_FILL

        return {
            "record_id": row["record_id"],
            "store_id": row["store_id"],
            "is_open_flag": 1 if row["is_open"] else 0,
            "promo_flag": 1 if promo else 0,
            "promo2_flag": 1 if row["promo2"] else 0,
            "school_holiday_flag": 1 if row["school_holiday"] else 0,
            "state_hol_public": 1 if hol == "a" else 0,
            "state_hol_easter": 1 if hol == "b" else 0,
            "state_hol_christmas": 1 if hol == "c" else 0,
            "any_state_holiday": 1 if hol in ("a", "b", "c") else 0,
            "holiday_x_school": 1 if hol != "0" and row["school_holiday"] else 0,
            "dow_mon": 1 if dow == 1 else 0,
            "dow_tue": 1 if dow == 2 else 0,
            "dow_wed": 1 if dow == 3 else 0,
            "dow_thu": 1 if dow == 4 else 0,
            "dow_fri": 1 if dow == 5 else 0,
            "dow_sat": 1 if dow == 6 else 0,
            "dow_sun": 1 if dow == 7 else 0,
            "is_weekend": 1 if dow >= 6 else 0,
            "promo_weekday": 1 if promo and dow <= 5 else 0,
            "comp_promo_pressure": 1 if row["competitor_promo"] and not promo else 0,
            # None == 'x' is False in Python, matching CASE falling through
            # NULL comparisons to ELSE
            "store_type_ord": (
                1
                if st == "a"
                else 2
                if st == "b"
                else 3
                if st == "c"
                else 4
                if st == "d"
                else 0
            ),
            "assortment_ord": (
                1 if sort == "a" else 2 if sort == "b" else 3 if sort == "c" else 0
            ),
            "promo_x_store_type": (
                0
                if not promo
                else 1
                if st == "a"
                else 2
                if st == "b"
                else 3
                if st == "c"
                else 4
                if st == "d"
                else 0
            ),
            "comp_distance": comp_dist,
            "comp_distance_inv": 1.0 / (1.0 + comp_dist),
            "comp_open_months": comp_open_months,
            "promo2_active_weeks": promo2_active_weeks,
            "store_mean_sales": mean_sales,
            "store_mean_customers": mean_customers,
            "store_spc": spc,
            "region_factor": g["region_factor"],
            "expected_sales": mean_sales
            * w["sales_factor"]
            * m["sales_factor"]
            * g["region_factor"]
            * (1.22 if promo else 1.0),
            "expected_customers": mean_customers
            * w["customers_factor"]
            * m["customers_factor"],
            "customers_ratio": (
                None if ce is None or mean_customers == 0.0 else ce / mean_customers
            ),
            "forecast_sales": None if ce is None else ce * spc,
            "markdown_filled": md_filled,
            "has_markdown": 0 if md is None else 1,
            "markdown_per_customer": (
                None if ce is None or ce == 0.0 else md_filled / ce
            ),
            "temp_norm": None if t is None else (t - 15.0) / 10.0,
            "cpi_unemployment_ratio": (
                None if cpi is None or unemp is None or unemp == 0.0 else cpi / unemp
            ),
            "econ_pressure": (fuel if fuel is not None else PRIOR_FUEL)
            * (unemp if unemp is not None else PRIOR_UNEMP),
            "days_since_promo_filled": ds_filled,
            "promo_fatigue": 1 if promo and ds_filled < 7 else 0,
            # -- wave-1 builtin features (the famous-solution versions) --
            # ln(1 + CompetitionDistance): the classic Rossmann skew fix
            "comp_distance_log": math.log(1.0 + comp_dist),
            # competition age bucketed to whole years; floor(double) is a
            # double in SQL, so the twin must produce a float too
            "comp_open_years": float(math.floor(comp_open_months / 12.0)),
            # trunc((day-1)/7) + 1: Rossmann week-of-month, kept as the
            # double the SQL trunc() yields (1.0 .. 4.0)
            "week_of_month": math.trunc((row["day_of_month"] - 1) / 7.0) + 1.0,
            # cyclical year/week encodings; evaluation order mirrors the SQL
            # left-to-right ((2.0 * pi) * k) / period exactly
            "woy_sin": math.sin(2.0 * math.pi * row["week_of_year"] / 52.0),
            "woy_cos": math.cos(2.0 * math.pi * row["week_of_year"] / 52.0),
            "dow_sin": math.sin(2.0 * math.pi * dow / 7.0),
            "dow_cos": math.cos(2.0 * math.pi * dow / 7.0),
            # canonical 'Jan,Apr,Jul,Oct' PromoInterval renewal-month flag
            "promo2_interval_month": (
                1 if row["promo2"] and row["month"] in (1, 4, 7, 10) else 0
            ),
            "peak_season": 1 if row["month"] in (11, 12) else 0,
            "markdown_log": math.log(1.0 + md_filled),
            "store_mean_sales_log": math.log(1.0 + mean_sales),
            # least(x, 156): promo2 effect saturates after ~3 years
            "promo2_weeks_capped": min(promo2_active_weeks, 156),
        }

    return fn
