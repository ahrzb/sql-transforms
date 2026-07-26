"""house_prices — Ames wide-arithmetic serving scenario for the SQL specializer.

The canonical Kaggle House Prices feature set, expressed as the serve-time
query a fitted pipeline reduces to: scalar expressions over the row plus
LEFT JOINs to prepare-time encoding tables. This is the wide-arith stress
case: 43 input columns, 54 output features, almost all of them arithmetic
and CASE over the row.

Originally authored before the wave-1 math/string builtins landed, so the
public-kernel log1p skew fixes, sqrt, decade bins, clamps, IN-set flags and
cyclical month encoding were dropped. They are back now that the engine
supports ln/sqrt/floor/least/greatest/IN/sin/cos/pi: log1p(TotalSF/LotArea/
GrLivArea) as ln(1.0 + x), sqrt(GrLivArea), YearBuilt decade via
floor(x/10)*10, garage-age clamp via greatest, Ex/Gd quality IN-sets, the
rich-neighborhood and PUD MSSubClass membership flags, and MoSold sin/cos.

Realistic serving gotchas baked into the row distributions:
- MSSubClass 150 and the GrnHill/Landmrk neighborhoods exist only outside
  the Kaggle train set -> LEFT JOIN misses at serve time (COALESCE to the
  train-global mean price, the standard target-encoding fallback).
- The test set has NA garage/basement numerics the train set never had
  (famous rows 2121/2189/2577) -> None on nullable columns, propagating
  through arithmetic exactly like SQL NULL.
- LotFrontage is missing ~17% of the time -> imputed from the fitted
  per-neighborhood median via the join table (the classic trick).
"""

import math
import random
from collections.abc import Callable

import pyarrow as pa

NAME = "house_prices"
KAGGLE = (
    "House Prices: Advanced Regression Techniques (Ames) — the canonical "
    "public-kernel feature set: TotalSF/TotalBath sums, age features, "
    "Ex..Po ordinal quality maps, porch total, has_* flags, nullif-guarded "
    "lot ratios, qual*cond crosses, neighborhood-median LotFrontage "
    "imputation, Neighborhood/MSSubClass target+frequency encodings "
    "served as join tables, log1p skew fixes, decade bins, quality/"
    "neighborhood/PUD membership flags, and cyclical MoSold encoding."
)

N_INPUT_COLS = 43
N_OUTPUT_COLS = 54

ROW_SCHEMA = {
    "id": "int",
    "ms_sub_class": "int",
    "lot_frontage": "float?",
    "lot_area": "int",
    "neighborhood": "str",
    "overall_qual": "int",
    "overall_cond": "int",
    "year_built": "int",
    "year_remod_add": "int",
    "exter_qual": "str",
    "exter_cond": "str",
    "mas_vnr_area": "float?",
    "bsmt_qual": "str?",
    "bsmt_fin_sf1": "float?",
    "bsmt_unf_sf": "float?",
    "total_bsmt_sf": "float?",
    "heating_qc": "str",
    "central_air": "str",
    "first_flr_sf": "int",
    "second_flr_sf": "int",
    "gr_liv_area": "int",
    "bsmt_full_bath": "int?",
    "bsmt_half_bath": "int?",
    "full_bath": "int",
    "half_bath": "int",
    "bedroom_abv_gr": "int",
    "kitchen_abv_gr": "int",
    "kitchen_qual": "str?",
    "tot_rms_abv_grd": "int",
    "fireplaces": "int",
    "fireplace_qu": "str?",
    "garage_yr_blt": "float?",
    "garage_cars": "int?",
    "garage_area": "float?",
    "garage_qual": "str?",
    "wood_deck_sf": "int",
    "open_porch_sf": "int",
    "enclosed_porch": "int",
    "three_ssn_porch": "int",
    "screen_porch": "int",
    "pool_area": "int",
    "mo_sold": "int",
    "yr_sold": "int",
}

# (name, train mean price, train count, median lot frontage) — the 25 Kaggle
# train neighborhoods. GrnHill/Landmrk exist in full Ames but not in train:
# the unseen-category LEFT JOIN miss at serve time.
_NBHDS = [
    ("NAmes", 145847, 225, 73.0),
    ("CollgCr", 197966, 150, 70.0),
    ("OldTown", 128225, 113, 60.0),
    ("Edwards", 128220, 100, 66.0),
    ("Somerst", 225380, 86, 73.5),
    ("Gilbert", 192854, 79, 64.0),
    ("NridgHt", 316271, 77, 92.0),
    ("Sawyer", 136793, 74, 71.0),
    ("NWAmes", 189050, 73, 80.0),
    ("SawyerW", 186556, 59, 66.5),
    ("BrkSide", 124834, 58, 52.0),
    ("Crawfor", 210625, 51, 70.0),
    ("Mitchel", 156270, 49, 73.0),
    ("NoRidge", 335295, 41, 91.0),
    ("Timber", 242247, 38, 85.0),
    ("IDOTRR", 100124, 37, 60.0),
    ("ClearCr", 212565, 28, 80.0),
    ("StoneBr", 310499, 25, 61.5),
    ("SWISU", 142591, 25, 60.0),
    ("Blmngtn", 194871, 17, 43.0),
    ("MeadowV", 98576, 17, 21.0),
    ("BrDale", 104494, 16, 21.0),
    ("Veenker", 238772, 11, 68.0),
    ("NPkVill", 142694, 9, 24.0),
    ("Blueste", 137500, 2, 24.0),
]
_UNSEEN_NBHDS = ["GrnHill", "Landmrk"]

# (MSSubClass, train mean price, train count). 150 is the famous class that
# appears only in the test set — the unseen join miss.
_SUBCLASSES = [
    (20, 185224, 536),
    (60, 240403, 299),
    (50, 143302, 144),
    (120, 200779, 87),
    (30, 95829, 69),
    (160, 138647, 63),
    (70, 166772, 60),
    (80, 169736, 58),
    (90, 133541, 52),
    (190, 129613, 30),
    (85, 147810, 20),
    (75, 192437, 16),
    (45, 108591, 12),
    (180, 102300, 10),
    (40, 156125, 4),
]
_UNSEEN_SUBCLASS = 150

_QUAL_LEVELS = ["Ex", "Gd", "TA", "Fa", "Po"]


def make_statics(seed: int) -> dict[str, pa.Table]:
    rng = random.Random(seed)
    names, means, freqs, medfr = [], [], [], []
    for name, base_mean, count, med in _NBHDS:
        names.append(name)
        means.append(round(base_mean * rng.uniform(0.97, 1.03), 2))
        freqs.append(round(count / 1460.0, 6))
        medfr.append(med)
    nbhd = pa.table(
        {
            "nbhd": pa.array(names, type=pa.string()),
            "mean_price": pa.array(means, type=pa.float64()),
            "freq": pa.array(freqs, type=pa.float64()),
            "median_frontage": pa.array(medfr, type=pa.float64()),
        }
    )
    classes, sc_means, sc_freqs = [], [], []
    for cls, base_mean, count in _SUBCLASSES:
        classes.append(cls)
        sc_means.append(round(base_mean * rng.uniform(0.97, 1.03), 2))
        sc_freqs.append(round(count / 1460.0, 6))
    sub = pa.table(
        {
            "sub_class": pa.array(classes, type=pa.int64()),
            "mean_price": pa.array(sc_means, type=pa.float64()),
            "freq": pa.array(sc_freqs, type=pa.float64()),
        }
    )
    return {"nbhd_price_enc": nbhd, "subclass_enc": sub}


def make_rows(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed)
    nb_names = [t[0] for t in _NBHDS]
    nb_wts = [t[2] for t in _NBHDS]
    sc_names = [t[0] for t in _SUBCLASSES]
    sc_wts = [t[2] for t in _SUBCLASSES]
    rows = []
    for i in range(n):
        yr_sold = rng.randint(2006, 2010)
        year_built = yr_sold if rng.random() < 0.06 else rng.randint(1872, yr_sold)
        if rng.random() < 0.45 and year_built < yr_sold:
            year_remod_add = rng.randint(max(1950, year_built), yr_sold)
        else:
            year_remod_add = max(1950, year_built)

        first_flr_sf = rng.randint(334, 2600)
        second_flr_sf = 0 if rng.random() < 0.55 else rng.randint(300, 1200)
        gr_liv_area = first_flr_sf + second_flr_sf

        u = rng.random()
        if u < 0.01:  # the test-set row with every basement field missing
            bsmt_qual = bsmt_fin_sf1 = bsmt_unf_sf = total_bsmt_sf = None
            bsmt_full_bath = bsmt_half_bath = None
        elif u < 0.05:  # no basement
            bsmt_qual = None
            bsmt_fin_sf1 = bsmt_unf_sf = total_bsmt_sf = 0.0
            bsmt_full_bath = bsmt_half_bath = 0
        else:
            total_bsmt_sf = float(rng.randint(300, first_flr_sf + 200))
            bsmt_fin_sf1 = float(rng.randint(0, int(total_bsmt_sf)))
            bsmt_unf_sf = total_bsmt_sf - bsmt_fin_sf1
            bsmt_qual = rng.choices(_QUAL_LEVELS, weights=[8, 40, 44, 6, 2])[0]
            bsmt_full_bath = rng.choices([0, 1, 2], weights=[58, 39, 3])[0]
            bsmt_half_bath = rng.choices([0, 1], weights=[94, 6])[0]

        u = rng.random()
        if u < 0.055:  # no garage
            garage_yr_blt = garage_qual = None
            garage_cars, garage_area = 0, 0.0
        elif u < 0.065:  # the famous test row 2577: garage fields missing
            garage_yr_blt = garage_qual = garage_cars = garage_area = None
        else:
            garage_yr_blt = float(rng.randint(year_built, yr_sold))
            garage_cars = rng.choices([1, 2, 3, 4], weights=[25, 55, 18, 2])[0]
            garage_area = float(rng.randint(200, 350) * garage_cars)
            garage_qual = rng.choices(_QUAL_LEVELS, weights=[1, 2, 90, 5, 2])[0]

        fireplaces = rng.choices([0, 1, 2, 3], weights=[47, 43, 9, 1])[0]
        bedroom_abv_gr = rng.choices(
            [0, 1, 2, 3, 4, 5, 6], weights=[1, 4, 24, 50, 17, 3, 1]
        )[0]
        kitchen_abv_gr = rng.choices([1, 2], weights=[95, 5])[0]
        full_bath = rng.choices([1, 2, 3], weights=[45, 50, 5])[0]
        half_bath = rng.choices([0, 1, 2], weights=[62, 36, 2])[0]

        rows.append(
            {
                "id": 1461 + i,  # Kaggle test-set ids start at 1461
                "ms_sub_class": _UNSEEN_SUBCLASS
                if rng.random() < 0.01
                else rng.choices(sc_names, weights=sc_wts)[0],
                "lot_frontage": None
                if rng.random() < 0.17
                else float(rng.randint(21, 150)),
                "lot_area": min(215245, max(1300, int(rng.lognormvariate(9.2, 0.45)))),
                "neighborhood": rng.choice(_UNSEEN_NBHDS)
                if rng.random() < 0.02
                else rng.choices(nb_names, weights=nb_wts)[0],
                "overall_qual": min(10, max(1, round(rng.gauss(6.1, 1.4)))),
                "overall_cond": min(10, max(1, round(rng.gauss(5.6, 1.1)))),
                "year_built": year_built,
                "year_remod_add": year_remod_add,
                "exter_qual": rng.choices(_QUAL_LEVELS, weights=[4, 33, 60, 2, 1])[0],
                "exter_cond": rng.choices(_QUAL_LEVELS, weights=[1, 10, 86, 2, 1])[0],
                "mas_vnr_area": None
                if rng.random() < 0.005
                else (0.0 if rng.random() < 0.6 else float(rng.randint(50, 1600))),
                "bsmt_qual": bsmt_qual,
                "bsmt_fin_sf1": bsmt_fin_sf1,
                "bsmt_unf_sf": bsmt_unf_sf,
                "total_bsmt_sf": total_bsmt_sf,
                "heating_qc": rng.choices(_QUAL_LEVELS, weights=[51, 16, 29, 3, 1])[0],
                "central_air": "Y" if rng.random() < 0.93 else "N",
                "first_flr_sf": first_flr_sf,
                "second_flr_sf": second_flr_sf,
                "gr_liv_area": gr_liv_area,
                "bsmt_full_bath": bsmt_full_bath,
                "bsmt_half_bath": bsmt_half_bath,
                "full_bath": full_bath,
                "half_bath": half_bath,
                "bedroom_abv_gr": bedroom_abv_gr,
                "kitchen_abv_gr": kitchen_abv_gr,
                "kitchen_qual": None
                if rng.random() < 0.01
                else rng.choices(_QUAL_LEVELS, weights=[7, 40, 50, 2, 1])[0],
                "tot_rms_abv_grd": bedroom_abv_gr + kitchen_abv_gr + rng.randint(2, 5),
                "fireplaces": fireplaces,
                "fireplace_qu": None
                if fireplaces == 0
                else rng.choices(_QUAL_LEVELS, weights=[3, 49, 41, 4, 3])[0],
                "garage_yr_blt": garage_yr_blt,
                "garage_cars": garage_cars,
                "garage_area": garage_area,
                "garage_qual": garage_qual,
                "wood_deck_sf": 0 if rng.random() < 0.48 else rng.randint(20, 800),
                "open_porch_sf": 0 if rng.random() < 0.45 else rng.randint(10, 500),
                "enclosed_porch": 0 if rng.random() < 0.86 else rng.randint(20, 400),
                "three_ssn_porch": 0 if rng.random() < 0.98 else rng.randint(100, 300),
                "screen_porch": 0 if rng.random() < 0.92 else rng.randint(80, 400),
                "pool_area": 0 if rng.random() < 0.995 else rng.randint(400, 800),
                "mo_sold": rng.choices(
                    list(range(1, 13)),
                    weights=[4, 4, 7, 9, 14, 17, 15, 8, 6, 6, 6, 4],
                )[0],
                "yr_sold": yr_sold,
            }
        )
    return rows


def _ord_case(col: str) -> str:
    """Ex..Po -> 5..1, anything else (incl. NULL: no branch matches) -> 0."""
    branches = " ".join(
        f"WHEN {col} = '{lvl}' THEN {5 - i}" for i, lvl in enumerate(_QUAL_LEVELS)
    )
    return f"CASE {branches} ELSE 0 END"


SQL = f"""SELECT
  id,
  coalesce(total_bsmt_sf, 0.0) + first_flr_sf + second_flr_sf AS total_sf,
  full_bath + CAST(0.5 AS DOUBLE) * half_bath + coalesce(bsmt_full_bath, 0)
    + CAST(0.5 AS DOUBLE) * coalesce(bsmt_half_bath, 0) AS total_bath,
  yr_sold - year_built AS house_age,
  yr_sold - year_remod_add AS remod_age,
  yr_sold - garage_yr_blt AS garage_age,
  CASE WHEN yr_sold = year_built THEN 1 ELSE 0 END AS is_new,
  CASE WHEN year_remod_add > year_built THEN 1 ELSE 0 END AS is_remodeled,
  {_ord_case("exter_qual")} AS exter_qual_ord,
  {_ord_case("exter_cond")} AS exter_cond_ord,
  CASE WHEN kitchen_qual IS NULL THEN 3 WHEN kitchen_qual = 'Ex' THEN 5
       WHEN kitchen_qual = 'Gd' THEN 4 WHEN kitchen_qual = 'TA' THEN 3
       WHEN kitchen_qual = 'Fa' THEN 2 WHEN kitchen_qual = 'Po' THEN 1
       ELSE 0 END AS kitchen_qual_ord,
  {_ord_case("heating_qc")} AS heating_qc_ord,
  {_ord_case("bsmt_qual")} AS bsmt_qual_ord,
  {_ord_case("fireplace_qu")} AS fireplace_qu_ord,
  {_ord_case("garage_qual")} AS garage_qual_ord,
  wood_deck_sf + open_porch_sf + enclosed_porch + three_ssn_porch
    + screen_porch AS porch_total,
  CASE WHEN pool_area > 0 THEN 1 ELSE 0 END AS has_pool,
  CASE WHEN coalesce(garage_area, 0.0) > 0.0 THEN 1 ELSE 0 END AS has_garage,
  CASE WHEN coalesce(total_bsmt_sf, 0.0) > 0.0 THEN 1 ELSE 0 END AS has_bsmt,
  CASE WHEN fireplaces > 0 THEN 1 ELSE 0 END AS has_fireplace,
  CASE WHEN second_flr_sf > 0 THEN 1 ELSE 0 END AS has_2nd_floor,
  CASE WHEN coalesce(mas_vnr_area, 0.0) > 0.0 THEN 1 ELSE 0 END AS has_mas_vnr,
  CASE WHEN central_air = 'Y' THEN 1 ELSE 0 END AS central_air_flag,
  CASE WHEN kitchen_abv_gr > 1 THEN 1 ELSE 0 END AS multi_kitchen,
  coalesce(lot_frontage, nbhd_price_enc.median_frontage, 69.0) AS lot_frontage_filled,
  lot_frontage / nullif(lot_area, 0) AS lot_frontage_ratio,
  CAST(gr_liv_area AS DOUBLE) / nullif(lot_area, 0) AS liv_lot_ratio,
  CAST(gr_liv_area AS DOUBLE) / nullif(tot_rms_abv_grd, 0) AS sf_per_room,
  CAST(bedroom_abv_gr AS DOUBLE) / nullif(full_bath + half_bath, 0) AS bed_bath_ratio,
  bsmt_fin_sf1 / nullif(total_bsmt_sf, 0.0) AS bsmt_fin_ratio,
  bsmt_unf_sf / nullif(total_bsmt_sf, 0.0) AS bsmt_unf_ratio,
  garage_area / nullif(garage_cars, 0) AS garage_area_per_car,
  overall_qual * overall_cond AS qual_cond_cross,
  overall_qual * overall_qual AS overall_qual_sq,
  overall_qual * gr_liv_area AS qual_sf_cross,
  coalesce(nbhd_price_enc.mean_price, 180921.0) AS nbhd_price,
  nbhd_price_enc.mean_price AS nbhd_price_raw,
  coalesce(nbhd_price_enc.freq, 0.0) AS nbhd_freq,
  coalesce(subclass_enc.mean_price, 180921.0) AS subclass_price,
  coalesce(subclass_enc.freq, 0.0) AS subclass_freq,
  CASE WHEN mo_sold >= 3 AND mo_sold <= 5 THEN 1
       WHEN mo_sold >= 6 AND mo_sold <= 8 THEN 2
       WHEN mo_sold >= 9 AND mo_sold <= 11 THEN 3
       ELSE 0 END AS season_sold,
  CASE WHEN mo_sold >= 5 AND mo_sold <= 7 THEN 1 ELSE 0 END AS is_peak_season,
  ln(1.0 + (coalesce(total_bsmt_sf, 0.0) + first_flr_sf + second_flr_sf))
    AS log_total_sf,
  ln(1.0 + lot_area) AS log_lot_area,
  ln(1.0 + gr_liv_area) AS log_gr_liv_area,
  sqrt(CAST(gr_liv_area AS DOUBLE)) AS sqrt_gr_liv_area,
  floor(year_built / 10.0) * 10.0 AS decade_built,
  greatest(yr_sold - garage_yr_blt, 0.0) AS garage_age_clamped,
  CASE WHEN exter_qual IN ('Ex', 'Gd') THEN 1 ELSE 0 END AS exter_qual_good,
  CASE WHEN kitchen_qual IN ('Ex', 'Gd') THEN 1 ELSE 0 END AS kitchen_qual_good,
  CASE WHEN neighborhood IN ('NoRidge', 'NridgHt', 'StoneBr') THEN 1 ELSE 0 END
    AS nbhd_rich,
  CASE WHEN ms_sub_class IN (120, 150, 160, 180) THEN 1 ELSE 0 END AS is_pud,
  sin(2.0 * pi() * mo_sold / 12.0) AS mo_sold_sin,
  cos(2.0 * pi() * mo_sold / 12.0) AS mo_sold_cos
FROM __THIS__
LEFT JOIN nbhd_price_enc ON neighborhood = nbhd_price_enc.nbhd
LEFT JOIN subclass_enc ON ms_sub_class = subclass_enc.sub_class"""


def handcrafted(statics: dict[str, pa.Table]) -> Callable[[dict], dict]:
    t = statics["nbhd_price_enc"]
    nb = {
        k: (mp, fr, mf)
        for k, mp, fr, mf in zip(
            t.column("nbhd").to_pylist(),
            t.column("mean_price").to_pylist(),
            t.column("freq").to_pylist(),
            t.column("median_frontage").to_pylist(),
            strict=True,
        )
    }
    t = statics["subclass_enc"]
    sc = {
        k: (mp, fr)
        for k, mp, fr in zip(
            t.column("sub_class").to_pylist(),
            t.column("mean_price").to_pylist(),
            t.column("freq").to_pylist(),
            strict=True,
        )
    }
    q_ord = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}

    def fe(r: dict) -> dict:
        tb = r["total_bsmt_sf"]
        fin = r["bsmt_fin_sf1"]
        unf = r["bsmt_unf_sf"]
        ga = r["garage_area"]
        gc = r["garage_cars"]
        gy = r["garage_yr_blt"]
        lf = r["lot_frontage"]
        la = r["lot_area"]
        mv = r["mas_vnr_area"]
        kq = r["kitchen_qual"]
        bf = r["bsmt_full_bath"]
        bh = r["bsmt_half_bath"]
        oq = r["overall_qual"]
        gla = r["gr_liv_area"]
        bath_den = r["full_bath"] + r["half_bath"]
        nbe = nb.get(r["neighborhood"])
        sce = sc.get(r["ms_sub_class"])
        lff = lf
        if lff is None:
            lff = nbe[2] if nbe is not None else None
        if lff is None:
            lff = 69.0
        tsf = (tb if tb is not None else 0.0) + r["first_flr_sf"] + r["second_flr_sf"]
        return {
            "id": r["id"],
            "total_sf": tsf,
            "total_bath": r["full_bath"]
            + 0.5 * r["half_bath"]
            + (bf if bf is not None else 0)
            + 0.5 * (bh if bh is not None else 0),
            "house_age": r["yr_sold"] - r["year_built"],
            "remod_age": r["yr_sold"] - r["year_remod_add"],
            "garage_age": None if gy is None else r["yr_sold"] - gy,
            "is_new": 1 if r["yr_sold"] == r["year_built"] else 0,
            "is_remodeled": 1 if r["year_remod_add"] > r["year_built"] else 0,
            "exter_qual_ord": q_ord.get(r["exter_qual"], 0),
            "exter_cond_ord": q_ord.get(r["exter_cond"], 0),
            "kitchen_qual_ord": 3 if kq is None else q_ord.get(kq, 0),
            "heating_qc_ord": q_ord.get(r["heating_qc"], 0),
            "bsmt_qual_ord": q_ord.get(r["bsmt_qual"], 0),
            "fireplace_qu_ord": q_ord.get(r["fireplace_qu"], 0),
            "garage_qual_ord": q_ord.get(r["garage_qual"], 0),
            "porch_total": r["wood_deck_sf"]
            + r["open_porch_sf"]
            + r["enclosed_porch"]
            + r["three_ssn_porch"]
            + r["screen_porch"],
            "has_pool": 1 if r["pool_area"] > 0 else 0,
            "has_garage": 1 if (ga if ga is not None else 0.0) > 0.0 else 0,
            "has_bsmt": 1 if (tb if tb is not None else 0.0) > 0.0 else 0,
            "has_fireplace": 1 if r["fireplaces"] > 0 else 0,
            "has_2nd_floor": 1 if r["second_flr_sf"] > 0 else 0,
            "has_mas_vnr": 1 if (mv if mv is not None else 0.0) > 0.0 else 0,
            "central_air_flag": 1 if r["central_air"] == "Y" else 0,
            "multi_kitchen": 1 if r["kitchen_abv_gr"] > 1 else 0,
            "lot_frontage_filled": lff,
            "lot_frontage_ratio": None if lf is None or la == 0 else lf / la,
            "liv_lot_ratio": None if la == 0 else gla / la,
            "sf_per_room": None
            if r["tot_rms_abv_grd"] == 0
            else gla / r["tot_rms_abv_grd"],
            "bed_bath_ratio": None if bath_den == 0 else r["bedroom_abv_gr"] / bath_den,
            "bsmt_fin_ratio": None
            if fin is None or tb is None or tb == 0.0
            else fin / tb,
            "bsmt_unf_ratio": None
            if unf is None or tb is None or tb == 0.0
            else unf / tb,
            "garage_area_per_car": None
            if ga is None or gc is None or gc == 0
            else ga / gc,
            "qual_cond_cross": oq * r["overall_cond"],
            "overall_qual_sq": oq * oq,
            "qual_sf_cross": oq * gla,
            "nbhd_price": nbe[0] if nbe is not None else 180921.0,
            "nbhd_price_raw": nbe[0] if nbe is not None else None,
            "nbhd_freq": nbe[1] if nbe is not None else 0.0,
            "subclass_price": sce[0] if sce is not None else 180921.0,
            "subclass_freq": sce[1] if sce is not None else 0.0,
            "season_sold": 1
            if 3 <= r["mo_sold"] <= 5
            else 2
            if 6 <= r["mo_sold"] <= 8
            else 3
            if 9 <= r["mo_sold"] <= 11
            else 0,
            "is_peak_season": 1 if 5 <= r["mo_sold"] <= 7 else 0,
            "log_total_sf": math.log(1.0 + tsf),
            "log_lot_area": math.log(1.0 + la),
            "log_gr_liv_area": math.log(1.0 + gla),
            "sqrt_gr_liv_area": math.sqrt(gla),
            "decade_built": math.floor(r["year_built"] / 10.0) * 10.0,
            # greatest() is NULL-ignoring: greatest(NULL, 0.0) -> 0.0
            "garage_age_clamped": 0.0 if gy is None else max(r["yr_sold"] - gy, 0.0),
            "exter_qual_good": 1 if r["exter_qual"] in ("Ex", "Gd") else 0,
            "kitchen_qual_good": 1 if kq in ("Ex", "Gd") else 0,
            "nbhd_rich": 1
            if r["neighborhood"] in ("NoRidge", "NridgHt", "StoneBr")
            else 0,
            "is_pud": 1 if r["ms_sub_class"] in (120, 150, 160, 180) else 0,
            "mo_sold_sin": math.sin(2.0 * math.pi * r["mo_sold"] / 12.0),
            "mo_sold_cos": math.cos(2.0 * math.pi * r["mo_sold"] / 12.0),
        }

    return fe
