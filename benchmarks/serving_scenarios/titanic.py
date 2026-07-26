"""Titanic survival serving scenario for the SQL specializer benchmark.

Reproduces the canonical Kaggle-Titanic public-kernel feature pipeline as it
looks at SERVE time: scalar expressions over the raw passenger row plus LEFT
JOINs to prepare-time fitted tables (per-pclass fare medians, per-title age
medians and group mean fares, sex-x-pclass target-mean encoding, embarked
target encoding).
"""

import random
from collections.abc import Callable

import pyarrow as pa

NAME = "titanic_survival_features"
KAGGLE = (
    "Kaggle Titanic (survival prediction) — canonical public-kernel tricks: "
    "family_size/is_alone, fare_per_person with per-pclass median imputation, "
    "cabin deck letter + has_cabin, title-based age imputation with missing "
    "flag, age*pclass interaction, age bins, embarked one-hots, sex-x-pclass "
    "target-mean encoding and title-GROUP mean fare as fitted join tables."
)

N_INPUT_COLS = 10
N_OUTPUT_COLS = 20

ROW_SCHEMA = {
    "passenger_id": "int",
    "pclass": "int",
    "sex": "str",
    "age": "float?",
    "sibsp": "int",
    "parch": "int",
    "fare": "float?",
    "cabin": "str?",
    "embarked": "str?",
    "title": "str",
}

# Fitted fallback constants a real pipeline bakes into its generated SQL.
_GLOBAL_MEDIAN_AGE = 29.7
_GLOBAL_TITLE_FARE = 32.2
_MODE_PORT_RATE = 0.339

SQL = """
SELECT
  passenger_id,
  sibsp + parch + 1 AS family_size,
  CASE WHEN sibsp + parch = 0 THEN 1 ELSE 0 END AS is_alone,
  coalesce(fare, pclass_dim.median_fare) AS fare_filled,
  coalesce(fare, pclass_dim.median_fare) / (sibsp + parch + 1) AS fare_per_person,
  coalesce(upper(substr(trim(cabin), 1, 1)), 'U') AS deck,
  CASE WHEN cabin IS NULL THEN 0 ELSE 1 END AS has_cabin,
  CASE WHEN age IS NULL THEN 1 ELSE 0 END AS age_missing,
  coalesce(age, title_stats.median_age, 29.7) AS age_filled,
  coalesce(age, title_stats.median_age, 29.7) * pclass AS age_class,
  CASE
    WHEN coalesce(age, title_stats.median_age, 29.7) < 13.0 THEN 'child'
    WHEN coalesce(age, title_stats.median_age, 29.7) < 20.0 THEN 'teen'
    WHEN coalesce(age, title_stats.median_age, 29.7) < 60.0 THEN 'adult'
    ELSE 'senior'
  END AS age_bin,
  CASE WHEN coalesce(embarked, 'S') = 'C' THEN 1 ELSE 0 END AS embarked_c,
  CASE WHEN coalesce(embarked, 'S') = 'Q' THEN 1 ELSE 0 END AS embarked_q,
  CASE WHEN coalesce(embarked, 'S') = 'S' THEN 1 ELSE 0 END AS embarked_s,
  CASE WHEN sex = 'female' THEN 1 ELSE 0 END AS sex_female,
  sex || '-' || CAST(pclass AS VARCHAR) AS sex_pclass,
  sex_pclass_enc.survival_rate AS sex_pclass_rate,
  coalesce(title_stats.group_mean_fare, 32.2) AS title_fare_mean,
  pclass_dim.pclass_survival_rate AS pclass_rate,
  coalesce(embarked_enc.port_survival_rate, 0.339) AS port_rate
FROM __THIS__
LEFT JOIN pclass_dim ON pclass = pclass_dim.pc
LEFT JOIN title_stats ON title = title_stats.t_title
LEFT JOIN sex_pclass_enc
  ON sex = sex_pclass_enc.sp_sex AND pclass = sex_pclass_enc.sp_pclass
LEFT JOIN embarked_enc ON embarked = embarked_enc.port
"""

# Title -> (group median age, denormalized title-GROUP mean fare).  The
# grouping {Mr, Mrs, Miss, Master, Rare} happened at fit time; the fitted
# table is keyed on the raw title.  Capt/Jonkheer/Dona are deliberately NOT
# here: unseen-at-fit titles cause serve-time LEFT JOIN misses.
_TITLES = {
    "Mr": (30.0, 24.4),
    "Mrs": (35.0, 45.0),
    "Miss": (21.0, 43.8),
    "Master": (3.5, 37.0),
    "Dr": (46.5, 40.9),
    "Rev": (46.5, 40.9),
    "Col": (46.5, 40.9),
    "Major": (46.5, 40.9),
    "Mlle": (21.0, 43.8),
    "Ms": (21.0, 43.8),
    "Mme": (35.0, 45.0),
    "Lady": (46.5, 40.9),
    "Sir": (46.5, 40.9),
    "Countess": (46.5, 40.9),
}


def make_statics(seed: int) -> dict[str, pa.Table]:
    r = random.Random(seed)

    def jit(x: float) -> float:
        return round(x + r.uniform(-0.015, 0.015), 6)

    pclass_dim = pa.table(
        {
            "pc": [1, 2, 3],
            "median_fare": [round(jit(60.2875), 4), 14.25, 8.05],
            "pclass_survival_rate": [jit(0.6296), jit(0.4728), jit(0.2424)],
        }
    )
    titles = sorted(_TITLES)
    title_stats = pa.table(
        {
            "t_title": titles,
            "median_age": [_TITLES[t][0] for t in titles],
            "group_mean_fare": [jit(_TITLES[t][1]) for t in titles],
        }
    )
    sex_pclass_enc = pa.table(
        {
            "sp_sex": ["female", "female", "female", "male", "male", "male"],
            "sp_pclass": [1, 2, 3, 1, 2, 3],
            "survival_rate": [
                jit(0.9681),
                jit(0.9211),
                jit(0.5000),
                jit(0.3689),
                jit(0.1574),
                jit(0.1354),
            ],
        }
    )
    embarked_enc = pa.table(
        {
            "port": ["S", "C", "Q"],
            "port_survival_rate": [jit(0.3370), jit(0.5539), jit(0.3896)],
        }
    )
    return {
        "pclass_dim": pclass_dim,
        "title_stats": title_stats,
        "sex_pclass_enc": sex_pclass_enc,
        "embarked_enc": embarked_enc,
    }


def make_rows(seed: int, n: int) -> list[dict]:
    r = random.Random(seed)
    male_titles = ["Mr"] * 90 + ["Master"] * 6 + ["Dr", "Rev", "Capt", "Jonkheer"]
    female_titles = (
        ["Miss"] * 44
        + ["Mrs"] * 44
        + ["Mlle", "Ms", "Mme", "Lady", "Countess", "Dona"] * 2
    )
    decks = {1: "ABCDE", 2: "DEF", 3: "EFG"}
    fare_base = {1: 84.15, 2: 20.66, 3: 13.68}
    rows = []
    for i in range(n):
        pclass = r.choices([1, 2, 3], weights=[24, 21, 55])[0]
        sex = "male" if r.random() < 0.65 else "female"
        title = r.choice(male_titles if sex == "male" else female_titles)
        if r.random() < 0.20:
            age = None
        elif title == "Master":
            age = round(r.uniform(0.5, 12.0) * 2) / 2
        else:
            age = min(80.0, max(14.0, round(r.gauss(30.0, 13.0) * 2) / 2))
        sibsp = r.choices([0, 1, 2, 3, 4, 8], weights=[68, 23, 4, 3, 1, 1])[0]
        parch = r.choices([0, 1, 2, 5], weights=[76, 13, 9, 2])[0]
        if r.random() < 0.01:
            fare = None
        elif r.random() < 0.015:
            fare = 0.0
        else:
            fare = round(fare_base[pclass] * (0.35 + r.random() * 2.2), 4)
        if r.random() < 0.77:
            cabin = None
        else:
            letter = r.choice(decks[pclass])
            if r.random() < 0.15:
                letter = letter.lower()
            cabin = f"{letter}{r.randint(1, 130)}"
            if r.random() < 0.10:
                cabin = f" {cabin} "
        embarked = (
            None
            if r.random() < 0.02
            else r.choices(["S", "C", "Q"], weights=[72, 19, 9])[0]
        )
        rows.append(
            {
                "passenger_id": 1000 + i,
                "pclass": pclass,
                "sex": sex,
                "age": age,
                "sibsp": sibsp,
                "parch": parch,
                "fare": fare,
                "cabin": cabin,
                "embarked": embarked,
                "title": title,
            }
        )
    return rows


def handcrafted(statics: dict[str, pa.Table]) -> Callable[[dict], dict]:
    def cols(t: pa.Table) -> dict[str, list]:
        return {name: t.column(name).to_pylist() for name in t.column_names}

    p = cols(statics["pclass_dim"])
    pclass_lut = {
        k: (mf, sr)
        for k, mf, sr in zip(
            p["pc"], p["median_fare"], p["pclass_survival_rate"], strict=True
        )
    }
    t = cols(statics["title_stats"])
    title_lut = {
        k: (ma, gf)
        for k, ma, gf in zip(
            t["t_title"], t["median_age"], t["group_mean_fare"], strict=True
        )
    }
    s = cols(statics["sex_pclass_enc"])
    sp_lut = {
        (sx, pc): sr
        for sx, pc, sr in zip(
            s["sp_sex"], s["sp_pclass"], s["survival_rate"], strict=True
        )
    }
    e = cols(statics["embarked_enc"])
    port_lut = dict(zip(e["port"], e["port_survival_rate"], strict=True))

    def fn(row: dict) -> dict:
        sibsp, parch, pclass = row["sibsp"], row["parch"], row["pclass"]
        family_size = sibsp + parch + 1
        median_fare, pclass_rate = pclass_lut.get(pclass, (None, None))
        fare = row["fare"]
        fare_filled = fare if fare is not None else median_fare
        fare_per_person = None if fare_filled is None else fare_filled / family_size
        cabin = row["cabin"]
        deck = "U" if cabin is None else cabin.strip()[:1].upper()
        median_age, group_fare = title_lut.get(row["title"], (None, None))
        age = row["age"]
        if age is not None:
            age_filled = age
        elif median_age is not None:
            age_filled = median_age
        else:
            age_filled = _GLOBAL_MEDIAN_AGE
        if age_filled < 13.0:
            age_bin = "child"
        elif age_filled < 20.0:
            age_bin = "teen"
        elif age_filled < 60.0:
            age_bin = "adult"
        else:
            age_bin = "senior"
        embarked = row["embarked"]
        emb = embarked if embarked is not None else "S"
        port_rate = port_lut.get(embarked) if embarked is not None else None
        sex = row["sex"]
        return {
            "passenger_id": row["passenger_id"],
            "family_size": family_size,
            "is_alone": 1 if sibsp + parch == 0 else 0,
            "fare_filled": fare_filled,
            "fare_per_person": fare_per_person,
            "deck": deck,
            "has_cabin": 0 if cabin is None else 1,
            "age_missing": 1 if age is None else 0,
            "age_filled": age_filled,
            "age_class": age_filled * pclass,
            "age_bin": age_bin,
            "embarked_c": 1 if emb == "C" else 0,
            "embarked_q": 1 if emb == "Q" else 0,
            "embarked_s": 1 if emb == "S" else 0,
            "sex_female": 1 if sex == "female" else 0,
            "sex_pclass": f"{sex}-{pclass}",
            "sex_pclass_rate": sp_lut.get((sex, pclass)),
            "title_fare_mean": group_fare
            if group_fare is not None
            else _GLOBAL_TITLE_FARE,
            "pclass_rate": pclass_rate,
            "port_rate": port_rate if port_rate is not None else _MODE_PORT_RATE,
        }

    return fn
