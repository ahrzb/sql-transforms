"""Titanic survival — THE classic Kaggle problem, as a serving-path scenario.

Reproduces the canonical public-kernel feature pipeline at serve time:
scalar expressions over the passenger row + LEFT JOINs to fitted encoding
tables (a target-mean encoding IS a join table at serve time).
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable

import pyarrow as pa

NAME = "titanic"
KAGGLE = (
    "Titanic: Machine Learning from Disaster — the canonical public-kernel recipe: "
    "FamilySize/IsAlone, FarePerPerson, deck letter from Cabin + HasCabin, "
    "Age-imputation flag + Age*Pclass interaction, age bins, Embarked one-hots, "
    "sex x pclass survival target-mean encoding and title-group mean-fare/survival "
    "encodings served as fitted join tables (incl. the famous unseen-'Dona' miss)."
)

N_INPUT_COLS = 10
N_OUTPUT_COLS = 24

ROW_SCHEMA = {
    "passenger_id": "int",
    "pclass": "int",
    "sex": "str",
    "title": "str",
    "age": "float?",
    "sibsp": "int",
    "parch": "int",
    "fare": "float?",
    "cabin": "str?",
    "embarked": "str?",
}

# Fitted constants a pipeline would bake into the serving query as literals.
AGE_MEDIAN = 28.0
FARE_MEDIAN = 14.4542
GLOBAL_RATE = 0.383838

# Title -> group, as the canonical kernels collapse rare titles. "Dona" is
# deliberately NOT here: it appears only in the test set and is the classic
# unseen-category trap — served as a LEFT JOIN miss + coalesce fallback.
_TITLE_TO_GROUP = {
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Mme": "Mrs",
    "Miss": "Miss",
    "Mlle": "Miss",
    "Ms": "Miss",
    "Master": "Master",
    "Dr": "Rare",
    "Rev": "Rare",
    "Col": "Rare",
    "Major": "Rare",
    "Capt": "Rare",
    "Sir": "Rare",
    "Lady": "Rare",
    "Don": "Rare",
    "Countess": "Rare",
    "Jonkheer": "Rare",
}


def make_statics(seed: int) -> dict[str, pa.Table]:
    rnd = random.Random(seed)

    def jit(v: float) -> float:
        return round(v + rnd.uniform(-0.015, 0.015), 6)

    # sex x pclass survival target-mean encoding (train-set rates + fit noise).
    sp_base = [
        ("female", 1, 0.968, 94),
        ("female", 2, 0.921, 76),
        ("female", 3, 0.500, 144),
        ("male", 1, 0.369, 122),
        ("male", 2, 0.157, 108),
        ("male", 3, 0.135, 347),
    ]
    sex_pclass_enc = pa.Table.from_pylist(
        [
            {"sex": s, "pclass": p, "survival_rate": jit(r), "n": n}
            for s, p, r, n in sp_base
        ],
        schema=pa.schema(
            [
                ("sex", pa.string()),
                ("pclass", pa.int64()),
                ("survival_rate", pa.float64()),
                ("n", pa.int64()),
            ]
        ),
    )

    # Title-group encodings: every title carries its GROUP's fitted stats,
    # exactly as a materialized groupby-join would.
    group_fare = {
        k: jit(v)
        for k, v in [
            ("Mr", 24.44),
            ("Mrs", 45.14),
            ("Miss", 43.80),
            ("Master", 37.98),
            ("Rare", 33.50),
        ]
    }
    group_rate = {
        k: jit(v)
        for k, v in [
            ("Mr", 0.157),
            ("Mrs", 0.792),
            ("Miss", 0.703),
            ("Master", 0.575),
            ("Rare", 0.444),
        ]
    }
    title_dim = pa.Table.from_pylist(
        [
            {
                "title": t,
                "title_group": g,
                "group_mean_fare": group_fare[g],
                "group_survival_rate": group_rate[g],
            }
            for t, g in _TITLE_TO_GROUP.items()
        ],
        schema=pa.schema(
            [
                ("title", pa.string()),
                ("title_group", pa.string()),
                ("group_mean_fare", pa.float64()),
                ("group_survival_rate", pa.float64()),
            ]
        ),
    )

    embarked_dim = pa.Table.from_pylist(
        [
            {"embarked": "S", "survival_rate": jit(0.339)},
            {"embarked": "C", "survival_rate": jit(0.554)},
            {"embarked": "Q", "survival_rate": jit(0.390)},
        ],
        schema=pa.schema([("embarked", pa.string()), ("survival_rate", pa.float64())]),
    )

    return {
        "sex_pclass_enc": sex_pclass_enc,
        "title_dim": title_dim,
        "embarked_dim": embarked_dim,
    }


_MALE_TITLES = [
    "Mr",
    "Master",
    "Dr",
    "Rev",
    "Col",
    "Major",
    "Capt",
    "Sir",
    "Don",
    "Jonkheer",
]
_MALE_W = [80, 8, 3, 2, 2, 1, 1, 1, 1, 1]
# "Dona" (~3% of women) is unseen by title_dim -> the LEFT JOIN miss path.
_FEMALE_TITLES = ["Miss", "Mrs", "Mlle", "Mme", "Ms", "Lady", "Countess", "Dona"]
_FEMALE_W = [47, 41, 3, 2, 2, 1, 1, 3]
_DECKS_BY_CLASS = {1: "ABCDE", 2: "DEF", 3: "EFG"}
_CABIN_MISS_P = {1: 0.20, 2: 0.75, 3: 0.94}


def make_rows(seed: int, n: int) -> list[dict]:
    rnd = random.Random(seed)
    rows: list[dict] = []
    for i in range(n):
        pclass = rnd.choices([1, 2, 3], [24, 21, 55])[0]
        sex = "male" if rnd.random() < 0.65 else "female"
        if sex == "male":
            title = rnd.choices(_MALE_TITLES, _MALE_W)[0]
        else:
            title = rnd.choices(_FEMALE_TITLES, _FEMALE_W)[0]

        if rnd.random() < 0.199:  # 177/891 missing in the train set
            age = None
        elif title == "Master":
            age = round(rnd.uniform(0.42, 12.0) * 2) / 2
        else:
            mu, sd = {
                "Miss": (22.0, 10.0),
                "Mrs": (36.0, 12.0),
                "Mr": (32.0, 12.0),
            }.get(title, (45.0, 10.0))
            age = round(min(80.0, max(0.42, rnd.gauss(mu, sd))) * 2) / 2

        sibsp = rnd.choices([0, 1, 2, 3, 4, 5, 8], [68, 23, 3, 2, 2, 1, 1])[0]
        parch = rnd.choices([0, 1, 2, 3, 4, 5, 6], [76, 13, 8, 1, 1, 1, 1])[0]

        if rnd.random() < 0.008:  # the lone missing Fare is a test-set classic
            fare = None
        else:
            mu, sd = {1: (4.2, 0.7), 2: (2.7, 0.4), 3: (2.1, 0.45)}[pclass]
            fare = round(math.exp(rnd.gauss(mu, sd)), 4)

        if rnd.random() < _CABIN_MISS_P[pclass]:
            cabin = None
        else:
            deck = rnd.choice(_DECKS_BY_CLASS[pclass])
            cabin = f"{deck}{rnd.randint(1, 130)}"
            if rnd.random() < 0.08:  # multi-cabin families: "C23 C25"
                cabin = f"{cabin} {deck}{rnd.randint(1, 130)}"
            if rnd.random() < 0.15:  # messy serving payloads
                cabin = cabin.lower()
            if rnd.random() < 0.10:
                cabin = " " + cabin
            if rnd.random() < 0.10:
                cabin = cabin + " "

        embarked = (
            None
            if rnd.random() < 0.012
            else rnd.choices(["S", "C", "Q"], [72, 19, 9])[0]
        )

        rows.append(
            {
                "passenger_id": 892 + i,
                "pclass": pclass,
                "sex": sex,
                "title": title,
                "age": age,
                "sibsp": sibsp,
                "parch": parch,
                "fare": fare,
                "cabin": cabin,
                "embarked": embarked,
            }
        )
    return rows


SQL = """
SELECT
  __THIS__.passenger_id AS passenger_id,
  __THIS__.pclass AS pclass,
  CASE WHEN __THIS__.sex = 'male' THEN 1 ELSE 0 END AS sex_male,
  __THIS__.sex || '_' || CAST(__THIS__.pclass AS VARCHAR) AS sex_pclass_key,
  __THIS__.sibsp + __THIS__.parch + 1 AS family_size,
  CASE WHEN __THIS__.sibsp + __THIS__.parch = 0 THEN 1 ELSE 0 END AS is_alone,
  CASE WHEN __THIS__.fare IS NULL THEN 1 ELSE 0 END AS fare_missing,
  coalesce(__THIS__.fare, 14.4542) AS fare_filled,
  coalesce(__THIS__.fare, 14.4542) / (__THIS__.sibsp + __THIS__.parch + 1)
    AS fare_per_person,
  CASE WHEN __THIS__.age IS NULL THEN 1 ELSE 0 END AS age_missing,
  coalesce(__THIS__.age, 28.0) AS age_filled,
  coalesce(__THIS__.age, 28.0) * __THIS__.pclass AS age_x_pclass,
  CASE
    WHEN __THIS__.age IS NULL THEN 'unknown'
    WHEN __THIS__.age < 13.0 THEN 'child'
    WHEN __THIS__.age < 20.0 THEN 'teen'
    WHEN __THIS__.age < 41.0 THEN 'adult'
    WHEN __THIS__.age < 61.0 THEN 'mid'
    ELSE 'senior'
  END AS age_bin,
  CASE WHEN __THIS__.cabin IS NULL THEN 0 ELSE 1 END AS has_cabin,
  CASE
    WHEN __THIS__.cabin IS NULL THEN 'U'
    ELSE upper(substr(trim(__THIS__.cabin), 1, 1))
  END AS deck,
  CASE WHEN __THIS__.embarked = 'S' THEN 1 ELSE 0 END AS embarked_s,
  CASE WHEN __THIS__.embarked = 'C' THEN 1 ELSE 0 END AS embarked_c,
  CASE WHEN __THIS__.embarked = 'Q' THEN 1 ELSE 0 END AS embarked_q,
  sp.survival_rate AS sex_pclass_rate,
  coalesce(td.title_group, 'Rare') AS title_group,
  coalesce(td.group_survival_rate, 0.383838) AS title_rate,
  td.group_mean_fare AS title_fare_mean,
  coalesce(__THIS__.fare, 14.4542) - td.group_mean_fare AS fare_minus_title_mean,
  coalesce(em.survival_rate, 0.383838) AS embarked_rate
FROM __THIS__
LEFT JOIN sex_pclass_enc AS sp
  ON __THIS__.sex = sp.sex AND __THIS__.pclass = sp.pclass
LEFT JOIN title_dim AS td ON __THIS__.title = td.title
LEFT JOIN embarked_dim AS em ON __THIS__.embarked = em.embarked
"""


def handcrafted(statics: dict[str, pa.Table]) -> Callable[[dict], dict]:
    """What a competent engineer hand-writes for a Python microservice:
    plain-dict lookups prepared once, a per-row closure after that."""
    sp = {
        (r["sex"], r["pclass"]): r["survival_rate"]
        for r in statics["sex_pclass_enc"].to_pylist()
    }
    td = {r["title"]: r for r in statics["title_dim"].to_pylist()}
    em = {
        r["embarked"]: r["survival_rate"] for r in statics["embarked_dim"].to_pylist()
    }

    def infer(row: dict) -> dict:
        sex = row["sex"]
        pclass = row["pclass"]
        age = row["age"]
        fare = row["fare"]
        cabin = row["cabin"]
        emb = row["embarked"]
        sibsp = row["sibsp"]
        parch = row["parch"]

        family_size = sibsp + parch + 1
        fare_filled = fare if fare is not None else 14.4542
        age_filled = age if age is not None else 28.0
        t = td.get(row["title"])

        if age is None:
            age_bin = "unknown"
        elif age < 13.0:
            age_bin = "child"
        elif age < 20.0:
            age_bin = "teen"
        elif age < 41.0:
            age_bin = "adult"
        elif age < 61.0:
            age_bin = "mid"
        else:
            age_bin = "senior"

        return {
            "passenger_id": row["passenger_id"],
            "pclass": pclass,
            "sex_male": 1 if sex == "male" else 0,
            "sex_pclass_key": sex + "_" + str(pclass),
            "family_size": family_size,
            "is_alone": 1 if sibsp + parch == 0 else 0,
            "fare_missing": 1 if fare is None else 0,
            "fare_filled": fare_filled,
            "fare_per_person": fare_filled / family_size,
            "age_missing": 1 if age is None else 0,
            "age_filled": age_filled,
            "age_x_pclass": age_filled * pclass,
            "age_bin": age_bin,
            "has_cabin": 0 if cabin is None else 1,
            "deck": "U" if cabin is None else cabin.strip()[:1].upper(),
            "embarked_s": 1 if emb == "S" else 0,
            "embarked_c": 1 if emb == "C" else 0,
            "embarked_q": 1 if emb == "Q" else 0,
            "sex_pclass_rate": sp[(sex, pclass)],
            "title_group": t["title_group"] if t is not None else "Rare",
            "title_rate": t["group_survival_rate"] if t is not None else 0.383838,
            "title_fare_mean": t["group_mean_fare"] if t is not None else None,
            "fare_minus_title_mean": (
                fare_filled - t["group_mean_fare"] if t is not None else None
            ),
            "embarked_rate": em.get(emb, 0.383838),
        }

    return infer
