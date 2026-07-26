"""fraud_txn — IEEE-CIS-style transaction-fraud serving benchmark scenario.

Serving shape of the famous IEEE-CIS Fraud Detection winning solutions:
uid/frequency/target encodings materialized at prepare time become plain
LEFT-JOIN lookup tables at serve time; everything else is scalar math over
the raw transaction row — including the wave-1 builtin features the winning
solutions actually used: log1p amounts via ln(1.0 + x), the Deotte cents
feature via round(x, 2), amount decade via round(x, -1), cyclical hour/dow
sin/cos encodings, email-domain group flags via IN / starts_with / ends_with
/ instr, and C-column outlier clips via least()."""

import math
import random
from collections.abc import Callable
from typing import Any

import pyarrow as pa

NAME = "fraud_txn"
KAGGLE = (
    "IEEE-CIS Fraud Detection (Kaggle 2019, Deotte/Yakovlev winning-solution "
    "serving shape): composite uid (card1 x addr1) + card1/addr1/email-domain "
    "frequency+target encodings as join tables, amt ratio/delta vs card1 and "
    "uid means, log1p amount, rounded cents feature, amount decade, D-column "
    "NULL flags, hour/day-of-week from unix ts + cyclical sin/cos encodings, "
    "amount buckets, ProductCD one-hots, email-domain group/suspicious flags, "
    "clipped C-columns."
)

N_INPUT_COLS = 32
N_OUTPUT_COLS = 57

ROW_SCHEMA: dict[str, str] = {
    "txn_id": "int",
    "transaction_amt": "float",
    "product_cd": "str",
    "card1": "int",
    "card2": "int?",
    "card3": "int?",
    "card4": "str?",
    "card5": "int?",
    "card6": "str?",
    "addr1": "int?",
    "addr2": "int?",
    "dist1": "float?",
    "dist2": "float?",
    "p_email_domain": "str?",
    "r_email_domain": "str?",
    "c1": "int",
    "c2": "int",
    "c3": "int",
    "c4": "int",
    "c5": "int",
    "c6": "int",
    "d1": "int?",
    "d2": "int?",
    "d3": "int?",
    "d4": "int?",
    "transaction_dt": "int",
    "m1": "bool?",
    "m2": "bool?",
    "m3": "bool?",
    "device_type": "str?",
    "v1": "float?",
    "v2": "float?",
}

_DOMAINS = [
    "gmail.com",
    "yahoo.com",
    "hotmail.com",
    "outlook.com",
    "aol.com",
    "anonymous.com",
    "protonmail.com",
    "icloud.com",
    "comcast.net",
    "live.com",
    "msn.com",
    "att.net",
    "verizon.net",
    "ymail.com",
    "mail.com",
    "optonline.net",
    "cox.net",
    "charter.net",
    "earthlink.net",
    "juno.com",
]
# serving traffic also carries domains never seen at prepare time -> join miss
_SERVE_DOMAINS = _DOMAINS + ["qq.com", "rocketmail.com", "protonmail.ch"]
_PRODUCTS = ["W", "C", "R", "H", "S"]

# The fitted uid (card1 x addr1) population is a property of the training
# data, not of the value-fit seed: fixed here so make_rows can send
# repeat-customer traffic that hits the uid encoding at a realistic rate.
_UID_SEED = 990721
_uid_cache: list[tuple[int, int]] | None = None


def _uid_pairs() -> list[tuple[int, int]]:
    global _uid_cache
    if _uid_cache is None:
        rng = random.Random(_UID_SEED)
        pairs: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        while len(pairs) < 2500:
            p = (rng.randint(1000, 1599), rng.randint(100, 559))
            if p not in seen:
                seen.add(p)
                pairs.append(p)
        _uid_cache = pairs
    return _uid_cache


def make_statics(seed: int) -> dict[str, pa.Table]:
    rng = random.Random(seed * 7919 + 1)

    card_ids = sorted(rng.sample(range(1000, 1500), 400))
    card1_stats = pa.table(
        {
            "card1_id": card_ids,
            "card1_amt_mean": [round(rng.uniform(8.0, 420.0), 2) for _ in card_ids],
            "card1_txn_cnt": [rng.randint(1, 5000) for _ in card_ids],
            "card1_fraud_rate": [round(rng.uniform(0.001, 0.35), 6) for _ in card_ids],
        }
    )

    addr_ids = sorted(rng.sample(range(100, 540), 150))
    addr1_stats = pa.table(
        {
            "addr1_id": addr_ids,
            "addr1_fraud_rate": [round(rng.uniform(0.001, 0.25), 6) for _ in addr_ids],
            "addr1_txn_cnt": [rng.randint(1, 20000) for _ in addr_ids],
        }
    )

    p_email_stats = pa.table(
        {
            "p_domain": list(_DOMAINS),
            "p_email_fraud_rate": [round(rng.uniform(0.002, 0.4), 6) for _ in _DOMAINS],
            "p_email_freq": [rng.randint(50, 200000) for _ in _DOMAINS],
        }
    )

    r_domains = _DOMAINS[:14]  # recipient encoding covers fewer domains
    r_email_stats = pa.table(
        {
            "r_domain": list(r_domains),
            "r_email_fraud_rate": [
                round(rng.uniform(0.002, 0.4), 6) for _ in r_domains
            ],
        }
    )

    pcd_stats = pa.table(
        {
            "pcd": list(_PRODUCTS),
            "pcd_fraud_rate": [round(rng.uniform(0.01, 0.12), 6) for _ in _PRODUCTS],
        }
    )

    uid_pairs = _uid_pairs()
    uid_stats = pa.table(
        {
            "uid_card1": [c for c, _ in uid_pairs],
            "uid_addr1": [a for _, a in uid_pairs],
            "uid_amt_mean": [round(rng.uniform(8.0, 420.0), 2) for _ in uid_pairs],
            "uid_txn_cnt": [rng.randint(1, 800) for _ in uid_pairs],
        }
    )

    return {
        "card1_stats": card1_stats,
        "addr1_stats": addr1_stats,
        "p_email_stats": p_email_stats,
        "r_email_stats": r_email_stats,
        "pcd_stats": pcd_stats,
        "uid_stats": uid_stats,
    }


def make_rows(seed: int, n: int) -> list[dict]:
    rng = random.Random(seed * 104729 + 3)
    uid_pairs = _uid_pairs()
    base_ts = 1_700_000_000
    rows: list[dict[str, Any]] = []
    for i in range(n):
        if rng.random() < 0.55:  # repeat customer: hits a fitted uid pair
            card1, addr1 = uid_pairs[rng.randrange(len(uid_pairs))]
        else:
            card1 = rng.randint(1000, 1599)  # tail misses the stats table
            addr1 = None if rng.random() < 0.3 else rng.randint(100, 560)
        if rng.random() < 0.2:  # whole-dollar transactions (the cents trick)
            amt = float(rng.choice([20, 25, 30, 50, 75, 100, 150, 200, 300, 500]))
        else:
            amt = round(math.exp(rng.gauss(3.8, 0.9)), 2)
        p_dom = None if rng.random() < 0.16 else rng.choice(_SERVE_DOMAINS)
        if p_dom is not None and rng.random() < 0.3:
            r_dom: str | None = p_dom  # purchaser == recipient is common
        else:
            r_dom = None if rng.random() < 0.45 else rng.choice(_SERVE_DOMAINS)
        rows.append(
            {
                "txn_id": 3_000_000 + i,
                "transaction_amt": amt,
                "product_cd": rng.choices(_PRODUCTS, weights=[65, 10, 8, 9, 8])[0],
                "card1": card1,
                "card2": None if rng.random() < 0.15 else rng.randint(100, 600),
                "card3": None if rng.random() < 0.1 else rng.choice([150, 185]),
                "card4": None
                if rng.random() < 0.1
                else rng.choice(
                    [
                        "visa",
                        "Visa ",
                        "mastercard",
                        "MasterCard",
                        "discover",
                        "american express",
                    ]
                ),
                "card5": None if rng.random() < 0.2 else rng.randint(100, 240),
                "card6": None
                if rng.random() < 0.08
                else rng.choices(
                    ["debit", "credit", "charge card"], weights=[70, 27, 3]
                )[0],
                "addr1": addr1,
                "addr2": None if rng.random() < 0.3 else rng.choice([87, 60, 96]),
                "dist1": None if rng.random() < 0.6 else float(rng.randint(0, 3000)),
                "dist2": None if rng.random() < 0.93 else float(rng.randint(0, 8000)),
                "p_email_domain": p_dom,
                "r_email_domain": r_dom,
                "c1": rng.randint(0, 20)
                if rng.random() < 0.85
                else rng.randint(20, 400),
                "c2": rng.randint(0, 15)
                if rng.random() < 0.85
                else rng.randint(15, 300),
                "c3": rng.randint(0, 2),
                "c4": rng.randint(0, 8),
                "c5": rng.randint(0, 30),
                "c6": rng.randint(0, 12),
                "d1": None if rng.random() < 0.45 else rng.randint(0, 640),
                "d2": None if rng.random() < 0.55 else rng.randint(0, 640),
                "d3": None if rng.random() < 0.6 else rng.randint(0, 500),
                "d4": None if rng.random() < 0.65 else rng.randint(0, 500),
                "transaction_dt": base_ts + rng.randint(0, 60 * 86400),
                "m1": None if rng.random() < 0.3 else rng.random() < 0.6,
                "m2": None if rng.random() < 0.35 else rng.random() < 0.5,
                "m3": None if rng.random() < 0.4 else rng.random() < 0.5,
                "device_type": None
                if rng.random() < 0.25
                else rng.choice(["desktop", "mobile"]),
                "v1": None if rng.random() < 0.5 else round(rng.uniform(0.0, 10.0), 4),
                "v2": None if rng.random() < 0.5 else round(rng.uniform(0.0, 10.0), 4),
            }
        )
    return rows


SQL = """
SELECT
  txn_id,
  transaction_amt AS amt,
  card1_stats.card1_amt_mean AS card1_amt_mean,
  card1_stats.card1_txn_cnt AS card1_txn_cnt,
  card1_stats.card1_fraud_rate AS card1_fraud_rate,
  transaction_amt / card1_stats.card1_amt_mean AS amt_to_card1_mean,
  transaction_amt - card1_stats.card1_amt_mean AS amt_minus_card1_mean,
  addr1_stats.addr1_fraud_rate AS addr1_fraud_rate,
  addr1_stats.addr1_txn_cnt AS addr1_txn_cnt,
  uid_stats.uid_amt_mean AS uid_amt_mean,
  uid_stats.uid_txn_cnt AS uid_txn_cnt,
  transaction_amt / uid_stats.uid_amt_mean AS amt_to_uid_mean,
  p_email_stats.p_email_fraud_rate AS p_email_fraud_rate,
  p_email_stats.p_email_freq AS p_email_freq,
  r_email_stats.r_email_fraud_rate AS r_email_fraud_rate,
  pcd_stats.pcd_fraud_rate AS pcd_fraud_rate,
  transaction_amt * pcd_stats.pcd_fraud_rate AS amt_x_pcd_rate,
  CAST(((transaction_dt % 86400) - (transaction_dt % 3600)) / 3600 AS INTEGER) AS txn_hour,
  (CAST((transaction_dt - (transaction_dt % 86400)) / 86400 AS INTEGER) + 4) % 7 AS txn_dow,
  CASE WHEN (transaction_dt % 86400) < 21600 THEN 1 ELSE 0 END AS is_night,
  CASE WHEN transaction_amt < 20.0 THEN 0
       WHEN transaction_amt < 50.0 THEN 1
       WHEN transaction_amt < 100.0 THEN 2
       WHEN transaction_amt < 300.0 THEN 3
       ELSE 4 END AS amt_bucket,
  transaction_amt % 1.0 AS amt_cents,
  CASE WHEN (transaction_amt % 1.0) = 0.0 THEN 1 ELSE 0 END AS is_whole_amt,
  CASE WHEN d1 IS NULL THEN 1 ELSE 0 END AS d1_null,
  CASE WHEN d2 IS NULL THEN 1 ELSE 0 END AS d2_null,
  CASE WHEN d3 IS NULL THEN 1 ELSE 0 END AS d3_null,
  CASE WHEN d4 IS NULL THEN 1 ELSE 0 END AS d4_null,
  coalesce(d1, -1) AS d1_filled,
  CASE WHEN product_cd = 'W' THEN 1 ELSE 0 END AS pcd_w,
  CASE WHEN product_cd = 'C' THEN 1 ELSE 0 END AS pcd_c,
  CASE WHEN product_cd = 'R' THEN 1 ELSE 0 END AS pcd_r,
  CASE WHEN product_cd = 'H' THEN 1 ELSE 0 END AS pcd_h,
  CASE WHEN product_cd = 'S' THEN 1 ELSE 0 END AS pcd_s,
  coalesce(upper(trim(card4)), 'UNK') AS card4_norm,
  CASE WHEN card6 = 'debit' THEN 1 ELSE 0 END AS is_debit,
  CASE WHEN p_email_domain = r_email_domain THEN 1
       WHEN p_email_domain IS NULL OR r_email_domain IS NULL THEN -1
       ELSE 0 END AS email_match,
  c1 + c2 + c3 + c4 + c5 + c6 AS c_sum,
  c1 / (c1 + c2 + c3 + c4 + c5 + c6 + 1) AS c1_share,
  coalesce(dist1, 0.0) AS dist1_filled,
  CASE WHEN m1 IS NULL THEN -1 WHEN m1 THEN 1 ELSE 0 END AS m1_flag,
  CASE WHEN addr1 IS NULL THEN 0 ELSE 1 END AS addr_known,
  ln(1.0 + transaction_amt) AS amt_log1p,
  round(transaction_amt, -1) AS amt_decade,
  round(transaction_amt - floor(transaction_amt), 2) AS amt_cents_r,
  sin(2.0 * pi() * (((transaction_dt % 86400) - (transaction_dt % 3600)) / 3600) / 24.0) AS hour_sin,
  cos(2.0 * pi() * (((transaction_dt % 86400) - (transaction_dt % 3600)) / 3600) / 24.0) AS hour_cos,
  sin(2.0 * pi() * ((CAST((transaction_dt - (transaction_dt % 86400)) / 86400 AS INTEGER) + 4) % 7) / 7.0) AS dow_sin,
  cos(2.0 * pi() * ((CAST((transaction_dt - (transaction_dt % 86400)) / 86400 AS INTEGER) + 4) % 7) / 7.0) AS dow_cos,
  CASE WHEN p_email_domain IN ('protonmail.com', 'protonmail.ch', 'anonymous.com', 'mail.com', 'qq.com')
       THEN 1 ELSE 0 END AS p_dom_suspicious,
  CASE WHEN p_email_domain IS NULL THEN 0
       WHEN starts_with(p_email_domain, 'gmail') THEN 1 ELSE 0 END AS p_is_gmail,
  CASE WHEN p_email_domain IN ('hotmail.com', 'outlook.com', 'live.com', 'msn.com')
       THEN 1 ELSE 0 END AS p_is_msft,
  CASE WHEN p_email_domain IS NULL THEN 0
       WHEN ends_with(p_email_domain, '.net') THEN 1 ELSE 0 END AS p_dom_net,
  CASE WHEN p_email_domain IS NULL THEN -1
       ELSE length(p_email_domain) - instr(p_email_domain, '.') END AS p_dom_suffix_len,
  least(c1, 50) AS c1_clip,
  least(c2, 40) AS c2_clip,
  greatest(coalesce(d1, 0), coalesce(d2, 0), coalesce(d3, 0), coalesce(d4, 0)) AS d_max,
  ln(1.0 + card1_stats.card1_txn_cnt) AS card1_cnt_log
FROM __THIS__
LEFT JOIN card1_stats ON card1 = card1_stats.card1_id
LEFT JOIN addr1_stats ON addr1 = addr1_stats.addr1_id
LEFT JOIN uid_stats ON card1 = uid_stats.uid_card1 AND addr1 = uid_stats.uid_addr1
LEFT JOIN p_email_stats ON p_email_domain = p_email_stats.p_domain
LEFT JOIN r_email_stats ON r_email_domain = r_email_stats.r_domain
LEFT JOIN pcd_stats ON product_cd = pcd_stats.pcd
"""


def _round_half_away(x: float, n: int) -> float:
    """SQL round(x, n): scale by 10**n, round half away from zero, unscale.
    Bit-identical to the engine/DuckDB for this scenario's inputs (probed)."""
    s = 10.0**n
    y = x * s
    return math.copysign(math.floor(abs(y) + 0.5), y) / s


_SUSPICIOUS_DOMAINS = frozenset(
    {"protonmail.com", "protonmail.ch", "anonymous.com", "mail.com", "qq.com"}
)
_MSFT_DOMAINS = frozenset({"hotmail.com", "outlook.com", "live.com", "msn.com"})


def handcrafted(statics: dict[str, pa.Table]) -> Callable[[dict], dict]:
    """What a competent engineer hand-writes for the same features: hydrate the
    encoding tables into plain dicts once, then a per-row closure."""
    card1_map = {
        r["card1_id"]: (r["card1_amt_mean"], r["card1_txn_cnt"], r["card1_fraud_rate"])
        for r in statics["card1_stats"].to_pylist()
    }
    addr1_map = {
        r["addr1_id"]: (r["addr1_fraud_rate"], r["addr1_txn_cnt"])
        for r in statics["addr1_stats"].to_pylist()
    }
    p_email_map = {
        r["p_domain"]: (r["p_email_fraud_rate"], r["p_email_freq"])
        for r in statics["p_email_stats"].to_pylist()
    }
    r_email_map = {
        r["r_domain"]: r["r_email_fraud_rate"]
        for r in statics["r_email_stats"].to_pylist()
    }
    pcd_map = {r["pcd"]: r["pcd_fraud_rate"] for r in statics["pcd_stats"].to_pylist()}
    uid_map = {
        (r["uid_card1"], r["uid_addr1"]): (r["uid_amt_mean"], r["uid_txn_cnt"])
        for r in statics["uid_stats"].to_pylist()
    }

    def fn(row: dict) -> dict:
        amt = row["transaction_amt"]
        ts = row["transaction_dt"]
        hour = (ts % 86400) // 3600
        dow = (ts // 86400 + 4) % 7

        c1s = card1_map.get(row["card1"])
        card1_amt_mean, card1_txn_cnt, card1_fraud_rate = (
            c1s if c1s else (None, None, None)
        )
        a1s = addr1_map.get(row["addr1"])
        addr1_fraud_rate, addr1_txn_cnt = a1s if a1s else (None, None)
        # NULL addr1 can never equal a fitted key, same as the SQL join
        us = uid_map.get((row["card1"], row["addr1"]))
        uid_amt_mean, uid_txn_cnt = us if us else (None, None)
        pes = p_email_map.get(row["p_email_domain"])
        p_email_fraud_rate, p_email_freq = pes if pes else (None, None)
        r_email_fraud_rate = r_email_map.get(row["r_email_domain"])
        pcd_fraud_rate = pcd_map.get(row["product_cd"])

        if amt < 20.0:
            amt_bucket = 0
        elif amt < 50.0:
            amt_bucket = 1
        elif amt < 100.0:
            amt_bucket = 2
        elif amt < 300.0:
            amt_bucket = 3
        else:
            amt_bucket = 4

        amt_cents = math.fmod(amt, 1.0)  # SQL float %: fmod, amounts positive

        p_dom, r_dom = row["p_email_domain"], row["r_email_domain"]
        if p_dom is not None and r_dom is not None and p_dom == r_dom:
            email_match = 1
        elif p_dom is None or r_dom is None:
            email_match = -1
        else:
            email_match = 0

        c_sum = row["c1"] + row["c2"] + row["c3"] + row["c4"] + row["c5"] + row["c6"]
        card4, m1 = row["card4"], row["m1"]

        return {
            "txn_id": row["txn_id"],
            "amt": amt,
            "card1_amt_mean": card1_amt_mean,
            "card1_txn_cnt": card1_txn_cnt,
            "card1_fraud_rate": card1_fraud_rate,
            "amt_to_card1_mean": amt / card1_amt_mean
            if card1_amt_mean is not None
            else None,
            "amt_minus_card1_mean": amt - card1_amt_mean
            if card1_amt_mean is not None
            else None,
            "addr1_fraud_rate": addr1_fraud_rate,
            "addr1_txn_cnt": addr1_txn_cnt,
            "uid_amt_mean": uid_amt_mean,
            "uid_txn_cnt": uid_txn_cnt,
            "amt_to_uid_mean": amt / uid_amt_mean if uid_amt_mean is not None else None,
            "p_email_fraud_rate": p_email_fraud_rate,
            "p_email_freq": p_email_freq,
            "r_email_fraud_rate": r_email_fraud_rate,
            "pcd_fraud_rate": pcd_fraud_rate,
            "amt_x_pcd_rate": amt * pcd_fraud_rate
            if pcd_fraud_rate is not None
            else None,
            "txn_hour": hour,
            "txn_dow": dow,
            "is_night": 1 if ts % 86400 < 21600 else 0,
            "amt_bucket": amt_bucket,
            "amt_cents": amt_cents,
            "is_whole_amt": 1 if amt_cents == 0.0 else 0,
            "d1_null": 1 if row["d1"] is None else 0,
            "d2_null": 1 if row["d2"] is None else 0,
            "d3_null": 1 if row["d3"] is None else 0,
            "d4_null": 1 if row["d4"] is None else 0,
            "d1_filled": row["d1"] if row["d1"] is not None else -1,
            "pcd_w": 1 if row["product_cd"] == "W" else 0,
            "pcd_c": 1 if row["product_cd"] == "C" else 0,
            "pcd_r": 1 if row["product_cd"] == "R" else 0,
            "pcd_h": 1 if row["product_cd"] == "H" else 0,
            "pcd_s": 1 if row["product_cd"] == "S" else 0,
            "card4_norm": card4.strip(" ").upper() if card4 is not None else "UNK",
            "is_debit": 1 if row["card6"] == "debit" else 0,
            "email_match": email_match,
            "c_sum": c_sum,
            "c1_share": row["c1"] / (c_sum + 1),
            "dist1_filled": row["dist1"] if row["dist1"] is not None else 0.0,
            "m1_flag": -1 if m1 is None else (1 if m1 else 0),
            "addr_known": 0 if row["addr1"] is None else 1,
            "amt_log1p": math.log(1.0 + amt),
            "amt_decade": _round_half_away(amt, -1),
            "amt_cents_r": _round_half_away(amt - math.floor(amt), 2),
            "hour_sin": math.sin(2.0 * math.pi * hour / 24.0),
            "hour_cos": math.cos(2.0 * math.pi * hour / 24.0),
            "dow_sin": math.sin(2.0 * math.pi * dow / 7.0),
            "dow_cos": math.cos(2.0 * math.pi * dow / 7.0),
            # NULL IN (...) is NULL -> CASE falls through to 0, same as .get miss
            "p_dom_suspicious": 1 if p_dom in _SUSPICIOUS_DOMAINS else 0,
            "p_is_gmail": 1 if p_dom is not None and p_dom.startswith("gmail") else 0,
            "p_is_msft": 1 if p_dom in _MSFT_DOMAINS else 0,
            "p_dom_net": 1 if p_dom is not None and p_dom.endswith(".net") else 0,
            # instr is 1-based codepoints; domains are ASCII with one dot
            "p_dom_suffix_len": -1
            if p_dom is None
            else len(p_dom) - (p_dom.index(".") + 1),
            "c1_clip": min(row["c1"], 50),
            "c2_clip": min(row["c2"], 40),
            "d_max": max(
                row["d1"] or 0, row["d2"] or 0, row["d3"] or 0, row["d4"] or 0
            ),
            "card1_cnt_log": math.log(1.0 + card1_txn_cnt)
            if card1_txn_cnt is not None
            else None,
        }

    return fn
