"""Worker loop: seeds on stdin, TWO JSON lines per seed on stdout.

    {"event": "case",   "seed": s, "sql": "...", "tags": [...], "case": {...}}
    {"event": "result", "seed": s, "kind": ..., "klass": ..., ...}

The case line is generated and FLUSHED BEFORE the case is evaluated, so the
parent owns the actual inputs — SQL, schemas, rows, static tables, UDF and
tree specs — even when this process dies mid-case (rust panic, abort,
unbounded recursion) or is killed for outrunning its timeout. A seed alone is
not a durable case: the generator changes, and then the seed reproduces a
DIFFERENT query. The recorded case is what survives that.

The case is generated exactly ONCE here and handed to `run_case_json`, so the
inputs announced to the parent are the inputs that were run, not a second
draw that happens to share a seed.

# The encoding, and why it is spelled out rather than pickled

Case values are ordinary Python: the query AST is dataclasses, decimal static
columns are `decimal.Decimal`, and the float columns carry NaN and +-inf,
which plain JSON cannot spell. Every one of those gets an explicit tagged
form (`ENCODING_LEGEND`, recorded into the campaign's provenance sidecar) and
the dump runs with `allow_nan=False`, so a value that slipped through
unencoded raises here instead of writing a NaN token no strict JSON reader
will take. Nothing is pickled and nothing is `eval`-ed: the sidecar is data a
reader can inspect, not code it has to trust.
"""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import json
import math
import sys

from . import gen as G
from .oracle import run_case_json

ENCODING_LEGEND = {
    "$dataclass": "a generator dataclass: {'$dataclass': <class name>, "
    "'fields': {<field>: <encoded>}}",
    "$tuple": "a tuple, as {'$tuple': [<encoded>, ...]}",
    "$decimal": "an exact decimal, as its string form",
    "$datetime": "a date, time or datetime, ISO 8601",
    "$float": "a non-finite float: 'nan' | 'inf' | '-inf'",
    "$error": "generation or encoding raised; the case is not recoverable",
}


def encode(v):
    """`v` as strictly-JSON-serializable data, by `ENCODING_LEGEND`."""
    if v is None or isinstance(v, (bool, int, str)):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else {"$float": _nonfinite(v)}
    if isinstance(v, decimal.Decimal):
        return {"$decimal": str(v)}
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return {"$datetime": v.isoformat()}
    if isinstance(v, dict):
        if any(not isinstance(k, str) for k in v):
            raise TypeError("case mappings require string keys")
        return {k: encode(x) for k, x in v.items()}
    if isinstance(v, list):
        return [encode(x) for x in v]
    if isinstance(v, tuple):
        return {"$tuple": [encode(x) for x in v]}
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {
            "$dataclass": type(v).__name__,
            "fields": {
                f.name: encode(getattr(v, f.name)) for f in dataclasses.fields(v)
            },
        }
    raise TypeError(f"no case encoding for {type(v).__name__}")


def _nonfinite(v: float) -> str:
    if math.isnan(v):
        return "nan"
    return "inf" if v > 0 else "-inf"


def _emit(obj: dict) -> None:
    sys.stdout.write(json.dumps(obj, allow_nan=False) + "\n")
    sys.stdout.flush()


def _case_event(seed: int, case: G.Case, sql: str) -> dict:
    return {
        "event": "case",
        "seed": seed,
        "sql": sql,
        "tags": case.tags,
        "case": encode(case),
    }


def _skip(seed: int, klass: str, detail: str) -> dict:
    return {
        "event": "result",
        "seed": seed,
        "sql": "",
        "kind": "SKIP",
        "klass": klass,
        "detail": detail[:500],
        "tags": [],
        "oracle_outcome": None,
    }


def run_seed(seed: int) -> None:
    """The two lines for one seed. Generation failing is the generator's own
    bug, and it still owes the parent a result: a missing line would be read
    as a dead worker and blamed on the engine."""
    try:
        case = G.gen(seed)
        sql = G.render(case.query)
    except Exception as e:  # noqa: BLE001
        broke = {"$error": f"gen:{type(e).__name__}: {e}"}
        _emit({"event": "case", "seed": seed, "sql": "", "tags": [], "case": broke})
        _emit(_skip(seed, f"gen:{type(e).__name__}", str(e)))
        return
    try:
        event = _case_event(seed, case, sql)
    except Exception as e:  # noqa: BLE001 — do not evaluate unrecordable inputs
        broke = {"$error": f"encode:{type(e).__name__}: {e}"}
        _emit(
            {
                "event": "case",
                "seed": seed,
                "sql": sql,
                "tags": case.tags,
                "case": broke,
            }
        )
        skipped = _skip(seed, f"encode:{type(e).__name__}", str(e))
        skipped["sql"] = sql
        _emit(skipped)
        return
    _emit(event)
    _emit({"event": "result", **run_case_json(case)})


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        run_seed(int(line))


if __name__ == "__main__":
    main()
