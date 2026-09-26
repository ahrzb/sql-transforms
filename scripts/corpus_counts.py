"""Record the dated corpus match count: the one home for the headline number.

`MATCH_FLOOR` in `tests/test_corpus_replay.py` is the shipped ratchet — a
constant a regression must not go below. The number the docs DISPLAY is a
measurement, and a measurement has a date, an engine revision and a
reference. This writes it, from the same replay the gate runs, to

    packages/confit/docs/reports/corpus-counts.json

    uv run python scripts/corpus_counts.py
    git diff        # the new count, reviewable next to the change that moved it

Docs quote that file's numbers with its date; `test_corpus_counts.py` keeps
them from drifting apart.
"""

import datetime
import json
import subprocess
import sys
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "packages/confit/tests"
OUT = ROOT / "packages/confit/docs/reports/corpus-counts.json"

sys.path.insert(0, str(TESTS))
from test_corpus_replay import MATCH_FLOOR, replay_counts  # noqa: E402


def _rev() -> str:
    run = subprocess.run(  # noqa: S603 — fixed argv
        ["git", "rev-parse", "HEAD"],  # noqa: S607
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return run.stdout.strip() or "unknown"


def main() -> None:
    counts, fails, total = replay_counts()
    record = {
        "date": datetime.date.today().isoformat(),
        "engine_revision": _rev(),
        # The replay compares against the rows the miner RECORDED, which a
        # fresh optimizer-on connection produced and nothing stamped (claim:
        # mined-corpus-provenance) -- not a live optimizer-off reading.
        "reference": "expected rows as mined (optimizer-on, unstamped); "
        f"installed duckdb {duckdb.__version__}",
        "corpus": "packages/confit/tests/corpus/duckdb_mined.jsonl",
        "total": total,
        "match": counts["match"],
        "unsupported": counts["unsupported"],
        "fail": counts["FAIL"],
        "floor": MATCH_FLOOR,
    }
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2))
    if fails:
        sys.exit(f"{len(fails)} FAILs: fix them before recording a count")


if __name__ == "__main__":
    main()
