"""The displayed corpus count has one home, and the docs quote it.

`MATCH_FLOOR` is the shipped ratchet; `docs/reports/corpus-counts.json` is
the dated measurement `scripts/corpus_counts.py` records. The docs that show
a headline number must show that file's number with that file's date, and
the recorded count can never sit below the floor.
"""

from __future__ import annotations

import json
from pathlib import Path

from test_corpus_replay import MATCH_FLOOR

ROOT = Path(__file__).resolve().parents[3]
COUNTS = json.loads(
    (ROOT / "packages/confit/docs/reports/corpus-counts.json").read_text("utf-8")
)
DISPLAYS = (
    "README.md",
    "packages/confit/README.md",
    "packages/confit/docs/known-limitations.md",
)


def test_the_recorded_count_respects_the_floor():
    assert COUNTS["fail"] == 0
    assert COUNTS["match"] >= MATCH_FLOOR
    assert COUNTS["match"] + COUNTS["unsupported"] == COUNTS["total"]


def test_every_display_quotes_the_recorded_count_and_its_date():
    want = f"{COUNTS['match']} of {COUNTS['total']}"
    for rel in DISPLAYS:
        text = (ROOT / rel).read_text("utf-8")
        assert want in text, f"{rel} does not show {want}"
        assert COUNTS["date"] in text, f"{rel} does not date its count"
