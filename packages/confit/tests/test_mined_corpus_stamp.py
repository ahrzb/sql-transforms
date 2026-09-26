"""A future mining run stamps what produced the corpus's expected rows.

The checked-in corpus predates the stamp and has none; it is not given one
after the fact. What is pinned here is the stamp a run WOULD write, built
without the duckdb/ clone.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[3]


def _miner():
    path = ROOT / "scripts" / "mine_duckdb_corpus.py"
    spec = importlib.util.spec_from_file_location("mine_duckdb_corpus", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_stamp_names_the_reference_and_its_optimizer_state():
    miner = _miner()
    stamp = miner.provenance(kept=3, seen=5, files=2)
    assert stamp["duckdb"] == duckdb.__version__
    assert stamp["settings"]["optimizer"] == "on"
    for key in ("date", "duckdb_clone_revision", "miner_revision", "miner_sha256"):
        assert stamp[key], key
    assert (stamp["kept"], stamp["seen"], stamp["files"]) == (3, 5, 2)
    assert miner.STAMP.name == "duckdb_mined.provenance.json"

