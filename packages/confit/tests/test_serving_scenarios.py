"""Standing parity gate for the serving-bench scenarios.

Each scenario (benchmarks/serving_scenarios/) is a realistic wide-table
feature-engineering inference path. The specializer's output must equal
DuckDB itself AND the handcrafted Python twin, exactly, on 300 seeded rows.
"""

from __future__ import annotations

import pytest

from benchmarks import serving_scenarios as sc


@pytest.mark.parametrize("name", sc.NAMES)
def test_scenario_three_way_parity(name):
    mod = sc.load(name)
    problems = sc.verify_parity(mod, n=300)
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("name", sc.NAMES)
def test_scenario_is_wide_and_deterministic(name):
    mod = sc.load(name)
    assert mod.N_INPUT_COLS == len(mod.ROW_SCHEMA)
    rows_a = mod.make_rows(sc.SEED, 50)
    rows_b = mod.make_rows(sc.SEED, 50)
    assert rows_a == rows_b, "make_rows must be deterministic"
    statics_a = mod.make_statics(sc.SEED)
    statics_b = mod.make_statics(sc.SEED)
    assert {k: t.to_pylist() for k, t in statics_a.items()} == {
        k: t.to_pylist() for k, t in statics_b.items()
    }, "make_statics must be deterministic"
