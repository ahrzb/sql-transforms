"""The OPT_EMULATED cases of 2026-10-05 (nightly #305 and campaigns).

OPT_EMULATED is a finding (docs/oracle/04): confit answered like optimizer-on
DuckDB where the optimizer-off reference differs. Each class here is fixed,
refused by name, or excused by its ruling.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow as pa
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from fuzz import exclusions as X  # noqa: E402
from fuzz.parity import assert_parity  # noqa: E402

T = pa.table(
    {
        "k": pa.array([1, 2], pa.int64()),
        "v": pa.array([9223372036854775807, 3], pa.int64()),
    }
)
NO_MATCH = pa.table(
    {
        "k": pa.array([100, 200], pa.int64()),
        "w": pa.array([9223372036854775807, 5], pa.int64()),
    }
)


@pytest.mark.parametrize(
    "on",
    [
        "t.k = s.k AND t.v < s.w * 2",
        "t.k = s.k AND t.v * 2 < s.w",
        "t.v < s.w * 2",
    ],
)
def test_a_trapping_join_condition_between_the_sides_refuses(on):
    # DuckDB evaluates each operand of such a comparison over its whole
    # table, matched or not (measured: traps with no key match); confit
    # evaluates a residual per matched pair (seed 4291817).
    v = assert_parity(
        f"SELECT 1 AS o FROM __THIS__ t JOIN s ON {on}",
        T,
        statics={"s": NO_MATCH},
        expect="REFUSED",
    )
    assert "comparison between the two sides" in v.detail


@pytest.mark.parametrize(
    "on",
    [
        "t.k = s.k AND t.v + s.w > 0",
        "t.k = s.k AND (t.v < s.w * 2 OR t.k = 1)",
        "t.k = s.k AND t.v < s.w",
    ],
)
def test_a_per_pair_join_condition_still_serves(on):
    assert_parity(
        f"SELECT 1 AS o FROM __THIS__ t JOIN s ON {on}", T, statics={"s": NO_MATCH}
    )


def test_nullif_null_over_a_trapping_operand_refuses():
    # DuckDB evaluates x in nullif(NULL, x) and traps on it (seed 992226).
    v = assert_parity(
        "SELECT nullif(NULL, v + v) AS o FROM __THIS__", T, expect="REFUSED"
    )
    assert "nullif(NULL, x)" in v.detail
    assert_parity("SELECT nullif(NULL, k) AS o FROM __THIS__", T)


def test_the_empty_static_witness_pads_a_decimal_column():
    # Seed 1159605: the witness row could not be built for a DECIMAL static
    # column, so the ruled empty-static case stayed a finding.
    assert X._plain_value(pa.decimal128(9, 4)) == 0
