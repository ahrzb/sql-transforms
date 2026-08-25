"""Wave-3 builtin pins: VARCHAR subscripts, codepoint probes, strip_accents.

Pins measured against duckdb 1.5.5 per
packages/confit/docs/specs/2026-07-26-wave3-builtin-pins.md:
- array_extract/list_extract on VARCHAR: 1-based CODEPOINT indexing,
  0 / out-of-range -> '' (empty string, not NULL), negative from end,
  NULL-strict, list_extract alias-identical.
- array_slice/list_slice on VARCHAR: both-ends-inclusive 1-based, negative
  from end, clamping at both ends, reversed/out-of-range -> '', NULL bound
  -> NULL (not an open bound), alias-identical, splits combining sequences.
  Step slice does not prepare (build-time reject).
- unicode/ord/ascii: first codepoint; '' -> -1 for unicode/ord but 0 for
  ascii (the sole divergence); NULL-strict.
- bit_length = 8 * strlen (UTF-8 bytes).
- strip_accents: utf8proc-style accent removal — decomposes-and-drops
  combining marks, rewrites accent-free lookalikes (U+212A -> 'K'),
  composes Hangul jamo, deletes Indic vowel signs; NUL truncates output
  only when the input has non-ASCII characters; idempotent; NULL-strict.

No runtime-trapping op in this family (out-of-range yields ''), so the
lazy-guard convention has nothing to witness here.
"""

from __future__ import annotations

import duckdb
import pytest
from test_duckdb_interpreter import duck_check


def test_array_extract_codepoints_negative_and_alias():
    # 1-based codepoint indexing on 'éa' plus an astral char; negative
    # indexes count from the end; i=0, -len-1 and len+1 all yield ''.
    # list_extract must be alias-identical (paired column).
    duck_check(
        "SELECT s, i, array_extract(s, i) AS e, list_extract(s, i) AS l FROM __THIS__",
        {"s": "str", "i": "int"},
        [
            {"s": "éa", "i": 1},
            {"s": "éa", "i": 2},
            {"s": "éa", "i": 0},
            {"s": "éa", "i": -1},
            {"s": "éa", "i": -2},
            {"s": "éa", "i": -3},
            {"s": "éa", "i": 3},
            {"s": "a\U0001f600b", "i": 2},
            {"s": "a\U0001f600b", "i": -2},
        ],
    )


def test_array_extract_out_of_range_is_empty_string_not_null():
    # The '' rows asserted explicitly: IS NOT NULL must be TRUE both ways.
    duck_check(
        "SELECT array_extract(s, 99) AS hi, array_extract(s, -99) AS lo, "
        "array_extract(s, 99) IS NOT NULL AS hi_nn, "
        "array_extract(s, -99) IS NOT NULL AS lo_nn FROM __THIS__",
        {"s": "str"},
        [{"s": "éa"}],
    )


def test_array_extract_null_strict():
    duck_check(
        "SELECT array_extract(s, i) AS e, list_extract(s, i) AS l FROM __THIS__",
        {"s": "str?", "i": "int?"},
        [{"s": None, "i": 1}, {"s": "abc", "i": None}],
    )


def test_array_slice_inclusive_clamping_and_alias():
    # Both-ends-inclusive 1-based; negative from end; a<=0 clamps to start;
    # b>len clamps to end; reversed / fully out-of-range -> ''.
    duck_check(
        "SELECT a, b, array_slice(s, a, b) AS r, list_slice(s, a, b) AS r2 "
        "FROM __THIS__",
        {"s": "str", "a": "int", "b": "int"},
        [
            {"s": "hello", "a": 2, "b": 4},  # 'ell'
            {"s": "hello", "a": -3, "b": -1},  # 'llo'
            {"s": "hello", "a": 2, "b": -2},  # 'ell'
            {"s": "hello", "a": 0, "b": 3},  # 'hel'
            {"s": "hello", "a": 4, "b": 100},  # 'lo'
            {"s": "hello", "a": 4, "b": 2},  # ''
            {"s": "hello", "a": 10, "b": 20},  # ''
        ],
    )


def test_array_slice_null_bound_is_null_not_open():
    # A NULL bound nulls the result — it is NOT treated as an open bound.
    duck_check(
        "SELECT array_slice(s, a, b) AS r, "
        "array_slice(s, a, b) IS NULL AS r_null FROM __THIS__",
        {"s": "str", "a": "int?", "b": "int?"},
        [
            {"s": "hello", "a": None, "b": 3},
            {"s": "hello", "a": 2, "b": None},
        ],
    )


def test_array_slice_and_extract_split_combining_sequence():
    # Codepoint semantics split 'e'+U+0301: the mark travels alone.
    duck_check(
        "SELECT array_slice(s, 1, 1) AS head, array_slice(s, 2, 3) AS tail, "
        "array_extract(s, 2) AS mark FROM __THIS__",
        {"s": "str"},
        [{"s": "éx"}],
    )


def test_array_slice_with_step_fails_to_build():
    with pytest.raises(ValueError, match="step"):
        duck_check(
            "SELECT array_slice(s, 1, 3, 2) AS r FROM __THIS__",
            {"s": "str"},
            [{"s": "hello"}],
        )


def test_unicode_ord_ascii_family_on_same_rows():
    # unicode/ord: first codepoint, '' -> -1. ascii: identical except
    # '' -> 0 — the sole divergence. NULL-strict for all three.
    duck_check(
        "SELECT unicode(s) AS u, ord(s) AS o, ascii(s) AS a FROM __THIS__",
        {"s": "str?"},
        [
            {"s": ""},
            {"s": "abc"},
            {"s": "\U0001f600x"},
            {"s": "é"},
            {"s": None},
        ],
    )


def test_bit_length_is_eight_times_strlen():
    # Witnesses: 'é' -> 16, astral -> 32, '' -> 0; plus an in-query sweep
    # asserting bit_length(s) = 8 * strlen(s) on every row.
    duck_check(
        "SELECT bit_length(s) AS b, "
        "bit_length(s) = 8 * strlen(s) AS agrees FROM __THIS__",
        {"s": "str"},
        [
            {"s": "é"},
            {"s": "\U0001f600"},
            {"s": ""},
            {"s": "abc"},
            {"s": "é"},
            {"s": "aé\U0001f600"},
        ],
    )


def test_strip_accents_witnesses_and_idempotence():
    # 'é' -> 'e'; U+212A KELVIN SIGN -> 'K' (accent-free rewrite);
    # combining mark deleted (attached and lone); Indic vowel sign
    # deleted; Hangul jamo compose to a syllable; NULL-strict.
    # The idem column pins idempotence on every row.
    duck_check(
        "SELECT strip_accents(s) AS r, "
        "strip_accents(strip_accents(s)) = strip_accents(s) AS idem "
        "FROM __THIS__",
        {"s": "str?"},
        [
            {"s": "é"},
            {"s": "K"},
            {"s": "é"},
            {"s": "́"},
            {"s": "가"},  # -> U+AC00
            {"s": "का"},  # vowel sign deleted -> U+0915
            {"s": "ा"},  # lone vowel sign -> ''
            {"s": None},
        ],
    )


def test_strip_accents_nul_context_quirk():
    # ASCII-only input passes NUL through verbatim; any non-ASCII char in
    # the input flips to the utf8proc path, which truncates at the NUL.
    duck_check(
        "SELECT strip_accents(s) AS r FROM __THIS__",
        {"s": "str"},
        [
            {"s": "a\x00b"},  # unchanged: 'a\x00b'
            {"s": "a\x00é"},  # truncated: 'a'
        ],
    )


def test_strip_accents_census_sample():
    # Deterministic sample (~300) of every codepoint strip_accents changes
    # in planes 0-1, extracted from the oracle itself, replayed through
    # both engines as single-char rows.
    con = duckdb.connect()
    cps = [
        r[0]
        for r in con.execute(
            "SELECT i FROM generate_series(1, 131071) t(i) "
            "WHERE i NOT BETWEEN 55296 AND 57343 "
            "AND strip_accents(chr(i::INTEGER)) != chr(i::INTEGER) "
            "ORDER BY i"
        ).fetchall()
    ]
    assert len(cps) > 1000  # the map is big; a tiny census means a bad probe
    sample = cps[:: max(1, len(cps) // 300)]
    duck_check(
        "SELECT s, strip_accents(s) AS r FROM __THIS__",
        {"s": "str"},
        [{"s": chr(c)} for c in sample],
    )
