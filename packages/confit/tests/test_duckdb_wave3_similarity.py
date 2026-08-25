"""Wave-3 similarity builtins vs the duckdb oracle.

Pins: packages/confit/docs/specs/2026-07-26-wave3-builtin-pins.md — the
levenshtein / editdist3 / damerau_levenshtein / jaccard / hamming /
mismatches family. Unit is raw UTF-8 BYTES everywhere; all BIGINT except
jaccard (DOUBLE); jaccard/hamming trap on empty or length-mismatched
inputs but NULL pre-empts every trap.
"""

from __future__ import annotations

import pytest
from test_duckdb_interpreter import duck_check

AB = {"a": "str", "b": "str"}


def test_levenshtein_unit_is_utf8_bytes():
    # 'é' is 2 bytes, U+1D11E is 4 bytes, U+0301 (combining) is 2 bytes.
    duck_check(
        "SELECT levenshtein(a, b) AS d FROM __THIS__",
        AB,
        [
            {"a": "é", "b": "e"},  # 2: substitute+insert at byte level
            {"a": "\U0001d11ea", "b": "a"},  # 4: drop four bytes
            {"a": "é", "b": "é"},  # 3: [65,CC,81] vs [C3,A9]
            {"a": "é", "b": "e"},  # 2: drop the combining bytes
        ],
    )


def test_levenshtein_empties_and_case():
    duck_check(
        "SELECT levenshtein(a, b) AS d FROM __THIS__",
        AB,
        [
            {"a": "", "b": ""},  # 0
            {"a": "", "b": "abc"},  # 3
            {"a": "abc", "b": ""},  # 3
            {"a": "a", "b": "A"},  # 1: case-sensitive
        ],
    )


def test_editdist3_is_levenshtein_pairwise():
    duck_check(
        "SELECT levenshtein(a, b) AS lv, editdist3(a, b) AS e3 FROM __THIS__",
        AB,
        [
            {"a": "kitten", "b": "sitting"},
            {"a": "é", "b": "e"},
            {"a": "", "b": "xy"},
            {"a": "ab", "b": "ba"},
        ],
    )


def test_damerau_is_unrestricted_dl():
    duck_check(
        "SELECT damerau_levenshtein(a, b) AS d FROM __THIS__",
        AB,
        [
            {"a": "ca", "b": "abc"},  # 2 unrestricted (OSA would say 3)
            {"a": "ab", "b": "ba"},  # 1: one transposition
            {"a": "a cat", "b": "an act"},  # 2
            {"a": "", "b": "abc"},  # 3: empties accepted
        ],
    )


def test_jaccard_single_byte_sets():
    # Set semantics over single BYTES: order and multiplicity ignored;
    # 'é'/'è' share their 0xC3 lead byte -> 1/3. DOUBLE output (repr
    # in the harness would catch an int 1 vs float 1.0 drift).
    duck_check(
        "SELECT jaccard(a, b) AS j FROM __THIS__",
        AB,
        [
            {"a": "ab", "b": "ba"},  # 1.0
            {"a": "aab", "b": "ab"},  # 1.0
            {"a": "abc", "b": "abd"},  # 0.5
            {"a": "é", "b": "è"},  # 1/3
            {"a": "a", "b": "A"},  # 0.0: case-sensitive
        ],
    )


def test_mismatches_is_hamming_pairwise():
    # Byte unit again: 'é' vs 'è' differ in one of two bytes.
    duck_check(
        "SELECT hamming(a, b) AS h, mismatches(a, b) AS m FROM __THIS__",
        AB,
        [
            {"a": "é", "b": "è"},  # 1
            {"a": "abcd", "b": "abcd"},  # 0
            {"a": "abcd", "b": "azcx"},  # 2
            {"a": "A", "b": "a"},  # 1: case-sensitive
        ],
    )


def test_hamming_byte_length_mismatch_traps():
    # 'é' is 2 bytes vs 1 -> trap, and the message says Mismatch.
    with pytest.raises(
        Exception, match="Mismatch Function: Strings must be of equal length!"
    ):
        duck_check(
            "SELECT hamming(a, b) AS h FROM __THIS__",
            AB,
            [{"a": "é", "b": "e"}],
        )


def test_hamming_both_empty_traps_length_gt_zero():
    with pytest.raises(
        Exception, match="Mismatch Function: Strings must be of length > 0!"
    ):
        duck_check(
            "SELECT hamming(a, b) AS h FROM __THIS__",
            AB,
            [{"a": "", "b": ""}],
        )


def test_jaccard_empty_either_side_traps():
    for row in [{"a": "", "b": "a"}, {"a": "a", "b": ""}]:
        with pytest.raises(Exception, match="Jaccard Function: An argument too short!"):
            duck_check(
                "SELECT jaccard(a, b) AS j FROM __THIS__",
                AB,
                [row],
            )


def test_null_pre_empts_every_trap():
    # NULL-strict on every argument: these exact rows would trap if the
    # traps ran before the NULL check — they must yield NULL instead.
    duck_check(
        "SELECT jaccard(a, b) AS j, hamming(a, b) AS h,"
        " levenshtein(a, b) AS lv, damerau_levenshtein(a, b) AS dl"
        " FROM __THIS__",
        {"a": "str?", "b": "str?"},
        [
            {"a": None, "b": ""},  # jaccard(NULL,'') / hamming(NULL,'')
            {"a": "", "b": None},
            {"a": None, "b": None},
        ],
    )


def test_where_guard_masks_trap_rows():
    # Filtered rows must never evaluate the trapping calls.
    duck_check(
        "SELECT hamming(a, b) AS h, jaccard(a, b) AS j FROM __THIS__ WHERE ok",
        {"a": "str", "b": "str", "ok": "bool"},
        [
            {"a": "ab", "b": "xy", "ok": True},
            {"a": "é", "b": "e", "ok": False},  # hamming length trap
            {"a": "", "b": "", "ok": False},  # both functions trap
        ],
    )


def test_embedded_nul_is_an_ordinary_byte():
    duck_check(
        "SELECT levenshtein(a, b) AS lv, hamming(a, c) AS h FROM __THIS__",
        {"a": "str", "b": "str", "c": "str"},
        [{"a": "a\x00b", "b": "ab", "c": "axb"}],
    )
