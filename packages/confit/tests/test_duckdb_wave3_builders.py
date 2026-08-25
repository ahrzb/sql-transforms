"""Wave-3 string-builder builtins vs the duckdb oracle.

Pins: packages/confit/docs/specs/2026-07-26-wave3-builtin-pins.md
Family: repeat, lpad, rpad, replace, translate, concat_ws, ucase, lcase.

Everything runs through `duck_check` (both engines, repr-multiset compare)
on columns of __THIS__ — literals only where the pin is about literals
(constant lengths for the pad-trap split).
"""

from __future__ import annotations

import pytest
from test_duckdb_interpreter import duck_check

# ---------------------------------------------------------------- repeat


def test_repeat_zero_negative_and_multibyte():
    # n=0 and n<0 both yield '' silently; multi-byte codepoints survive.
    duck_check(
        "SELECT repeat(s, n) AS r FROM __THIS__",
        {"s": "str", "n": "int"},
        [
            {"s": "ab", "n": 0},
            {"s": "ab", "n": -3},
            {"s": "é☃", "n": 2},
            {"s": "ab", "n": 3},
        ],
    )


def test_repeat_null_strict():
    duck_check(
        "SELECT repeat(s, n) AS r FROM __THIS__",
        {"s": "str?", "n": "int?"},
        [
            {"s": None, "n": 2},
            {"s": "ab", "n": None},
            {"s": None, "n": None},
            {"s": "ab", "n": 2},
        ],
    )


# ------------------------------------------------------------- lpad/rpad


def test_lpad_counts_codepoints_not_bytes():
    # 'éé' is 2 codepoints / 4 UTF-8 bytes: lpad to 4 must ADD two pads.
    duck_check(
        "SELECT lpad(s, 4, 'x') AS r FROM __THIS__",
        {"s": "str"},
        [{"s": "éé"}, {"s": "ab"}],
    )


def test_pad_truncation_keeps_first_codepoints():
    # Headline: rpad('abcdef',3,'x')='abc' — truncation keeps the FIRST l
    # codepoints for BOTH directions. Plus l<=0 -> '' and l=length identity.
    # DuckDB only binds lpad(VARCHAR, INTEGER, VARCHAR): cast the BIGINT col.
    duck_check(
        "SELECT lpad(s, CAST(l AS INTEGER), 'x') AS lp,"
        " rpad(s, CAST(l AS INTEGER), 'x') AS rp FROM __THIS__",
        {"s": "str", "l": "int"},
        [
            {"s": "abcdef", "l": 3},
            {"s": "éab☃", "l": 2},
            {"s": "abc", "l": 0},
            {"s": "abc", "l": -2},
            {"s": "abc", "l": 3},
        ],
    )


def test_pad_cycling_and_multibyte_pad_cut():
    # lpad('a',6,'xyz')='xyzxya', rpad='axyzxy'; multi-byte pad is cut at a
    # codepoint boundary: lpad('a',3,'é☃')='é☃a', rpad='aé☃'.
    duck_check(
        "SELECT lpad(s, CAST(l AS INTEGER), p) AS lp,"
        " rpad(s, CAST(l AS INTEGER), p) AS rp FROM __THIS__",
        {"s": "str", "l": "int", "p": "str"},
        [
            {"s": "a", "l": 6, "p": "xyz"},
            {"s": "a", "l": 3, "p": "é☃"},
        ],
    )


def test_pad_null_strict_all_three_args():
    duck_check(
        "SELECT lpad(s, CAST(l AS INTEGER), p) AS lp,"
        " rpad(s, CAST(l AS INTEGER), p) AS rp FROM __THIS__",
        {"s": "str?", "l": "int?", "p": "str?"},
        [
            {"s": None, "l": 3, "p": "x"},
            {"s": "a", "l": None, "p": "x"},
            {"s": "a", "l": 3, "p": None},
            {"s": "a", "l": 3, "p": "x"},
        ],
    )


def test_pad_empty_pad_shrink_is_silent():
    # Empty pad is fine as long as no growth is needed: lpad('abc',2,'')='ab'.
    duck_check(
        "SELECT lpad(s, 2, p) AS lp, rpad(s, 2, p) AS rp FROM __THIS__",
        {"s": "str", "p": "str"},
        [{"s": "abc", "p": ""}],
    )


def test_lpad_empty_pad_growth_traps():
    # Same arguments except l now requires growth: DATA-DEPENDENT trap.
    with pytest.raises(Exception, match="Insufficient padding in LPAD"):
        duck_check(
            "SELECT lpad(s, 5, p) AS r FROM __THIS__",
            {"s": "str", "p": "str"},
            [{"s": "abc", "p": ""}],
        )


def test_rpad_empty_pad_growth_traps():
    with pytest.raises(Exception, match="Insufficient padding in RPAD"):
        duck_check(
            "SELECT rpad(s, 5, p) AS r FROM __THIS__",
            {"s": "str", "p": "str"},
            [{"s": "abc", "p": ""}],
        )


def test_pad_empty_trap_where_guard_is_lazy():
    # The would-trap row ('abc' needs growth) is filtered out / branched away:
    # neither engine may evaluate the pad on it.
    duck_check(
        "SELECT lpad(s, 5, p) AS r FROM __THIS__ WHERE length(s) >= 5",
        {"s": "str", "p": "str"},
        [{"s": "abcdef", "p": ""}, {"s": "abc", "p": ""}],
    )
    duck_check(
        "SELECT CASE WHEN length(s) >= 5 THEN rpad(s, 5, p)"
        " ELSE NULL END AS r FROM __THIS__",
        {"s": "str", "p": "str"},
        [{"s": "abcdef", "p": ""}, {"s": "abc", "p": ""}],
    )


def test_pad_null_pad_preempts_empty_trap():
    # CRITICAL masking pin: lpad('abc',5,NULL) is NULL, NOT the padding trap —
    # NULL-strictness wins over the insufficient-padding check.
    duck_check(
        "SELECT lpad(s, 5, p) AS lp, rpad(s, 5, p) AS rp FROM __THIS__",
        {"s": "str", "p": "str?"},
        [{"s": "abc", "p": None}],
    )


# --------------------------------------------------------------- replace


def test_replace_single_pass_and_empty_needle():
    duck_check(
        "SELECT replace(s, n, r) AS o FROM __THIS__",
        {"s": "str", "n": "str", "r": "str"},
        [
            {"s": "abc", "n": "", "r": "x"},  # empty needle: strict no-op
            {"s": "aaa", "n": "aa", "r": "b"},  # leftmost non-overlap: 'ba'
            {"s": "abab", "n": "ab", "r": "ba"},  # single pass: 'baba'
            {"s": "ABC", "n": "b", "r": "x"},  # case-sensitive: unchanged
        ],
    )


def test_replace_null_strict():
    duck_check(
        "SELECT replace(s, n, r) AS o FROM __THIS__",
        {"s": "str?", "n": "str?", "r": "str?"},
        [
            {"s": None, "n": "a", "r": "b"},
            {"s": "a", "n": None, "r": "b"},
            {"s": "a", "n": "a", "r": None},
            {"s": "aa", "n": "a", "r": "b"},
        ],
    )


# ------------------------------------------------------------- translate


def test_translate_per_codepoint_semantics():
    duck_check(
        "SELECT translate(s, f, t) AS o FROM __THIS__",
        {"s": "str", "f": "str", "t": "str"},
        [
            # from longer than to: unmatched from-chars are DELETED -> 'x'
            {"s": "abcb", "f": "abc", "t": "x"},
            # duplicate in from: FIRST mapping wins (a->x, not a->y)
            {"s": "abc", "f": "aa", "t": "xy"},
            # to='': every from-char deleted
            {"s": "abca", "f": "a", "t": ""},
            # from='': identity
            {"s": "abc", "f": "", "t": "xyz"},
            # per-CODEPOINT, not per-byte, on multi-byte chars
            {"s": "éa☃", "f": "é☃", "t": "xy"},
        ],
    )


def test_translate_null_strict():
    duck_check(
        "SELECT translate(s, f, t) AS o FROM __THIS__",
        {"s": "str?", "f": "str?", "t": "str?"},
        [
            {"s": None, "f": "a", "t": "b"},
            {"s": "a", "f": None, "t": "b"},
            {"s": "a", "f": "a", "t": None},
            {"s": "ab", "f": "a", "t": "b"},
        ],
    )


# ------------------------------------------------------------- concat_ws


def test_concat_ws_null_skipping_and_null_separator():
    # NULL value args are SKIPPED with their separator; NULL separator (from a
    # nullable COLUMN) nulls the whole result; ALL values NULL -> '' not NULL.
    duck_check(
        "SELECT concat_ws(sep, a, b, c) AS o FROM __THIS__",
        {"sep": "str?", "a": "str?", "b": "str?", "c": "str?"},
        [
            {"sep": ",", "a": "x", "b": None, "c": "y"},
            {"sep": ",", "a": None, "b": None, "c": None},
            {"sep": None, "a": "x", "b": "y", "c": "z"},
            {"sep": "-", "a": "x", "b": "y", "c": "z"},
        ],
    )


def test_concat_ws_value_args_implicitly_cast():
    # int renders '7'; double renders like DuckDB varchar-casts doubles
    # (1.5 / -0.0 keeps its sign / 1e20 -> '1e+20'); bool -> 'true'/'false'.
    duck_check(
        "SELECT concat_ws(',', i, d, b) AS o FROM __THIS__",
        {"i": "int", "d": "float", "b": "bool"},
        [
            {"i": 7, "d": 1.5, "b": True},
            {"i": 0, "d": -0.0, "b": False},
            {"i": -3, "d": 1e20, "b": True},
        ],
    )


def test_concat_ws_separator_does_not_implicitly_cast():
    # An int separator is a BIND error in DuckDB; the engine must reject the
    # query at build time, not coerce.
    with pytest.raises(ValueError, match="no function matches concat_ws"):
        duck_check(
            "SELECT concat_ws(i, s, s) AS o FROM __THIS__",
            {"i": "int", "s": "str"},
            [{"i": 5, "s": "a"}],
        )


# ------------------------------------------------------------ ucase/lcase


def test_ucase_lcase_are_upper_lower_on_unicode():
    # Pair the alias with its canonical spelling in ONE query over a sweep
    # incl. 'ß' (uppercases to U+1E9E), U+0130 'İ', U+212A KELVIN SIGN.
    duck_check(
        "SELECT ucase(s) AS u1, upper(s) AS u2,"
        " lcase(s) AS l1, lower(s) AS l2 FROM __THIS__",
        {"s": "str"},
        [
            {"s": "ß"},
            {"s": "İ"},
            {"s": "K"},
            {"s": "Straße"},
            {"s": "aBcÉ é"},
            {"s": ""},
        ],
    )
