"""Standing differential regexp fuzzer: duckdb vs the engine.

Wave B translated DuckDB/RE2 patterns to rust-regex behind a measured reject
list (packages/confit/docs/specs/2026-07-27-waveB-regexp-pins.md).
The one-time battery had 98 entries; the residual risk is constructs slipping
through translate_pattern's pass-through path. This test generates patterns
from a grammar biased toward the divergence-prone axes (Perl classes in/out
of char classes, inline flags, alternation, bounded repetition incl. the 1000
cap, escapes, Unicode properties, char-class edge shapes like POSIX elements
and rust set-notation, options strings, replacement templates) and asserts,
per case:

  - duckdb ok  + engine ok      -> identical multiset of rows
  - duckdb ok  + engine rejects -> fine (conservative bind-time reject)
  - duckdb err + engine rejects -> fine (identically rejected)
  - duckdb err + engine serves  -> FAIL (wrong-answer risk)

A failure = a construct for the retrans.rs reject list + a pin note in the
spec addendum (pins discipline). Deterministic: fixed default seed; override
with REGEXP_FUZZ_SEED / REGEXP_FUZZ_N for exploratory deep runs. The failure
message carries seed, case index, and the SQL — rerun with that seed to
reproduce.
"""

from __future__ import annotations

import os
import random
from collections import Counter

from confit import DuckDBInferFn, compare
from confit.oracle import Trap
from test_duckdb_interpreter import _row_schema, static

SEED = int(os.environ.get("REGEXP_FUZZ_SEED", "20260727"))
# ~15ms/case (engine bind dominates); 250 lands in the ~2-5s budget.
N = int(os.environ.get("REGEXP_FUZZ_N", "250"))

SCHEMA = {"a": "int", "s": "str?"}
# Divergence-prone subjects: ASCII words/digits, Unicode digits (٣٤), accents,
# (?i)-fold traps (KELVIN K, sharp-s), newlines/tabs for (?m)/(?s)/\s, empty,
# NULL, and class-metachar literals (&, -, [, ]).
ROWS = [
    {"a": 1, "s": "hello world"},
    {"a": 2, "s": "abc123def"},
    {"a": 3, "s": None},
    {"a": 4, "s": ""},
    {"a": 5, "s": "héllo ٣٤"},
    {"a": 6, "s": "HELLO K ß"},
    {"a": 7, "s": "a\nb\tc d"},
    {"a": 8, "s": "x&&y--z[a]"},
]

_LITS = "abczXZ019 h"
_ULITS = "éß٣K"  # é, sharp-s, arabic-indic 3, KELVIN sign
_PERL = [r"\d", r"\D", r"\w", r"\W", r"\s", r"\S"]
_PROPS = [r"\p{L}", r"\p{Lu}", r"\p{Nd}", r"\P{L}", r"\pL"]
_ESCAPES = [r"\n", r"\t", r"\x41", r"\x{212A}", r"\052", r"\.", r"\*", r"\-", r"\&"]
_ANCHORS = ["^", "$", r"\A", r"\z", r"\b", r"\B"]
_FLAGS = ["(?i)", "(?s)", "(?m)", "(?U)", "(?-i)", "(?is)"]
# Mostly reject-or-both-error shapes: keep them flowing through the contract.
_WEIRD = [
    r"\Qa.\E",
    r"A",
    r"\C",
    r"\1",
    r"\Z",
    "a{2,1}",
    "a{1, 3}",
    "a{,3}",
    "{2}",
    "*",
]
_CLASS_RANGES = ["a-f", "0-7", "A-Z", "٠-٩", "é-ü"]
# rust-regex set notation / POSIX / nesting — literal chars to RE2.
_CLASS_EDGES = [
    "[:alpha:]",
    "[:^digit:]",
    "[",
    "[b]",
    "&&",
    "--",
    "~~",
    "-",
    r"\]",
    r"\b",
]


def _class(rng: random.Random) -> str:
    body = ""
    if rng.random() < 0.3:
        body += "^"
    if rng.random() < 0.15:
        body += "]"
    for _ in range(rng.randint(1, 3)):
        r = rng.random()
        if r < 0.30:
            body += rng.choice(_LITS.strip() + _ULITS)
        elif r < 0.50:
            body += rng.choice(_CLASS_RANGES)
        elif r < 0.70:
            body += rng.choice(_PERL)
        elif r < 0.85:
            body += rng.choice(_CLASS_EDGES)
        else:
            body += rng.choice(_ESCAPES + [r"\p{L}"])
    return f"[{body}]"


def _atom(rng: random.Random, depth: int) -> str:
    r = rng.random()
    if r < 0.25:
        return rng.choice(_LITS)
    if r < 0.33:
        return rng.choice(_ULITS)
    if r < 0.45:
        return rng.choice(_PERL)
    if r < 0.52:
        return rng.choice(_PROPS)
    if r < 0.57:
        return rng.choice(_ESCAPES)
    if r < 0.62:
        return rng.choice(_ANCHORS)
    if r < 0.67:
        return "."
    if r < 0.79:
        return _class(rng)
    if r < 0.83:
        return rng.choice(_WEIRD)
    if depth >= 2:
        return rng.choice(_LITS)
    inner = _pattern(rng, depth + 1)
    kind = rng.random()
    if kind < 0.4:
        return f"({inner})"
    if kind < 0.6:
        return f"(?:{inner})"
    if kind < 0.75:
        return f"(?P<g{rng.randint(0, 2)}>{inner})"  # dup names on purpose
    if kind < 0.85:
        return f"(?i:{inner})"
    return f"(?<g{rng.randint(0, 2)}>{inner})"  # angle group: pinned reject


def _quantified(rng: random.Random, depth: int) -> str:
    a = _atom(rng, depth)
    r = rng.random()
    if r < 0.55:
        return a
    if r < 0.75:
        q = rng.choice(["*", "+", "?", "*?", "+?", "??"])
    elif r < 0.9:
        lo = rng.randint(0, 3)
        q = rng.choice(
            [f"{{{lo}}}", f"{{{lo},}}", f"{{{lo},{lo + rng.randint(0, 3)}}}"]
        )
    else:
        n = rng.choice([999, 1000, 1001, 1002])
        q = f"{{{rng.choice(['', '1,'])}{n}}}"
    if rng.random() < 0.04:
        q += rng.choice(["*", "+"])  # stacked: pinned reject
    return a + q


def _pattern(rng: random.Random, depth: int = 0) -> str:
    branches = []
    for _ in range(rng.randint(1, 2 if depth else 3)):
        parts = [_quantified(rng, depth) for _ in range(rng.randint(1, 4))]
        if rng.random() < 0.25:
            parts.insert(rng.randrange(len(parts) + 1), rng.choice(_FLAGS))
        branches.append("".join(parts))
    return "|".join(branches)


def _rewrite(rng: random.Random) -> str:
    parts = [
        rng.choice(["L", "-", "$", "$1", r"\0", r"\1", r"\2", r"\\", r"\x", "\\"])
        for _ in range(rng.randint(0, 3))
    ]
    return "".join(parts)


def _options(rng: random.Random) -> str:
    if rng.random() < 0.5:
        return ""
    n = rng.randint(1, 3)
    alphabet = "cilmnpsg" + ("q " if rng.random() < 0.1 else "")
    return "".join(rng.choice(alphabet) for _ in range(n))


def _case_sql(rng: random.Random) -> str:
    # Grammar alphabets exclude single quotes, so patterns embed directly.
    p = _pattern(rng)
    assert "'" not in p
    form = rng.random()
    if form < 0.3:
        args = f"s, '{p}'"
        if rng.random() < 0.4:
            args += f", '{_options(rng)}'"
        expr = f"regexp_matches({args})"
    elif form < 0.45:
        expr = f"regexp_full_match(s, '{p}')"
    elif form < 0.7:
        args = f"s, '{p}', {rng.randint(0, 3)}"
        if rng.random() < 0.3:
            args += f", '{_options(rng)}'"
        expr = f"regexp_extract({args})"
    else:
        args = f"s, '{p}', '{_rewrite(rng)}'"
        if rng.random() < 0.5:
            args += f", '{_options(rng)}'"
        expr = f"regexp_replace({args})"
    return f"SELECT a, {expr} AS r FROM __THIS__"


def test_regexp_differential_fuzz(oracle):
    rng = random.Random(SEED)  # noqa: S311 - deterministic fuzzing, not crypto
    schema = _row_schema(SCHEMA)
    inputs = ROWS
    oracle.load("__THIS__", static(SCHEMA, ROWS))

    stats: Counter[str] = Counter()
    for i in range(N):
        sql = _case_sql(rng)
        ctx = f"seed={SEED} case={i}: {sql!r}"

        # RE2's \C can serve raw bytes inside multibyte chars — duckdb
        # produces invalid UTF-8 the oracle side can't even decode. The
        # engine (rust String) never can; treat like a duckdb error.
        result = oracle.try_answer(sql)
        want = None if isinstance(result, Trap) else result.to_pylist()

        try:
            fn = DuckDBInferFn(sql, row_tables={"__THIS__": schema}, static_tables={})
        except ValueError:
            stats["duck_err both_reject" if want is None else "reject"] += 1
            continue
        got = fn.infer_rows(inputs)

        assert want is not None, f"duckdb errored but the engine served rows: {ctx}"
        compare.assert_rows(got, want, ctx=ctx)
        stats["match"] += 1

    # Self-check: a degenerate generator (everything rejected) proves nothing.
    assert stats["match"] >= N * 0.2, f"fuzzer degenerated: {dict(stats)}"
    print(f"\nregexp fuzz seed={SEED} n={N}: {dict(stats)}")
