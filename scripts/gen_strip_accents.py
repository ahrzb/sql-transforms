"""Generate packages/confit/src/specializer/exec/strip_accents.rs by measuring DuckDB.

Why this file exists
--------------------
DuckDB's strip_accents() is utf8proc NFD-decompose + drop-marks + recompose,
with tables from whatever utf8proc vintage this duckdb build ships — which
lags Unicode 16 by dozens of codepoints. Any host Unicode library (Rust
crates, Python unicodedata, ICU) is therefore the wrong spec. Measurement
(pins-wave3/strip_accents.json) shows the observable behavior is exactly:

  1. all-ASCII input -> returned verbatim (embedded NULs preserved);
  2. otherwise truncate at the first NUL, then
  3. map each codepoint through a fixed table (4460 changed entries: 2450
     marks delete to '', the rest map 1:1 — never multi-codepoint), then
  4. compose adjacent Hangul jamo (L+V -> LV, LV+T -> LVT, formulaic).

This script sweeps every non-surrogate codepoint through the vectorized
path, extracts the changed entries, verifies the 4-step model end-to-end
against duckdb on random compositions, and freezes the table. The oracle
census in packages/confit/tests/test_duckdb_interpreter.py is the ongoing authority.

Regenerate with `uv run python scripts/gen_strip_accents.py` after bumping
the duckdb dependency, then run the gate.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import duckdb
import pyarrow as pa

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages" / "confit" / "src" / "specializer" / "exec" / "strip_accents.rs"

S_BASE, L_BASE, V_BASE, T_BASE = 0xAC00, 0x1100, 0x1161, 0x11A7
V_COUNT, T_COUNT, S_COUNT = 21, 28, 11172


def hangul_compose(cps: list[int]) -> list[int]:
    out: list[int] = []
    for c in cps:
        if out:
            p = out[-1]
            if 0x1100 <= p <= 0x1112 and 0x1161 <= c <= 0x1175:
                out[-1] = (
                    S_BASE + (p - L_BASE) * (V_COUNT * T_COUNT) + (c - V_BASE) * T_COUNT
                )
                continue
            if (
                S_BASE <= p < S_BASE + S_COUNT
                and (p - S_BASE) % T_COUNT == 0
                and 0x11A8 <= c <= 0x11C2
            ):
                out[-1] = p + (c - T_BASE)
                continue
        out.append(c)
    return out


def model(s: str, table: dict[int, str]) -> str:
    if all(ord(c) < 0x80 for c in s):
        return s
    nul = s.find("\x00")
    if nul >= 0:
        s = s[:nul]
    mapped = [table.get(ord(c), c) for c in s]
    cps = [ord(c) for m in mapped for c in m]
    return "".join(chr(c) for c in hangul_compose(cps))


def main() -> int:
    con = duckdb.connect()
    cps = [cp for cp in range(1, 0x110000) if not (0xD800 <= cp <= 0xDFFF)]
    con.register("cps_arrow", pa.table({"cp": cps, "s": [chr(c) for c in cps]}))
    con.execute("CREATE TABLE cps AS SELECT * FROM cps_arrow")
    rows = con.execute("SELECT cp, s, strip_accents(s) FROM cps ORDER BY cp").fetchall()

    table: dict[int, str] = {}
    for cp, s, out in rows:
        if out != s:
            assert len(out) <= 1, f"multi-codepoint output at U+{cp:04X}: {out!r}"
            table[cp] = out

    # Witnesses from the pins: fail loudly if the oracle moved.
    assert table[0x00E9] == "e" and table[0x212A] == "K" and table[0x0301] == ""

    # End-to-end model check: random multi-char strings incl. jamo,
    # marks, NULs, astral chars — the model must match duckdb exactly.
    rng = random.Random(49)  # noqa: S311 -- deterministic probe seed, not crypto
    pool = (
        [rng.choice(cps) for _ in range(400)]
        + list(range(0x1100, 0x1113))
        + list(range(0x1161, 0x1176))
        + list(range(0x11A8, 0x11C3))
        + [0x0301, 0x0300, 0xE9, 0x41, 0x62, 0x1F600, 0xAC00, 0x212A, 0]
    )
    probes = [
        "".join(chr(rng.choice(pool)) for _ in range(rng.randrange(1, 12)))
        for _ in range(600)
    ]
    con.register("probes_arrow", pa.table({"s": probes}))
    got = con.execute("SELECT strip_accents(s) FROM probes_arrow").fetchall()
    bad = [
        (p, g[0], model(p, table))
        for p, g in zip(probes, got, strict=True)
        if g[0] != model(p, table)
    ]
    assert not bad, (
        f"model diverges from duckdb on {len(bad)} probes, first: {bad[0]!r}"
    )

    deleted = sorted(cp for cp, m in table.items() if m == "")
    mapped = sorted((cp, m) for cp, m in table.items() if m != "")

    def fmt_deleted() -> str:
        body = "\n".join(f"    '\\u{{{cp:04X}}}'," for cp in deleted)
        return f"pub(super) const STRIP_DELETED: &[char] = &[\n{body}\n];\n"

    def fmt_mapped() -> str:
        body = "\n".join(
            f"    ('\\u{{{cp:04X}}}', '\\u{{{ord(m):04X}}}')," for cp, m in mapped
        )
        return f"pub(super) const STRIP_MAPPED: &[(char, char)] = &[\n{body}\n];\n"

    OUT.write_text(
        f"""//! DuckDB's strip_accents() per-codepoint map, as measured — see
//! scripts/gen_strip_accents.py for the model (ASCII fast path, NUL
//! truncation, this map, then Hangul jamo composition) and why a host
//! Unicode library would be the wrong spec (utf8proc version skew).
//!
//! GENERATED by `uv run python scripts/gen_strip_accents.py` against
//! duckdb-python {duckdb.__version__}; regenerate after a duckdb bump. The oracle
//! census in packages/confit/tests/test_duckdb_interpreter.py is the authority.

/// Per-codepoint result: `None` = unchanged, `Some(None)` = deleted,
/// `Some(Some(m))` = mapped 1:1.
pub(super) fn strip_map(c: char) -> Option<Option<char>> {{
    if (c as u32) < 0x80 {{
        return None;
    }}
    if STRIP_DELETED.binary_search(&c).is_ok() {{
        return Some(None);
    }}
    if let Ok(i) = STRIP_MAPPED.binary_search_by_key(&c, |e| e.0) {{
        return Some(Some(STRIP_MAPPED[i].1));
    }}
    None
}}

{fmt_deleted()}
{fmt_mapped()}""",
        encoding="utf-8",
    )
    print(f"wrote {OUT}: {len(deleted)} deleted + {len(mapped)} mapped entries")
    return 0


if __name__ == "__main__":
    sys.exit(main())
