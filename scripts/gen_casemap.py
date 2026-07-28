"""Generate packages/confit/src/specializer/exec/casemap.rs by measuring DuckDB.

Why this file exists
--------------------
DuckDB's upper()/lower() use utf8proc's SIMPLE case mapping: every codepoint
maps to exactly one codepoint (UnicodeData.txt's Simple_*case_Mapping).
Rust std only exposes the FULL mapping (SpecialCasing.txt), where a codepoint
may expand ('ss' -> "SS"). The engine's fallback rule — take Rust's full map
iff it is 1:1, else keep the codepoint — agrees with DuckDB almost everywhere,
and this script measures the exact exceptions and freezes them into a table.

The exceptions come from two sources, both captured by measurement:
  1. Simple-vs-full divergence: codepoints whose full map is multi-char but
     which have a DIFFERENT 1:1 simple map (the sharp s, Turkish dotted I,
     the Greek ypogegrammeni block U+1F80..U+1FFC).
  2. Unicode VERSION skew: case pairs added to Unicode after the utf8proc
     release inside this duckdb build (e.g. U+019B/U+A7DC from Unicode 16) —
     Rust maps them, DuckDB does not. DuckDB is the oracle, so its identity
     behavior wins.

Why no dependency
-----------------
A unicode-data crate would give us *some* Unicode version's simple maps —
which is the wrong spec twice over: it ignores class 2 entirely (the crate's
tables won't match utf8proc's vintage), and it drags a dependency in for
what measurement shows is ~83 codepoints. The measured table is smaller,
exactly right by construction, and verified end-to-end by the full-codepoint
census in packages/confit/tests/test_duckdb_interpreter.py (which will fail
loudly if this table ever drifts from the duckdb the suite runs against).

Regenerate with `uv run python scripts/gen_casemap.py` after bumping the
duckdb dependency, then run the gate: the census test is the authority.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import pyarrow as pa

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "packages" / "confit" / "src" / "specializer" / "exec" / "casemap.rs"


def rust_rule(c: str, upper: bool) -> str:
    """Phase-1 approximation of the engine's dependency-free fallback (full
    map iff 1:1, else keep). Python's tables APPROXIMATE Rust's: all three of
    Python, Rust, and duckdb's utf8proc can ship different Unicode versions,
    so phase 2 measures the actual compiled engine and patches the residue
    (e.g. Unicode-16 case pairs Rust maps but Python does not)."""
    m = c.upper() if upper else c.lower()
    return m if len(m) == 1 else c


def engine_residue(chunks: list[str], duck_u: list[str], duck_l: list[str]):
    """Phase 2: run the census through the INSTALLED engine and report every
    codepoint still diverging from duckdb — Rust-vs-Python table skew the
    phase-1 approximation cannot see. Requires the built sql_transform wheel;
    skipped (with a warning) when it is not importable."""
    try:
        from confit import DuckDBInferFn
        from pydantic import create_model
    except ImportError as e:  # pragma: no cover -- generator convenience
        print(
            f"WARNING: engine not importable ({e}); phase 2 skipped — "
            "rebuild and rerun, the census test is the authority"
        )
        return [], []
    model = create_model("Row", s=(str, ...))
    fn = DuckDBInferFn(
        "SELECT upper(s) AS u, lower(s) AS l FROM __THIS__",
        row_tables={"__THIS__": model},
        static_tables={},
    )
    ru, rl = [], []
    for s, du, dl in zip(chunks, duck_u, duck_l, strict=True):
        r = fn.infer({"__THIS__": [model(s=s)]})[0]
        ru.extend((ord(c), b) for c, a, b in zip(s, r.u, du, strict=True) if a != b)
        rl.extend((ord(c), b) for c, a, b in zip(s, r.l, dl, strict=True) if a != b)
    return ru, rl


def main() -> int:
    con = duckdb.connect()
    cps = [cp for cp in range(1, 0x110000) if not (0xD800 <= cp <= 0xDFFF)]
    con.register("cps_arrow", pa.table({"cp": cps, "s": [chr(c) for c in cps]}))
    con.execute("CREATE TABLE cps AS SELECT * FROM cps_arrow")
    rows = con.execute(
        "SELECT cp, s, upper(s), lower(s) FROM cps ORDER BY cp"
    ).fetchall()

    upper_ex, lower_ex = {}, {}
    for cp, s, u, lo_ in rows:
        assert len(u) == 1 and len(lo_) == 1, f"non-1:1 duckdb map at U+{cp:04X}"
        if rust_rule(s, True) != u:
            upper_ex[cp] = u
        if rust_rule(s, False) != lo_:
            lower_ex[cp] = lo_

    # Carry forward every codepoint already in casemap.rs, re-measured
    # against duckdb. Phase 2 measures the INSTALLED engine, which already
    # contains the current table — so skew entries it fixed report no
    # residue and would be silently dropped on regeneration without this.
    # An entry whose value equals what the fallback gives is harmless.
    if OUT.exists():
        import re

        by_cp = {cp: (u, lo_) for cp, _, u, lo_ in rows}
        text = OUT.read_text(encoding="utf-8")
        entry = r"\('\\u\{([0-9A-F]+)\}', '\\u\{[0-9A-F]+\}'\)"
        # Split at the const DEFINITION (the lookup fns mention the names too).
        upper_part, _, lower_part = text.partition("const LOWER_EXCEPTIONS")
        for part, ex, idx in ((upper_part, upper_ex, 0), (lower_part, lower_ex, 1)):
            for m in re.finditer(entry, part):
                cp = int(m.group(1), 16)
                if cp in by_cp:
                    ex.setdefault(cp, by_cp[cp][idx])

    # Phase 2: census the compiled engine in chunks, patch Rust-side skew.
    step = 0x8000
    chunks, duck_u, duck_l = [], [], []
    for lo in range(1, 0x110000, step):
        s = "".join(
            chr(c)
            for c in range(lo, min(lo + step, 0x110000))
            if not (0xD800 <= c <= 0xDFFF)
        )
        if s:
            chunks.append(s)
            u, lo_ = con.execute("SELECT upper(?), lower(?)", [s, s]).fetchone()
            duck_u.append(u)
            duck_l.append(lo_)
    ru, rl = engine_residue(chunks, duck_u, duck_l)
    n_skew = len([cp for cp, _ in ru if cp not in upper_ex]) + len(
        [cp for cp, _ in rl if cp not in lower_ex]
    )
    upper_ex.update(ru)
    lower_ex.update(rl)
    print(f"phase 2: {n_skew} Rust-vs-Python skew entries patched in")
    upper_ex = sorted(upper_ex.items())
    lower_ex = sorted(lower_ex.items())

    def table(name: str, entries: list[tuple[int, str]]) -> str:
        body = "\n".join(
            f"    ('\\u{{{cp:04X}}}', '\\u{{{ord(m):04X}}}'), // {chr(cp)!r} -> {m!r}"
            for cp, m in entries
        )
        return f"pub(super) const {name}: &[(char, char)] = &[\n{body}\n];\n"

    duck_version = duckdb.__version__
    OUT.write_text(
        f"""//! DuckDB's SIMPLE case mapping, as measured — see scripts/gen_casemap.py
//! for the full story (simple-vs-full mapping, Unicode version skew, and why
//! a dependency would be the wrong spec twice over).
//!
//! GENERATED by `uv run python scripts/gen_casemap.py` against duckdb-python
//! {duck_version}; regenerate after a duckdb bump. The full-codepoint census in
//! packages/confit/tests/test_duckdb_interpreter.py fails loudly if this table
//! drifts from the duckdb the suite runs against — the census, not this file, is the
//! authority.
//!
//! Entries are (codepoint, mapped) pairs, sorted by codepoint, covering only
//! the codepoints where the dependency-free fallback (Rust's full map iff it
//! is 1:1, else identity) disagrees with DuckDB. Everything else goes through
//! the fallback in `simple_upper`/`simple_lower`.

/// Simple uppercase for one codepoint, exactly as DuckDB computes it.
pub fn simple_upper(c: char) -> char {{
    if let Ok(i) = UPPER_EXCEPTIONS.binary_search_by_key(&c, |e| e.0) {{
        return UPPER_EXCEPTIONS[i].1;
    }}
    let mut it = c.to_uppercase();
    let first = it.next().expect("case iterators are non-empty");
    if it.next().is_none() {{
        first
    }} else {{
        c
    }}
}}

/// Simple lowercase for one codepoint, exactly as DuckDB computes it.
pub fn simple_lower(c: char) -> char {{
    if let Ok(i) = LOWER_EXCEPTIONS.binary_search_by_key(&c, |e| e.0) {{
        return LOWER_EXCEPTIONS[i].1;
    }}
    let mut it = c.to_lowercase();
    let first = it.next().expect("case iterators are non-empty");
    if it.next().is_none() {{
        first
    }} else {{
        c
    }}
}}

{table("UPPER_EXCEPTIONS", upper_ex)}
{table("LOWER_EXCEPTIONS", lower_ex)}""",
        encoding="utf-8",
    )
    print(f"wrote {OUT}: {len(upper_ex)} upper + {len(lower_ex)} lower exceptions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
