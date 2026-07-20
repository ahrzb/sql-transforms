"""Generate the DataFusion function catalogue (Backlog.md doc-1) from installed DataFusion.

The catalogue is the DataFusion (batch-path) function surface — the parity target
the Rust InferFn interpreter matches against, and the menu the authoring SQL surface
draws from. Regenerate after a DataFusion upgrade:

    uv run python scripts/gen_datafusion_catalogue.py

Signatures/descriptions are DataFusion's own `information_schema.routines` metadata
verbatim — including a few upstream quirks where an alias documents under another
name's signature (e.g. `mean`→`avg`, `covar`→`covar_samp`).
"""

import re
from pathlib import Path

from datafusion import SessionConfig, SessionContext

# ponytail: filename is Backlog.md's tool-managed doc name (spaces + id); keep it in
# sync with the doc-1 entry if the tool ever renumbers. Frontmatter below keeps the
# regenerated file a valid Backlog.md doc.
OUT = (
    Path(__file__).resolve().parent.parent
    / "backlog"
    / "docs"
    / "doc-1 - DataFusion-function-catalogue.md"
)
FRONTMATTER = [
    "---",
    "id: doc-1",
    "title: DataFusion function catalogue",
    "type: other",
    "---",
]


def clean(s: str | None) -> str:
    return re.sub(r"\s+", " ", s).replace("|", r"\|").strip() if s else ""


def main() -> None:
    ctx = SessionContext(SessionConfig().with_information_schema(True))
    df = ctx.sql(
        "SELECT DISTINCT routine_name, function_type, syntax_example, description "
        "FROM information_schema.routines"
    )
    # Key on (name, type): a function can be both AGGREGATE and WINDOW (e.g.
    # first_value) and should appear in both sections.
    by_key: dict[tuple[str, str], tuple] = {}
    for batch in df.collect():
        for i in range(batch.num_rows):
            name, ftype = batch.column(0)[i].as_py(), batch.column(1)[i].as_py()
            syn, desc = batch.column(2)[i].as_py(), batch.column(3)[i].as_py()
            key = (name, ftype)
            # prefer a row that actually carries a signature
            if key not in by_key or (syn and not by_key[key][2]):
                by_key[key] = (name, ftype, syn, desc)

    ver = __import__("datafusion").__version__
    groups: dict[str, list] = {"AGGREGATE": [], "WINDOW": [], "SCALAR": []}
    for f in by_key.values():
        groups.setdefault(f[1], []).append(f)
    for g in groups.values():
        g.sort(key=lambda x: x[0])

    n_agg, n_win, n_scalar = (len(groups[k]) for k in ("AGGREGATE", "WINDOW", "SCALAR"))
    lines = [
        "# DataFusion function & aggregate catalogue",
        "",
        f"Auto-generated from the installed **DataFusion {ver}** (the engine behind the",
        "`transform` / `fit` path) via its `information_schema.routines`. This is the",
        "**parity target**: every function here runs in the DataFusion (batch) path, so the",
        "Rust `InferFn` interpreter must match any of these it claims to support — and it is",
        "also the menu the authoring SQL surface can draw from. See",
        "[SQL_SUPPORT.md](../../docs/SQL_SUPPORT.md) for what the *interpreter* implements today.",
        "",
        "Regenerate after a DataFusion upgrade: `uv run python",
        "scripts/gen_datafusion_catalogue.py`. Signatures/descriptions are DataFusion's own",
        "metadata verbatim, including a few upstream quirks where an alias documents under",
        "another name's signature (e.g. `mean`→`avg`).",
        "",
        f"**Totals ({ver}):** {n_agg} aggregate · {n_win} window · {n_scalar} scalar "
        "(a few names appear in two sections — both an aggregate and a window function).",
        "",
    ]
    for title, key in [
        ("Aggregate functions", "AGGREGATE"),
        ("Window functions", "WINDOW"),
        ("Scalar functions", "SCALAR"),
    ]:
        lines += [
            f"## {title} ({len(groups[key])})",
            "",
            "| Function | Signature | Description |",
            "|---|---|---|",
        ]
        lines += [
            f"| `{name}` | {clean(syn) or '—'} | {clean(desc) or '—'} |"
            for name, _ftype, syn, desc in groups[key]
        ]
        lines.append("")

    OUT.write_text("\n".join(FRONTMATTER + lines), encoding="utf-8")
    print(f"wrote {OUT} — {n_agg} aggregate, {n_win} window, {n_scalar} scalar")


if __name__ == "__main__":
    main()
