"""Catalog coverage over sklearn's transformers: the loop's scoreboard.

    uv run python -m sql_transform.native.coverage --write

regenerates the table in docs/native/coverage.md; `coverage_test.py` fails
while the file is stale (`--modules` prints the counts per sklearn module,
for the milestone reports). Every transformer sklearn lists is in exactly
one row: native (in the catalog, with its bound), composition (served by
composing entries, not an entry of its own), out of scope (with the reason,
from docs/native/goal.md "Scope"), or not yet.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.utils import all_estimators

from sql_transform.native._registry import catalog

DOC = Path(__file__).parents[2] / "docs" / "native" / "coverage.md"
BEGIN, END = "<!-- coverage:begin -->", "<!-- coverage:end -->"

# Served by composing the entries of their parts.
COMPOSITIONS = {
    "ColumnTransformer": "routes columns to parts",
    "FeatureUnion": "concatenates parts' outputs",
    "StackingClassifier": "a final estimator over parts' predictions",
    "StackingRegressor": "a final estimator over parts' predictions",
    "VotingClassifier": "averages parts' predictions",
    "VotingRegressor": "averages parts' predictions",
}

# Outside the catalog by goal.md "Scope", each with its reason.
OUT_OF_SCOPE = {
    "DictVectorizer": "input is dicts, not a row of columns",
    "FeatureHasher": "input is dicts or token lists, not a row of columns",
    "HashingVectorizer": "input is text documents",
    "TfidfTransformer": "input is a sparse count matrix",
    "PatchExtractor": "input is images",
    "LabelEncoder": "encodes a target, not features",
    "LabelBinarizer": "encodes a target, not features",
    "MultiLabelBinarizer": "encodes a target, not features",
    "KernelCenterer": "input is a kernel matrix",
    "TSNE": "no transform: the embedding is of the fit data only",
}


def rows() -> list[tuple[str, str, str]]:
    native = {c.__name__: e for c, e in catalog().items()}
    out = []
    for name, _ in sorted(all_estimators(type_filter="transformer")):
        if name in native:
            b = native[name].ulps
            out.append((name, "native", "bit-exact" if b == 0 else f"within {b} ulps"))
        elif name in COMPOSITIONS:
            out.append((name, "composition", COMPOSITIONS[name]))
        elif name in OUT_OF_SCOPE:
            out.append((name, "out of scope", OUT_OF_SCOPE[name]))
        else:
            out.append((name, "not yet", ""))
    return out


def table() -> str:
    rs = rows()
    counts = {k: sum(1 for r in rs if r[1] == k) for k in ("native", "not yet")}
    in_scope = sum(1 for r in rs if r[1] in ("native", "not yet"))
    lines = [
        BEGIN,
        f"**{counts['native']} of {in_scope}** in-scope transformers are native"
        f" ({len(rs)} listed by sklearn).",
        "",
        "| transformer | status | note |",
        "|---|---|---|",
        *(f"| `{n}` | {s} | {note} |" for n, s, note in rs),
        END,
    ]
    return "\n".join(lines)


STATUSES = ("native", "not yet", "composition", "out of scope")


def modules() -> str:
    """The same rows counted per sklearn module, for the milestone reports
    (docs/native/reports.md, "Scoreboard")."""
    status = {n: s for n, s, _ in rows()}
    counts: dict[str, dict[str, int]] = {}
    for name, cls in all_estimators(type_filter="transformer"):
        module = "sklearn." + cls.__module__.split(".")[1]
        counts.setdefault(module, dict.fromkeys(STATUSES, 0))[status[name]] += 1
    total = {s: sum(c[s] for c in counts.values()) for s in STATUSES}
    lines = [
        "| module | " + " | ".join(STATUSES) + " |",
        "|---|" + "---|" * len(STATUSES),
        *(
            f"| `{m}` | " + " | ".join(str(c[s]) for s in STATUSES) + " |"
            for m, c in sorted(counts.items())
        ),
        "| **total** | " + " | ".join(f"**{total[s]}**" for s in STATUSES) + " |",
    ]
    return "\n".join(lines)


def block(text: str) -> str:
    return text[text.index(BEGIN) : text.index(END) + len(END)]


def write() -> None:
    text = DOC.read_text()
    DOC.write_text(text.replace(block(text), table()))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--write", action="store_true", help="regenerate docs/native/coverage.md"
    )
    p.add_argument(
        "--modules", action="store_true", help="print the counts per sklearn module"
    )
    a = p.parse_args()
    if a.write:
        write()
    elif a.modules:
        print(modules())
    else:
        print(table())
