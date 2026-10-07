"""Catalog coverage over sklearn's transformers: the loop's scoreboard.

    uv run python -m sql_transform.native.coverage --write

regenerates the table in packages/sql-transform/spec/native/coverage.md;
`coverage_test.py` fails while the file is stale. For the milestone reports,
`--modules` prints the counts per sklearn module and `--bounds` the KPI
`nonzero_ulp_bounds`. Every transformer sklearn lists is in exactly one row:
native (in the catalog, with its bound), composition (served by composing
entries, not an entry of its own; "served" when the catalog composes it), out
of scope (with the reason, from claim: catalog-scope in
spec/native/catalog-contract.md), or not yet.
A composition the catalog serves that sklearn does not list as a
transformer (`Pipeline`, not a `TransformerMixin`) gets a row of its own,
outside the counts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from sklearn.utils import all_estimators

from sql_transform.native._registry import Entry, catalog

DOC = Path(__file__).parents[2] / "spec" / "native" / "coverage.md"
BEGIN, END = "<!-- coverage:begin -->", "<!-- coverage:end -->"

# Served by composing the entries of their parts.
COMPOSITIONS = {
    "ColumnTransformer": "routes columns to parts",
    "FeatureUnion": "concatenates parts' outputs",
    "Pipeline": "chains steps",
    "StackingClassifier": "a final estimator over parts' predictions",
    "StackingRegressor": "a final estimator over parts' predictions",
    "VotingClassifier": "averages parts' predictions",
    "VotingRegressor": "averages parts' predictions",
}

# Outside the catalog by claim: catalog-scope, each with its reason.
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


def _classes() -> dict[str, type]:
    """sklearn's transformers, and the compositions the catalog serves
    that sklearn does not list among them."""
    listed = dict(all_estimators(type_filter="transformer"))
    served = {
        c.__name__: c
        for c in catalog()
        if c.__name__ in COMPOSITIONS and c.__name__ not in listed
    }
    return {**listed, **served}


def _exactness(entry: Entry) -> str:
    """An entry's bound as the table reads it. A per-estimator bound runs
    from bit-exact (the configurations whose own bound is 0) to the
    class's ceiling. A bound above 0 serves only on request."""
    b = entry.ulps
    on_request = "served only with `allow_bound=True`"
    if entry.scale is not None:
        return f"within K·eps·S + τ (S its error scale), {on_request}"
    if b == 0:
        return "bit-exact"
    if entry.varies:
        return f"bit-exact; within {b} ulps for some configurations, {on_request}"
    return f"within {b} ulps, {on_request}"


def nonzero_ulp_bounds() -> int:
    """The KPI of that name (loops/native/report-format.md): the catalog
    classes some configuration of which serves within a bound above 0,
    with `allow_bound=True`: an ulp bound above 0, or an error scale."""
    return sum(1 for e in catalog().values() if e.ulps or e.scale is not None)


def rows() -> list[tuple[str, str, str]]:
    native = {c.__name__: e for c, e in catalog().items()}
    out = []
    for name in sorted(_classes()):
        # A composition stays one, served or not (claim: catalog-scope); served,
        # it composes bit-exact entries only (compose.py).
        if name in COMPOSITIONS:
            note = COMPOSITIONS[name]
            if name in native:
                note += "; served over bit-exact entries"
            out.append((name, "composition", note))
        elif name in native:
            out.append((name, "native", _exactness(native[name])))
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
        f" ({len(all_estimators(type_filter='transformer'))} listed by sklearn).",
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
    (loops/native/report-format.md, "Scoreboard")."""
    status = {n: s for n, s, _ in rows()}
    counts: dict[str, dict[str, int]] = {}
    for name, cls in _classes().items():
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
        "--write",
        action="store_true",
        help="regenerate packages/sql-transform/spec/native/coverage.md",
    )
    p.add_argument(
        "--modules", action="store_true", help="print the counts per sklearn module"
    )
    p.add_argument(
        "--bounds", action="store_true", help="print the KPI nonzero_ulp_bounds"
    )
    a = p.parse_args()
    if a.write:
        write()
    elif a.modules:
        print(modules())
    elif a.bounds:
        print(f"nonzero_ulp_bounds: {nonzero_ulp_bounds()}")
    else:
        print(table())
