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
import ast
import hashlib
import inspect
import json
import re
import textwrap
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.utils import all_estimators

from sql_transform.native._registry import Entry, catalog

DOC = Path(__file__).parents[2] / "spec" / "native" / "coverage.md"
BEGIN, END = "<!-- coverage:begin -->", "<!-- coverage:end -->"
BOUNDS_DOC = DOC.with_name("bounds.md")
# Fixed witnesses, not fitted models: K depends on shape and configuration,
# never on random fitted coefficients. Include both n-dominant and m-dominant.
BOUND_SHAPES = ((1, 1), (8, 3), (32, 64), (128, 16))

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

# Where a served composition holds an entry with a parity bound above 0
# (compose.py).
_CARRIED = "where each later step carries its bound"
BOUNDED_IN = {
    "ColumnTransformer": f"in a part, weighted or not, {_CARRIED}",
    "FeatureUnion": f"in a part, weighted or not, {_CARRIED}",
    "Pipeline": _CARRIED,
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
        on = f" {entry.scale.on}" if entry.scale.on else ""
        within = f"within K·eps·S + τ{on} (S its error scale)"
        if entry.varies:
            return f"bit-exact; {within} for some configurations, {on_request}"
        return f"{within}, {on_request}"
    if b == 0:
        return "bit-exact"
    if entry.varies:
        return f"bit-exact; within {b} ulps for some configurations, {on_request}"
    return f"within {b} ulps, {on_request}"


def nonzero_ulp_bounds() -> int:
    """The KPI of that name (mandates/native/README.md#reporting-contract): the catalog
    classes some configuration of which serves within a bound above 0,
    with `allow_bound=True`: an ulp bound above 0, or an error scale."""
    return sum(1 for e in catalog().values() if e.ulps or e.scale is not None)


def rows() -> list[tuple[str, str, str]]:
    native = {c.__name__: e for c, e in catalog().items()}
    out = []
    for name in sorted(_classes()):
        # A composition stays one, served or not (claim: catalog-scope); served,
        # it composes entries (compose.py).
        if name in COMPOSITIONS:
            note = COMPOSITIONS[name]
            if name in native:
                note += (
                    "; served over entries, one with a parity bound above 0"
                    f" {BOUNDED_IN[name]}, with `allow_bound=True`"
                )
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
    (mandates/native/README.md#reporting-contract, "Scoreboard")."""
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


def _source_hash(func) -> str:
    """Hash numerical source and its Python helper dependencies, not line numbers.

    AST removes formatting/comments. Following referenced Python functions also
    catches a change to a helper of S (the Nystroem/T54 failure mode).
    """
    sources = {}

    def visit(f):
        name = f"{f.__module__}.{f.__qualname__}"
        if name in sources:
            return
        tree = ast.parse(textwrap.dedent(inspect.getsource(f)))
        sources[name] = ast.dump(tree, include_attributes=False)
        for symbol in f.__code__.co_names:
            dependency = f.__globals__.get(symbol)
            if inspect.isfunction(dependency):
                visit(dependency)

    visit(func)
    return hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()


def bounds_table() -> str:
    """The approved ceilings and named error-scale implementations."""
    from sql_transform.native import function

    lines = [
        "# Native parity bounds",
        "",
        "Generated by `python -m sql_transform.native.coverage --write`; do not edit.",
        "K witnesses use (input features n, output components m), not fitted data.",
        "PowerTransformer witnesses also cover both methods and standardization.",
        "Hashes cover the AST source of K, S, τ and the comparison map, including",
        "referenced Python helpers. They are change alarms, not proofs of a bound.",
        "Decision references are those cited anywhere in the translator's module.",
        "",
        "| class | parity | ulps | per-function ulps | K witnesses "
        "| source hashes | decisions |",
        "|---|---|---|---|---|---|---|",
    ]
    for cls, entry in sorted(catalog().items(), key=lambda pair: pair[0].__name__):
        functions = (
            {f.__name__: function._BOUNDS.get(f, 0) for f in function._SERVED}
            if cls.__name__ == "FunctionTransformer"
            else {}
        )
        ks, hashes = {}, {}
        if entry.scale is not None:
            for n, m in BOUND_SHAPES:
                for method in (
                    ("box-cox", "yeo-johnson")
                    if cls.__name__ == "PowerTransformer"
                    else ("",)
                ):
                    for standardize in (False, True) if method else (False,):
                        est = SimpleNamespace(
                            n_features_in_=n,
                            components_=np.empty((m, n)),
                            normalization_=np.empty((m, m)),
                            method=method,
                            standardize=standardize,
                        )
                        key = f"n={n},m={m}"
                        if method:
                            key += f",{method},standardize={standardize}"
                        ks[key] = np.asarray(entry.scale.k(est)).tolist()
            hashes = {
                name: _source_hash(f)
                for name in ("k", "s", "tau", "g")
                if (f := getattr(entry.scale, name)) is not None
            }
        module = inspect.getmodule(entry.translate)
        decisions = sorted(
            set(
                re.findall(
                    r"(?:proposals|(?:loops/native/)?decisions)/closed/[\w-]+\.md",
                    inspect.getsource(module),
                )
            )
        )
        parity = (
            "parts"
            if entry.composes
            else (
                "error scale"
                if entry.scale
                else "ulp ceiling"
                if entry.ulps
                else "bit-exact"
            )
        )
        values = (cls.__name__, parity, entry.ulps, functions, ks, hashes, decisions)
        lines.append(
            "| "
            + " | ".join(
                json.dumps(v, sort_keys=True) if not isinstance(v, str) else v
                for v in values
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def write() -> None:
    text = DOC.read_text(encoding="utf-8")
    DOC.write_text(text.replace(block(text), table()), encoding="utf-8")
    BOUNDS_DOC.write_text(bounds_table(), encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--write",
        action="store_true",
        help="regenerate spec/native/coverage.md and spec/native/bounds.md",
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
