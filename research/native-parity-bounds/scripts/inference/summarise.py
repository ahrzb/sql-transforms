"""Markdown tables from results/*.json."""

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
sys.path.insert(0, str(HERE))
from common import PLAN  # noqa: E402

ROWK = ["default_rerun/row", "threads1/row", "haswell/row", "sandybridge/row",
        "npy_noavx512/row", "avx2_box/row", "sse_box/row"]
BATCH = ["default/batch", "threads1/batch", "haswell/batch", "sandybridge/batch",
         "npy_noavx512/batch", "avx2_box/batch", "sse_box/batch"]


def g(x):
    if isinstance(x, (int,)) or (isinstance(x, float) and x.is_integer() and abs(x) < 1e6):
        return f"{int(x)}"
    return f"{x:.2g}"


def load():
    out = {}
    for fam in PLAN:
        for dn in PLAN[fam][0]:
            p = RES / f"{fam}__{dn}.json"
            if p.exists():
                out[(fam, dn)] = json.loads(p.read_text())
    return out


def agg(r, names, key, part="feat"):
    vals = [(r["comps"][n][part][key], n) for n in names if n in r["comps"]]
    if not vals:
        return None, None
    return max(vals, key=lambda v: v[0])


def feature_table(R):
    print("| family | dataset | lanes | cmp | entries differ | max ulps | max abs | K max (eps·S) | K p99 | f32 differs |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for (fam, dn), r in R.items():
        groups = [("N", ["N"]), ("T' row", ROWK), ("T' batch", BATCH)]
        if fam.startswith("kmeans"):
            groups.insert(1, ("N direct", ["N_direct"]))
        for gname, names in groups:
            cells = []
            for key in ["entries_diff", "max_ulps", "max_abs", "K_max", "K_p99", "f32_diff"]:
                v, _ = agg(r, names, key)
                cells.append("-" if v is None else (f"{v:.3f}" if key in ("entries_diff",) else (f"{v:.1e}" if key == "f32_diff" else g(v))))
            print(f"| {fam} | {dn} | {r['lanes']} | {gname} | " + " | ".join(cells) + " |")


DOWN = [("logreg_flips", "LR flips"), ("logreg_dproba_abs", "LR dP abs"), ("logreg_dproba_ulps", "LR dP ulps"),
        ("linreg_dabs", "Lin d abs"), ("linreg_dulps", "Lin d ulps"),
        ("dtree_leafflips", "DT leaf"), ("dtree_flips", "DT lab"), ("dtreg_leafflips", "DTreg leaf"),
        ("rf_rows_leafflip", "RF leaf rows"), ("rf_flips", "RF lab"),
        ("hgb_rows_rawdiff", "HGB raw rows"), ("hgb_flips", "HGB lab"), ("hgb_dproba_abs", "HGB dP abs"),
        ("argmin_flips", "argmin")]


def down_table(R):
    print("| family | dataset | rows | cmp | " + " | ".join(h for _, h in DOWN) + " |")
    print("|---|---|---|---|" + "---|" * len(DOWN))
    for (fam, dn), r in R.items():
        groups = [("N", ["N"]), ("T' row", ROWK), ("T' batch", BATCH), ("fit_transform", ["fit_transform(train rows)"])]
        if fam.startswith("kmeans"):
            groups.insert(1, ("N direct", ["N_direct"]))
        for gname, names in groups:
            cells = []
            for key, _ in DOWN:
                if key == "argmin_flips" and not fam.startswith("kmeans"):
                    cells.append("")
                    continue
                v, _ = agg(r, names, key, "down")
                cells.append("-" if v is None else g(v))
            rows = r["n_train"] if gname == "fit_transform" else r["n_rows"]
            print(f"| {fam} | {dn} | {rows} | {gname} | " + " | ".join(cells) + " |")


def expected_table(R):
    print("| family | dataset | cmp | DT exp | DT obs | RF exp | RF obs(leaf-rows) | HGB exp | HGB obs | LR exp | LR obs | argmin exp | argmin obs |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for (fam, dn), r in R.items():
        for cname in ["N", "haswell/row", "sandybridge/row", "default/batch", "avx2_box/row"]:
            if cname not in r["comps"]:
                continue
            c = r["comps"][cname]
            e, d = c["expected"], c["down"]
            am = (g(e.get("argmin", 0)), g(d.get("argmin_flips", 0))) if fam.startswith("kmeans") else ("", "")
            print(f"| {fam} | {dn} | {cname} | {g(e['dtree'])} | {d['dtree_leafflips']} | {g(e['rf'])} | {d['rf_rows_leafflip']} | "
                  f"{g(e['hgb'])} | {d['hgb_rows_rawdiff']} | {g(e['logreg'])} | {d['logreg_flips']} | {am[0]} | {am[1]} |")


def per_env(R):
    """every comparator, compact: flips summed over models."""
    print("| family | dataset | comparator | rows≠ | max abs | max ulps | LR | DT leaf | RF leaf rows | HGB raw rows | argmin |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for (fam, dn), r in R.items():
        for cname, c in r["comps"].items():
            f, d = c["feat"], c["down"]
            print(f"| {fam} | {dn} | {cname} | {f['rows_any_diff']} | {g(f['max_abs'])} | {g(f['max_ulps'])} | {d['logreg_flips']} | "
                  f"{d['dtree_leafflips']} | {d['rf_rows_leafflip']} | {d['hgb_rows_rawdiff']} | {d.get('argmin_flips', '')} |")


if __name__ == "__main__":
    R = load()
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("feat", "all"):
        feature_table(R)
        print()
    if which in ("down", "all"):
        down_table(R)
        print()
    if which in ("exp", "all"):
        expected_table(R)
        print()
    if which in ("env",):
        per_env(R)


def compact(R):
    """one row per case: N | max over every T' (row and batch, all envs)."""
    TP = ROWK + BATCH
    hdr = ["case", "rows", "lanes", "entries≠ N/T'", "max ulps N/T'", "max abs N/T'", "K max N/T'",
           "f32≠ N/T'", "LR lab N/T'", "LR dP N/T'", "Lin d N/T'", "DT leaf N/T'", "DT lab N/T'",
           "RF leaf-rows N/T'", "RF lab N/T'", "HGB branch-rows N/T'/fit", "HGB lab N/T'/fit", "argmin N/T'"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for (fam, dn), r in R.items():
        def fe(key, fmt=g):
            a = r["comps"]["N"]["feat"][key]
            b, _ = agg(r, TP, key)
            return f"{fmt(a)} / {fmt(b)}"

        def dn_(key):
            a = r["comps"]["N"]["down"][key]
            b, _ = agg(r, TP, key, "down")
            return f"{g(a)} / {g(b)}"
        ft = r["comps"]["fit_transform(train rows)"]["down"]
        cells = [f"{fam}/{dn}", str(r["n_rows"]), str(r["lanes"]),
                 fe("entries_diff", lambda v: f"{v:.3f}"), fe("max_ulps"), fe("max_abs"), fe("K_max"),
                 fe("f32_diff", lambda v: f"{v:.0e}" if v else "0"),
                 dn_("logreg_flips"), dn_("logreg_dproba_abs"), dn_("linreg_dabs"),
                 dn_("dtree_leafflips"), dn_("dtree_flips"), dn_("rf_rows_leafflip"), dn_("rf_flips"),
                 dn_("hgb_rows_rawdiff") + f" / {ft['hgb_rows_rawdiff']}",
                 dn_("hgb_flips") + f" / {ft['hgb_flips']}",
                 dn_("argmin_flips") if fam.startswith("kmeans") else ""]
        print("| " + " | ".join(cells) + " |")


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "compact":
    compact(load())
