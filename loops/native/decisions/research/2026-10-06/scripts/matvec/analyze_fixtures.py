"""Per-lane comparison on the catalog's PCA(whiten=True) fixtures.

Lanes in twin_run.py's order. For each lane: x (row), c (component), mc
(projected mean, numpy in this process, as the translation computes it), s
(clipped whitening scale). Quantities:
  native  = fl(fl(LTR(x.c) - mc) / s)          the entry (left to right, unfused)
  exact_xc = correctly rounded x.c             (fsum of TwoProduct pieces)
  S   = (sum|x_i c_i| + |mc|) / s              the record's term scale
  K(a,b) = |a - b| / (eps * S)
Usage: analyze_fixtures.py FIXTURES.pkl REF.npz OTHER.npz... (label=path)
"""

import math
import pickle
import struct
import sys
import warnings

import numpy as np

sys.path.insert(0, __import__("os").path.dirname(__file__))
from emul import numpy_row_matvec  # noqa: E402

EPS = 2.0**-52


def ordered(x):
    (i,) = struct.unpack("<q", struct.pack("<d", x))
    return i if i >= 0 else -(i & 0x7FFF_FFFF_FFFF_FFFF)


def ulps(a, b):
    if math.isnan(a) or math.isnan(b):
        return 0 if (math.isnan(a) and math.isnan(b)) else 2**64
    return abs(ordered(a) - ordered(b))


def exact_dot(x, c):
    parts = []
    for a, b in zip(x, c):
        p = a * b
        parts.append(p)
        if math.isfinite(p):
            parts.append(math.fma(a, b, -p))
    try:
        return math.fsum(parts)
    except (OverflowError, ValueError):
        return float("nan")


fx_path = sys.argv[1]
labels, paths = zip(*(a.split("=", 1) for a in sys.argv[2:]))
with open(fx_path, "rb") as f:
    fixtures = pickle.load(f)

warnings.simplefilter("ignore")
lanes = []  # (seed, x, c, mc, s, native, emu_twin, exact_xc, S, S0)
for fx in fixtures:
    for iid, est in fx["instances"].items():
        C = est.components_
        mc = (np.reshape(est.mean_, (1, -1)) @ C.T)[0]
        scale = np.sqrt(est.explained_variance_)
        scale[scale < np.finfo(scale.dtype).eps] = np.finfo(scale.dtype).eps
        Cl = C.tolist()
        for r in fx["rows"]:
            if r["__iid"] != iid:
                continue
            vals = [float("nan") if r[n] is None else float(r[n]) for n in fx["names"]]
            try:
                est.transform([vals])
            except ValueError:
                continue
            emu = numpy_row_matvec(vals, Cl, "SkylakeX")
            for k in range(len(Cl)):
                c = Cl[k]
                acc = vals[0] * c[0]
                for i in range(1, len(vals)):
                    acc = acc + vals[i] * c[i]
                s = float(scale[k])
                m = float(mc[k])
                native = (acc - m) / s
                emu_twin = (emu[k] - m) / s
                S0 = sum(abs(a * b) for a, b in zip(vals, c))
                S = (S0 + abs(m)) / s
                Sm = sum(abs(a * b) for a, b in zip(est.mean_.tolist(), c))
                S2 = (S0 + Sm) / s
                lanes.append((fx["seed"], native, emu_twin, exact_dot(vals, c), acc, emu[k], S, S0, S2, len(vals), len(Cl), bool(C.flags.c_contiguous)))

seed = np.array([l[0] for l in lanes])
native = np.array([l[1] for l in lanes])
emu_twin = np.array([l[2] for l in lanes])
S = np.array([l[6] for l in lanes])
S_rec = S.copy()
S2 = np.array([l[8] for l in lanes])
import os
if os.environ.get("SCALE") == "S2":
    S = S2
    print("term scale: S2 = (sum|x_i c_i| + sum|m_i c_i|) / s")
else:
    print("term scale: S = (sum|x_i c_i| + |m.c|) / s (the record's)")
twins = {lab: np.load(p)["row"] for lab, p in zip(labels, paths)}
batches = {lab + "/batch": np.load(p)["batch"] for lab, p in zip(labels, paths)}
ref_label = labels[0]
ref = twins[ref_label]
assert len(ref) == len(native), (len(ref), len(native))
print(f"lanes {len(native)}; finite S {np.isfinite(S).sum()}")
print(f"emulated SkylakeX row twin == {ref_label} row twin: "
      f"{int(np.sum((emu_twin == ref) | (np.isnan(emu_twin) & np.isnan(ref))))}/{len(ref)} bit-exact")


def stats(a, b, mask):
    u = np.array([ulps(float(p), float(q)) for p, q in zip(a[mask], b[mask])], dtype=float)
    d = np.abs(a[mask] - b[mask])
    same = (a[mask] == b[mask]) | (np.isnan(a[mask]) & np.isnan(b[mask]))
    d[same] = 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        K = d / (EPS * S[mask])
    K[same] = 0.0
    Kf = K[np.isfinite(K)]
    return dict(
        n=int(mask.sum()),
        differ=int((~same).sum()),
        max_ulps=int(u.max()) if len(u) else 0,
        p99_K=float(np.percentile(Kf, 99)) if len(Kf) else 0,
        max_K=float(Kf.max()) if len(Kf) else 0,
        over_1e3_ulps=int((u > 1e3).sum()),
    )


for upto in [int(v) for v in __import__("os").environ.get("UPTO", "40,200").split(",")]:
    mask = (seed < upto) & np.isfinite(S)
    print(f"\n== seeds 0..{upto - 1}, lanes with finite S: {mask.sum()} "
          f"(all lanes {(seed < upto).sum()})")
    rows = [("native (LTR) vs " + ref_label, native, ref)]
    for lab in labels[1:]:
        rows.append((f"{lab} row vs {ref_label} row", twins[lab], ref))
    for lab in labels:
        rows.append((f"{lab} batch(gemm) vs {ref_label} row", batches[lab + "/batch"], ref))
    for lab in labels[1:]:
        rows.append((f"native vs {lab} row", native, twins[lab]))
    print(f"{'pair':48s} {'lanes':>6s} {'differ':>6s} {'max ulps':>14s} {'>1e3ulps':>8s} {'p99 K':>7s} {'max K':>7s}")
    for name, a, b in rows:
        st = stats(a, b, mask)
        print(f"{name:48s} {st['n']:6d} {st['differ']:6d} {st['max_ulps']:14d} {st['over_1e3_ulps']:8d} {st['p99_K']:7.3f} {st['max_K']:7.3f}")

# errors of the x.c sums against the exact value, in eps * S0 units
exact = np.array([l[3] for l in lanes])
ltr = np.array([l[4] for l in lanes])
blas = np.array([l[5] for l in lanes])
S0 = np.array([l[7] for l in lanes])
ok = np.isfinite(exact) & np.isfinite(S0) & (EPS * S0 > 1e-290) & np.isfinite(ltr) & np.isfinite(blas)
with np.errstate(invalid="ignore"):
    e_ltr = np.abs(ltr[ok] - exact[ok]) / (EPS * S0[ok])
    e_blas = np.abs(blas[ok] - exact[ok]) / (EPS * S0[ok])
print(f"\nx.c vs exact, eps*S0 units (S0 = sum|x_i c_i|), {ok.sum()} lanes:"
      f" LTR max {e_ltr.max():.3f} p99 {np.percentile(e_ltr, 99):.3f};"
      f" SkylakeX gemv max {e_blas.max():.3f} p99 {np.percentile(e_blas, 99):.3f}")

# the worst twin-vs-twin lanes
if len(labels) > 1:
    for lab in labels[1:]:
        t = twins[lab]
        d = np.abs(t - ref)
        with np.errstate(invalid="ignore", divide="ignore"):
            K = np.where(t == ref, 0.0, d / (EPS * S))
        K[~np.isfinite(K)] = 0
        i = int(np.argmax(K))
        l = lanes[i]
        print(f"worst {lab} vs {ref_label}: K={K[i]:.2f} seed={l[0]} m={l[9]} k={l[10]} C-contig={l[11]} "
              f"S_rec={S_rec[i]:.3e} S2={S2[i]:.3e} S2/S_rec={S2[i]/S_rec[i]:.1f}")

# lanes where the term-scale tolerance underflows or overflows
tiny = 2.0**-1022
for lab in labels:
    t = twins[lab]
    differ = ~((native == t) | (np.isnan(native) & np.isnan(t)))
    sub = differ & np.isfinite(S) & (EPS * S < tiny)
    nonfin = differ & ~np.isfinite(S)
    print(f"{lab}: native!=twin lanes with eps*S subnormal/zero: {int(sub.sum())}; with S non-finite: {int(nonfin.sum())}")
    for i in np.where(sub | nonfin)[0][:3]:
        print("   e.g. lane", i, "native", native[i], "twin", t[i], "S", S[i], "ulps", ulps(float(native[i]), float(t[i])))
