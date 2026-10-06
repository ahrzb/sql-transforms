import numpy as np, sys
D = sys.argv[1]; E = np.load(f"{D}/entry.npz"); T = {c: np.load(f"{D}/tw_{c}.npz") for c in ("SkylakeX", "Sandybridge", "Haswell")}
ent = E["ent"]; S2 = E["S2"]; EPS = 2.0**-52
pairs = {"entry-SKX": (ent, T["SkylakeX"]["row"]), "batch-row": (T["SkylakeX"]["batch"], T["SkylakeX"]["row"]), "SNB-SKX": (T["Sandybridge"]["row"], T["SkylakeX"]["row"]), "entry-SNB": (ent, T["Sandybridge"]["row"])}
for k, (a, b) in pairs.items():
    fa, fb = np.isfinite(a), np.isfinite(b)
    one = fa ^ fb
    bothinf_diff = (~fa & ~fb) & ~((a == b) | (np.isnan(a) & np.isnan(b)))
    print(k, "exactly one non-finite:", int(one.sum()), "both non-finite but different:", int(bothinf_diff.sum()), "lanes with S2 inf:", int((~np.isfinite(S2)).sum()),
          "differing lanes with S2 inf:", int((~np.isfinite(S2) & (a != b) & fa & fb).sum()))
    for i in np.where(one)[0][:3]: print("   ", a[i], b[i], "S2", S2[i])
