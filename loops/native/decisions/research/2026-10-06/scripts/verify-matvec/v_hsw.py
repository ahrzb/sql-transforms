import numpy as np, pickle, sys
D = sys.argv[1]; fx = {f["seed"]: f for f in pickle.load(open(sys.argv[2], "rb"))}
sk = np.load(f"{D}/tw_SkylakeX.npz"); hw = np.load(f"{D}/tw_Haswell.npz"); key = sk["key"]
for lab in ("row", "batch"):
    d = np.where(sk[lab] != hw[lab])[0]
    cats = {}
    for i in d:
        sd, iid, ri, k = key[i]; est = fx[int(sd)]["instances"][int(iid)]
        c = ("k=1" if est.components_.shape[0] == 1 else "") + ("F" if not est.components_.flags.c_contiguous else "C")
        cats[c] = cats.get(c, 0) + 1
    print(lab, "lanes differ", len(d), cats, "(seeds<200:", int((key[d, 0] < 200).sum()), ")")
