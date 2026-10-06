import numpy as np, pickle, sys
D = sys.argv[1]; fx = {f["seed"]: f for f in pickle.load(open(sys.argv[2], "rb"))}
sk = np.load(f"{D}/tw_SkylakeX.npz"); hw = np.load(f"{D}/tw_Haswell.npz"); key = sk["key"]
d = np.where((sk["batch"] != hw["batch"]) & (key[:, 0] < 200))[0]
cc = [fx[int(key[i,0])]["instances"][int(key[i,1])].components_ for i in d]
print("seeds<200 HSW vs SKX batch differ:", len(d), "C-contig:", sum(c.flags.c_contiguous for c in cc), "max n among C:", max([c.shape[1] for c in cc if c.flags.c_contiguous], default=0))
