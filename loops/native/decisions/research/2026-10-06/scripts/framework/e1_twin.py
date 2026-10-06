"""E1 step 2: the twin's lanes, est.transform([x])[0] row by row (as
PythonTransform calls it), in whatever BLAS/numpy kernels the env selects.
Also the twin's own projected-mean constant M = mean_ @ components_.T."""
import pickle, sys, numpy as np, threadpoolctl
cases = pickle.load(open(sys.argv[1], "rb"))
lanes, Ms = [], []
for seed, est, x in cases:
    lanes.append(np.asarray(est.transform([x])[0], dtype=float))
    Ms.append((np.reshape(est.mean_, (1, -1)) @ est.components_.T)[0])
arch = threadpoolctl.threadpool_info()[0].get("architecture")
np.savez(sys.argv[2], lanes=np.concatenate(lanes), M=np.concatenate(Ms))
print("arch", arch, "lanes", sum(len(l) for l in lanes))
