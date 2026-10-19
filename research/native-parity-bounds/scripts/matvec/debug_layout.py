import pickle, sys, warnings
from collections import Counter
fixtures = pickle.load(open(sys.argv[1], "rb"))
cnt = Counter()
for fx in fixtures:
    for est in fx["instances"].values():
        C = est.components_
        cnt[(est._fit_svd_solver, C.flags.c_contiguous, C.flags.f_contiguous, C.shape[0] == 1 or C.shape[1] == 1)] += 1
for k, v in sorted(cnt.items()): print("solver, C-contig, F-contig, vector-shaped:", k, v)
