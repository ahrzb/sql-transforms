"""Transform the SAME fitted estimators (from v1_default.pkl) under the current kernel."""
import pickle, sys, numpy as np, threadpoolctl, warnings
warnings.simplefilter("ignore")
d = pickle.load(open("v1_default.pkl", "rb"))
row = [np.array(d["ests"][k].transform([x])[0], float) for k, x in d["recs"]]
Ms = [(d["ests"][k].mean_.reshape(1, -1) @ d["ests"][k].components_.T)[0] for k, x in d["recs"]]
arch = threadpoolctl.threadpool_info()[0].get("architecture")
pickle.dump(dict(arch=arch, row=row, M=Ms), open(sys.argv[1], "wb"))
print(arch)
