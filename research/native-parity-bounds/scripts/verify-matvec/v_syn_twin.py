import pickle, sys, warnings
import numpy as np
from threadpoolctl import threadpool_info
warnings.simplefilter("ignore")
info = [d for d in threadpool_info() if d["internal_api"] == "openblas"][0]
data = pickle.load(open(sys.argv[1], "rb")); res = {}
for key, d in data.items():
    est, X = d["est"], d["X"]
    res[key] = dict(row=np.array([est.transform([x])[0] for x in X]), batch=est.transform(X),
                    mc=(est.mean_.reshape(1, -1) @ est.components_.T)[0],
                    dot_row=np.array([(x[None, :] @ est.components_.T)[0] for x in X]))
pickle.dump(res, open(sys.argv[2], "wb")); print(info["architecture"], info["num_threads"])
