"""The twin: sklearn AdditiveChi2Sampler.transform itself, one feature column."""
import sys, numpy as np, warnings
from sklearn.kernel_approximation import AdditiveChi2Sampler
from numpy.lib.introspect import opt_func_info
x = np.load("v3_x.npy").reshape(-1, 1); out = {}
cfgs = [(2, None), (3, None), (2, 0.8), (3, 0.5), (4, 0.4), (6, 0.3), (8, 0.7)]
for steps, iv in cfgs:
    T = AdditiveChi2Sampler(sample_steps=steps, sample_interval=iv).fit(x).transform(x)
    out[f"{steps}_{iv}"] = T
cosh = {f"{j}_{s}": float(np.cosh(np.pi * j * s)) for s in [0.8, 0.5, 0.4, 0.3, 0.7] for j in range(1, 8)}
np.savez(sys.argv[1], cosh_keys=list(cosh), cosh_vals=list(cosh.values()), log=np.log(x[:, 0]), **out)
print(opt_func_info(func_name="^log$", signature="float64")["log"]["dd"]["current"])
