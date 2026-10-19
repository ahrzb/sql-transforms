import sys, numpy as np
a = np.load(sys.argv[1], allow_pickle=True).item(); b = np.load(sys.argv[2], allow_pickle=True).item()
for key in a:
    d = (a[key] != b[key]); print(key, "MN=%d" % (key[0]*key[1]), "lanes differ:", int(d.sum()), "/", d.size, "max rel", float(np.max(np.abs(a[key]-b[key])/np.abs(a[key]).max())))
