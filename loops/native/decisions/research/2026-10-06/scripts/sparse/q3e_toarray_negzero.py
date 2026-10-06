"""Q3e: does .toarray() preserve a STORED -0.0? (scipy's csr_todense
accumulates `out += data` into a zeroed buffer, so +0.0 + -0.0 = +0.0.)
Also duplicates in a non-canonical CSR are summed."""
import numpy as np
import scipy.sparse as sp

for name, M in (("csr_matrix", sp.csr_matrix), ("csr_array", sp.csr_array), ("csc_matrix(via csr)", lambda t, shape: sp.csc_matrix(sp.csr_matrix(t, shape=shape)))):
    m = M((np.array([-0.0, 1.0]), np.array([0, 2]), np.array([0, 2])), shape=(1, 3))
    a = m.toarray()
    print(f"{name}: stored {m.data.tolist()} signbit(data) {np.signbit(m.data).tolist()} -> toarray {a.tolist()} signbit {np.signbit(a).tolist()}")
    print(f"   todense signbit {np.signbit(np.asarray(m.todense())).tolist()}")
m = sp.coo_array((np.array([-0.0]), (np.array([0]),)), shape=(3,))
print("1-D coo_array stored -0.0 -> toarray", m.toarray().tolist(), np.signbit(m.toarray()).tolist())
dup = sp.csr_matrix((np.array([0.1, 0.2]), np.array([1, 1]), np.array([0, 2])), shape=(1, 3))
print("non-canonical CSR duplicates [0.1, 0.2] at col 1 -> toarray", dup.toarray().tolist(), "(0.1+0.2 =", 0.1 + 0.2, ")")
