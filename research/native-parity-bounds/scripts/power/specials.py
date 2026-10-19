import importlib.util, os, math
import numpy as np
from scipy import stats
spec = importlib.util.spec_from_file_location("km", os.path.join(os.path.dirname(os.path.abspath(__file__)), "kmeasure_lib.py"))
km = importlib.util.module_from_spec(spec); spec.loader.exec_module(km)
np.seterr(all="ignore")
print("np.log1p(-0.0)", np.log1p(np.array([-0.0])), "np.expm1(-0.0)", np.expm1(np.array([-0.0])))
bad = 0
for x in [0.0, -0.0, 5e-324, -5e-324, 1e-300, -1e-300, 1e-17, -2.5e-16, 1e300, -1e300, 1.7e308, -1.7e308]:
    for lam in [-50.0, -3.0, -2**-52, 0.0, 2**-53, 2**-52, 0.5, 1.0, 2 - 2**-52, 2.0, 2 + 2**-51, 2.5, 3.0, 50.0]:
        tw = float(stats.yeojohnson(np.array([x]), lam)[0])
        nt, _ = km.yj(x, lam)
        same = (repr(tw) == repr(nt)) or (math.isnan(tw) and math.isnan(nt))
        if not same:
            bad += 1
            print(f"x={x!r} lam={lam!r}: twin {tw!r} native {nt!r} ulps {km.ordered_dist(tw, nt) if math.isfinite(tw) and math.isfinite(nt) else 'inf/nan'}")
print("non-identical special lanes:", bad, "of", 12 * 14)
