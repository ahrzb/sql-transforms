"""Fit every (family, dataset) pipeline once, default env, and pickle it:
transformer -> downstream models, fitted as Pipeline.fit does (the
downstream model sees the transformer's fit_transform output)."""

import pickle
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, PLAN, case_path, datasets, make_transformer, standardize  # noqa: E402

warnings.filterwarnings("ignore")


def models():
    from sklearn.ensemble import (
        HistGradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    return {
        "logreg": (LogisticRegression(max_iter=5000), "clf"),
        "linreg": (LinearRegression(), "reg"),
        "dtree": (DecisionTreeClassifier(random_state=0), "clf"),
        "dtree_reg": (DecisionTreeRegressor(random_state=0, min_samples_leaf=5), "reg"),
        "rf": (RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=4), "clf"),
        "hgb": (HistGradientBoostingClassifier(random_state=0), "clf"),
    }


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    ds = datasets()
    only = sys.argv[1:] or list(PLAN)
    for fam in only:
        names, std = PLAN[fam]
        for dn in names:
            t0 = time.time()
            d = ds[dn]
            if std:
                Xtr, Xte = standardize(d)
            else:
                Xtr, Xte = d["Xtr"], d["Xte"]
            est = make_transformer(fam)
            Ftr = est.fit_transform(Xtr)
            fitted = {}
            for mn, (m, kind) in models().items():
                y = d["yclf"] if kind == "clf" else d["yreg"]
                fitted[mn] = m.fit(Ftr, y)
            case = dict(family=fam, dataset=dn, est=est, Xtr=Xtr, Xte=Xte, Ftr=Ftr,
                        yclf=d["yclf"], yreg=d["yreg"], models=fitted)
            with open(case_path(fam, dn), "wb") as f:
                pickle.dump(case, f)
            print(f"{fam:9s} {dn:13s} lanes={Ftr.shape[1]:4d} test={len(Xte)} "
                  f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
