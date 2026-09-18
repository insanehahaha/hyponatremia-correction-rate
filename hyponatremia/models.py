"""Model fitting helpers: logistic regression (statsmodels and a fast lbfgs variant) and Cox models."""
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines import utils as lifelines_utils
from lifelines.exceptions import ConvergenceError
from scipy import stats
from scipy.special import expit
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


def _warning_names(records):
    return [type(w.message).__name__ for w in records
            if not issubclass(w.category, (FutureWarning, DeprecationWarning)) and "penalty=None will ignore" not in str(w.message)]


def fit_logit(X, y, maxiter=500):
    """Newton-Raphson logistic regression with convergence and separation checks."""
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        Xc = sm.add_constant(X, has_constant="add")
        res = sm.Logit(y, Xc).fit(disp=0, maxiter=maxiter)
    p = np.asarray(res.predict(Xc))
    params = np.asarray(res.params)
    bse = np.asarray(res.bse)
    return dict(b0=float(params[0]), beta=params[1:], params=params, bse=bse, cov=np.asarray(res.cov_params()), pred=p,
                converged=bool(res.mle_retvals.get("converged", False)), n_iter=int(res.mle_retvals.get("iterations", -1)),
                finite=bool(np.isfinite(params).all() and np.isfinite(bse).all()),
                extreme=int(((p < 1e-8) | (p > 1 - 1e-8)).sum()), warnings=_warning_names(records))


def fit_logit_fast(X, y, max_iter=1000):
    """Unpenalised logistic regression by lbfgs on standardised features (used inside the bootstrap)."""
    mu = X.mean(0)
    sd = X.std(0)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd
    names = []
    for iters in (max_iter, 10 * max_iter):
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            lr = LogisticRegression(C=np.inf, solver="lbfgs", max_iter=iters, tol=1e-6).fit(Z, y)
        names += _warning_names(records)
        n_iter = int(lr.n_iter_[0])
        converged = n_iter < iters and not any(issubclass(w.category, ConvergenceWarning) for w in records)
        if converged:
            break
    beta = lr.coef_[0] / sd
    b0 = float(lr.intercept_[0] - (lr.coef_[0] * mu / sd).sum())
    p = expit(b0 + X @ beta)
    return dict(b0=b0, beta=beta, converged=bool(converged), n_iter=n_iter, finite=bool(np.isfinite(beta).all()),
                extreme=int(((p < 1e-8) | (p > 1 - 1e-8)).sum()), warnings=names)


def predict(b0, beta, X):
    return expit(b0 + X[:, :len(beta)] @ beta)


def fit_cox(D, cols, robust=True):
    """Cox model with optional robust standard errors; returns (fitter or None, converged, warning names)."""
    names = []
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        cph = CoxPHFitter(penalizer=0.0)
        try:
            cph.fit(D[cols + ["T", "E", "w"]], duration_col="T", event_col="E", weights_col="w", robust=robust)
            ok = True
        except ConvergenceError as err:
            ok = False
            cph = None
            names.append(f"ConvergenceError: {str(err)[:80]}")
    names += [type(w.message).__name__ for w in records if not issubclass(w.category, (FutureWarning, DeprecationWarning))]
    return cph, ok, names


def sandwich(cph, D, cols):
    """Full robust (Huber-White) covariance matrix, matching lifelines' own standard errors."""
    S = D[cols + ["T", "E", "w"]].sort_values(by=["T", "E"])
    Xn = pd.DataFrame(lifelines_utils.normalize(S[cols].values, cph._norm_mean.values, cph._norm_std.values), index=S.index, columns=cols)
    V = cph._compute_sandwich_estimator(Xn, S["T"], S["E"], S["w"])
    assert np.max(np.abs(np.sqrt(np.diag(V)) - cph.standard_errors_[cols].values)) < 1e-8
    return pd.DataFrame(V, index=cols, columns=cols)


def wald(cph, names, V=None):
    """Joint Wald test for the coefficients in `names` (robust covariance if V is given)."""
    b = cph.params_[names].values
    Vm = (V if V is not None else cph.variance_matrix_).loc[names, names].values
    W = float(b @ np.linalg.solve(Vm, b))
    return W, len(names), float(stats.chi2.sf(W, len(names)))


def hazard_table(cph, label):
    s = cph.summary
    return pd.DataFrame({"model": label, "term": s.index, "HR": s["exp(coef)"].values, "lower": s["exp(coef) lower 95%"].values,
                         "upper": s["exp(coef) upper 95%"].values, "p": s["p"].values, "se_robust": s["se(coef)"].values})


def aalen_johansen(T, event_type, target, horizon=None):
    """Non-parametric cumulative incidence of `target` when censoring occurs only at the horizon."""
    t = np.asarray(T, float)
    e = np.asarray(event_type)
    surv = 1.0
    cif = 0.0
    for tt in np.sort(np.unique(t[e > 0])):
        at_risk = (t >= tt).sum()
        d_all = ((t == tt) & (e > 0)).sum()
        d_k = ((t == tt) & (e == target)).sum()
        cif += surv * d_k / at_risk
        surv *= 1 - d_all / at_risk
    return cif
