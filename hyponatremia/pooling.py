"""Pooling of multiply imputed estimates (Rubin's rules, D1 multivariate Wald test)."""
import numpy as np
import pandas as pd
from scipy import stats


def rubin(coefs, ses):
    """Pooled estimate, total standard error and Barnard-Rubin degrees of freedom."""
    coefs = np.asarray(coefs)
    ses = np.asarray(ses)
    m = len(coefs)
    q = coefs.mean(0)
    u = (ses ** 2).mean(0)
    b = coefs.var(0, ddof=1)
    t = u + (1 + 1 / m) * b
    dof = (m - 1) * (1 + u / ((1 + 1 / m) * b + 1e-12)) ** 2
    return q, np.sqrt(t), dof


def d1_test(coefs, covs, idx):
    """D1 test of Li, Raghunathan and Rubin (1991) for the joint null on the coefficients in idx."""
    m = len(coefs)
    k = len(idx)
    Q = np.array([c[idx] for c in coefs])
    U = np.mean([V[np.ix_(idx, idx)] for V in covs], axis=0)
    qbar = Q.mean(0)
    B = np.cov(Q.T, ddof=1) if k > 1 else np.array([[Q.var(ddof=1)]])
    r = (1 + 1 / m) * np.trace(B @ np.linalg.inv(U)) / k
    T = (1 + r) * U
    D1 = float(qbar @ np.linalg.solve(T, qbar) / k)
    t = k * (m - 1)
    nu = 4 + (t - 4) * (1 + (1 - 2 / t) / r) ** 2 if t > 4 else t * (1 + 1 / k) * (1 + 1 / r) ** 2 / 2
    p = 1 - stats.f.cdf(D1, k, nu)
    return D1, k, nu, p, r


def d1_vector(estimates, covs):
    """D1 test for a vector of contrasts estimated in each imputation (pseudo-inverse for U)."""
    m = len(estimates)
    k = len(estimates[0])
    Q = np.array(estimates)
    U = np.mean(covs, axis=0)
    qbar = Q.mean(0)
    B = np.cov(Q.T, ddof=1) if k > 1 else np.array([[np.var(Q[:, 0], ddof=1)]])
    r = (1 + 1 / m) * np.trace(B @ np.linalg.pinv(U)) / k
    T = (1 + r) * U
    D1 = float(qbar @ np.linalg.solve(T, qbar) / k)
    t = k * (m - 1)
    nu = 4 + (t - 4) * (1 + (1 - 2 / t) / r) ** 2 if t > 4 else t * (1 + 1 / k) * (1 + 1 / r) ** 2 / 2
    return D1, k, nu, float(1 - stats.f.cdf(D1, k, nu))


def pooled_contrast(cvec, coefs, covs):
    """Pool a linear contrast c'beta across imputations; returns estimate, SE, lower, upper, dof."""
    est = np.array([float(cvec @ b) for b in coefs])
    var = np.array([float(cvec @ V @ cvec) for V in covs])
    m = len(est)
    q = est.mean()
    u = var.mean()
    b = est.var(ddof=1)
    total = u + (1 + 1 / m) * b
    dof = (m - 1) * (1 + u / ((1 + 1 / m) * b + 1e-12)) ** 2
    t = stats.t.ppf(0.975, dof)
    return q, np.sqrt(total), q - t * np.sqrt(total), q + t * np.sqrt(total), dof


def ratio_table(coefs, ses, names, label=None, measure="OR"):
    """Exponentiated pooled coefficients with t-based 95% CI and p values."""
    q, se, dof = rubin(coefs, ses)
    t = stats.t.ppf(0.975, dof)
    table = pd.DataFrame({"term": names, measure: np.exp(q), "lower": np.exp(q - t * se), "upper": np.exp(q + t * se),
                          "p": 2 * (1 - stats.t.cdf(np.abs(q / se), dof)), "rubin_df": dof})
    if label is not None:
        table.insert(0, "analysis", label)
    return table, q, se, dof


def percentile_ci(values, axis=0):
    return np.percentile(values, 2.5, axis=axis), np.percentile(values, 97.5, axis=axis)
