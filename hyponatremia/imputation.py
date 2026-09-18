"""Multiple imputation of the four incomplete covariates by fully conditional specification.

Bayesian normal linear regression for each incomplete variable, using all other design
columns plus the outcome (or another outcome-related matrix) as predictors. Imputed values
are truncated to the range observed in the cohort.
"""
import numpy as np
import pandas as pd

from .config import N_ITER
from .design import IMPUTED, IMPUTED_IDX


def limits_from(df):
    """Observed [min, max] of each imputed variable, keyed by design-matrix column index."""
    inv = {v: k for k, v in IMPUTED.items()}
    return {j: (float(df[inv[name]].min()), float(df[inv[name]].max())) for j, name in IMPUTED_IDX.items()}


def limits_by_name(df):
    """Observed [min, max] of each imputed variable, keyed by design-matrix column name."""
    return {name: lim for (j, lim), name in zip(limits_from(df).items(), IMPUTED_IDX.values())}


def impute_array(X, y, rng, limits, n_iter=N_ITER, trace=False):
    """FCS on a numpy design matrix; predictors are the other columns and the outcome y.

    Returns the completed matrix and a diagnostics dict (per-iteration mean/sd of the
    imputed values, number of truncated draws, number missing).
    """
    X = X.copy()
    n = len(X)
    missing = {j: np.isnan(X[:, j]) for j in IMPUTED_IDX if np.isnan(X[:, j]).any()}
    diag = {"mean": {j: [] for j in missing}, "sd": {j: [] for j in missing},
            "mean_raw": {j: [] for j in missing}, "sd_raw": {j: [] for j in missing},
            "truncated": {j: 0 for j in missing}, "n_missing": {j: int(m.sum()) for j, m in missing.items()}}
    if not missing:
        return X, diag
    for j, m in missing.items():
        X[m, j] = np.nanmedian(X[:, j])
    for it in range(n_iter):
        for j, m in missing.items():
            A = np.column_stack([np.ones(n), np.delete(X, j, axis=1), y])
            obs = ~m
            Ao, yo = A[obs], X[obs, j]
            XtXi = np.linalg.inv(Ao.T @ Ao + 1e-6 * np.eye(A.shape[1]))
            beta = XtXi @ (Ao.T @ yo)
            resid = yo - Ao @ beta
            dof = max(int(obs.sum()) - A.shape[1], 1)
            sigma2 = (resid @ resid) / rng.chisquare(dof)
            draw = beta + np.linalg.cholesky(sigma2 * XtXi + 1e-12 * np.eye(A.shape[1])) @ rng.standard_normal(A.shape[1])
            v = A[m] @ draw + rng.normal(0, np.sqrt(sigma2), int(m.sum()))
            lo, hi = limits[j]
            if it == n_iter - 1:
                diag["truncated"][j] = int(((v < lo) | (v > hi)).sum())
            X[m, j] = np.clip(v, lo, hi)
            if trace:
                diag["mean"][j].append(float(X[m, j].mean()))
                diag["sd"][j].append(float(X[m, j].std()))
                diag["mean_raw"][j].append(float(v.mean()))
                diag["sd_raw"][j].append(float(v.std()))
    return X, diag


def impute_frame(X, extra, rng, limits, n_iter=N_ITER):
    """FCS on a design DataFrame; predictors are the other columns plus `extra` (n x k array).

    `limits` is keyed by design column name. Returns the completed frame and per-variable traces.
    """
    X = X.copy()
    cols = list(X.columns)
    A0 = X.values.astype(float)
    n = len(X)
    missing = {cols.index(c): np.isnan(A0[:, cols.index(c)]) for c in IMPUTED.values()
               if c in cols and np.isnan(A0[:, cols.index(c)]).any()}   # fixed variable order
    trace = {j: {"mean": [], "sd": [], "truncated": 0} for j in missing}
    for j, m in missing.items():
        A0[m, j] = np.nanmedian(A0[:, j])
    for it in range(n_iter):
        for j, m in missing.items():
            A = np.column_stack([np.ones(n), np.delete(A0, j, axis=1), extra])
            obs = ~m
            XtXi = np.linalg.inv(A[obs].T @ A[obs] + 1e-6 * np.eye(A.shape[1]))
            beta = XtXi @ (A[obs].T @ A0[obs, j])
            resid = A0[obs, j] - A[obs] @ beta
            dof = max(int(obs.sum()) - A.shape[1], 1)
            sigma2 = (resid @ resid) / rng.chisquare(dof)
            draw = beta + np.linalg.cholesky(sigma2 * XtXi + 1e-12 * np.eye(A.shape[1])) @ rng.standard_normal(A.shape[1])
            v = A[m] @ draw + rng.normal(0, np.sqrt(sigma2), int(m.sum()))
            lo, hi = limits[cols[j]]
            if it == n_iter - 1:
                trace[j]["truncated"] = int(((v < lo) | (v > hi)).sum())
            A0[m, j] = np.clip(v, lo, hi)
            trace[j]["mean"].append(A0[m, j].mean())
            trace[j]["sd"].append(A0[m, j].std())
    return pd.DataFrame(A0, columns=cols, index=X.index), {cols[j]: t for j, t in trace.items()}


def rhat(chains):
    """Gelman-Rubin R-hat over the second half of each chain (chains: m x n)."""
    c = np.asarray(chains)
    c = c[:, c.shape[1] // 2:]
    m, n = c.shape
    within = c.var(1, ddof=1).mean()
    between = n * c.mean(1).var(ddof=1)
    return float(np.sqrt(((n - 1) / n * within + between / n) / within)) if within > 0 else np.nan


def drift(chains):
    """Mean slope over the second half of the chains times half the chain length."""
    c = np.asarray(chains)
    c = c[:, c.shape[1] // 2:]
    t = np.arange(c.shape[1])
    return float(np.mean([np.polyfit(t, row, 1)[0] for row in c]) * (c.shape[1] - 1))
