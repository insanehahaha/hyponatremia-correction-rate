"""Design matrix for the mortality models: correction-rate categories, age spline and covariates."""
import numpy as np
import pandas as pd

from .config import AGE_CENTER, AGE_KNOTS, AGE_SCALE, RATE_LEVELS, RATE_REF

RATE_COLS = [f"rate_{lvl}" for lvl in RATE_LEVELS if lvl != RATE_REF]
CORE_COLS = RATE_COLS + ["age", "age_s1", "age_s2", "female", "sodium_index", "prior_low", "prior_unknown",
                         "charlson", "egfr", "albumin", "glucose", "potassium", "any_diuretic", "year"]  # 17 df
INTERACTION_COLS = [f"{c}_x_age" for c in RATE_COLS]                                                    # + 3 df
ALL_COLS = CORE_COLS + INTERACTION_COLS
IDX = {c: i for i, c in enumerate(ALL_COLS)}
N_CORE = len(CORE_COLS)
RATE_IDX = [IDX[c] for c in RATE_COLS]
INTERACTION_IDX = [IDX[c] for c in INTERACTION_COLS]
AGE_IDX = [IDX["age"], IDX["age_s1"], IDX["age_s2"]]
IMPUTED = {"albumin": "albumin", "glucose": "glucose", "potassium": "potassium", "egfr": "egfr"}  # dataset column -> design column
IMPUTED_IDX = {IDX[c]: c for c in IMPUTED.values()}


def rcs(x, knots=AGE_KNOTS):
    """Restricted cubic spline basis (Harrell): linear term plus k-2 non-linear terms."""
    x = np.asarray(x, float)
    k = np.asarray(knots, float)
    m = len(k)
    out = [x]
    for j in range(m - 2):
        t = (np.maximum(x - k[j], 0) ** 3
             - np.maximum(x - k[m - 2], 0) ** 3 * (k[m - 1] - k[j]) / (k[m - 1] - k[m - 2])
             + np.maximum(x - k[m - 1], 0) ** 3 * (k[m - 2] - k[j]) / (k[m - 1] - k[m - 2])) / (k[m - 1] - k[0]) ** 2
        out.append(t)
    return np.column_stack(out)


def design_matrix(df):
    """Full design matrix (17 core columns + 3 interaction columns) in the order of ALL_COLS."""
    X = pd.DataFrame(index=df.index)
    for lvl in RATE_LEVELS:
        if lvl != RATE_REF:
            X[f"rate_{lvl}"] = (df["rate_category"].astype(str) == lvl).astype(float)
    spline = rcs(df["age"].values)
    X["age"], X["age_s1"], X["age_s2"] = spline[:, 0], spline[:, 1], spline[:, 2]
    X["female"] = (df["sex"] == "female").astype(float)
    X["sodium_index"] = df["sodium_index"].astype(float)
    X["prior_low"] = (df["prior_sodium"] == "low").astype(float)
    X["prior_unknown"] = (df["prior_sodium"] == "unknown").astype(float)
    X["charlson"] = df["charlson"].astype(float)
    X["egfr"] = df["egfr"].astype(float)
    X["albumin"] = df["albumin"].astype(float)
    X["glucose"] = df["glucose"].astype(float)
    X["potassium"] = df["potassium"].astype(float)
    X["any_diuretic"] = df["any_diuretic"].astype(float)
    X["year"] = (df["year"] - 2018).astype(float)
    for c in RATE_COLS:
        X[f"{c}_x_age"] = X[c] * (X["age"] - AGE_CENTER) / AGE_SCALE
    assert list(X.columns) == ALL_COLS
    return X


def refresh_interactions(X):
    """Recompute the age x rate columns of a numpy design matrix after age or rate were changed."""
    X[:, INTERACTION_IDX] = X[:, RATE_IDX] * ((X[:, IDX["age"]] - AGE_CENTER) / AGE_SCALE)[:, None]
    return X


def set_rate(X, level):
    """Assign every row to one correction-rate category."""
    Xn = X.copy()
    Xn[:, RATE_IDX] = 0.0
    if level != RATE_REF:
        Xn[:, IDX[f"rate_{level}"]] = 1.0
    return refresh_interactions(Xn)


def set_age(X, age):
    """Fix age (and its spline terms) at one value for every row."""
    Xn = X.copy()
    Xn[:, AGE_IDX] = rcs(np.full(len(Xn), float(age)))
    return refresh_interactions(Xn)


def rate_dummies(df, col="rate_category"):
    return pd.DataFrame({f"rate_{lvl}": (df[col].astype(str) == lvl).astype(float)
                         for lvl in RATE_LEVELS if lvl != RATE_REF}, index=df.index)
