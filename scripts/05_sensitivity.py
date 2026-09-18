"""Sensitivity analyses, prespecified subgroups, spline dose-response and discharge model
(Supplementary Table S3, Figure 2).

All analyses of 30-day mortality use the primary logistic model structure (core covariates)
with multiple imputation (m = 20) and Rubin's rules; joint tests are D1 tests. No bootstrap.
  - alternative cohorts: all eligible episodes (patient-clustered SE), eligibility on measured
    sodium, Hillier glucose correction, landmark-uncertain excluded, complete case, admitted only
  - alternative exposures: binary thresholds >8 and >10, 48-hour rate (60-hour landmark),
    24-hour sodium measured at 18-30 h only, raw 24-hour change in sodium
  - subgroups: osmotic demyelination risk stratum, prior sodium status (single model with
    subgroup x rate interaction; within-subgroup >10 vs 4-8 contrasts)
  - dose-response: restricted cubic spline of the rate (4 knots), reference 6 mmol/L/24 h
  - discharge alive from index care: Fine-Gray model with death as competing event

Input : analysis dataset
Output: results/05_sensitivity.xlsx
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter
from lifelines.exceptions import ConvergenceError
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyponatremia import data
from hyponatremia.config import LANDMARK_DAYS, M_IMP, RATE_LEVELS, RATE_REF, RESULTS, SEED_SENSITIVITY
from hyponatremia.design import CORE_COLS, RATE_COLS, design_matrix, rate_dummies, rcs
from hyponatremia.imputation import impute_frame, limits_by_name, rhat
from hyponatremia.models import sandwich
from hyponatremia.pooling import d1_test, pooled_contrast, rubin

SEED = SEED_SENSITIVITY
COVARIATES = [c for c in CORE_COLS if c not in RATE_COLS]      # 14 core covariate columns
DISCHARGE_HORIZON = 30.0

df = data.load()
first = data.first_visits(df)
L = data.landmark_cohort(df)
LIMITS = limits_by_name(first)
y_L = L["death_30d"].values.astype(float)
OR_ALL, TESTS, FITS = [], [], []


def design(d, exposure):
    """Exposure columns followed by the 14 core covariates."""
    return pd.concat([exposure.astype(float), design_matrix(d)[COVARIATES]], axis=1)


def logistic_mi(X, y, label, exposure_cols, run_id, cluster=None, complete_case=False, extra_test=None):
    """Logistic regression with multiple imputation; D1 test of the exposure columns.

    cluster: patient identifiers for cluster-robust covariance (GLM). complete_case: single fit
    on complete rows. extra_test: (name, columns) for an additional D1 test.
    """
    rng = np.random.default_rng([SEED, run_id])
    coefs, ses, covs, fits, traces = [], [], [], [], []
    if complete_case:
        keep = X.notna().all(axis=1).values
        X, y = X[keep], y[keep]
        cluster = cluster[keep] if cluster is not None else None
    M = 1 if complete_case else M_IMP
    for i in range(M):
        Xi, tr = (X.copy(), {}) if complete_case else impute_frame(X, y[:, None], rng, LIMITS)
        traces.append(tr)
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            try:
                if cluster is not None:
                    r = sm.GLM(y, sm.add_constant(Xi, has_constant="add"), family=sm.families.Binomial()).fit(cov_type="cluster", cov_kwds={"groups": cluster})
                else:
                    r = sm.Logit(y, sm.add_constant(Xi, has_constant="add")).fit(disp=0, maxiter=2000)
                converged = bool(r.mle_retvals.get("converged", True)) if hasattr(r, "mle_retvals") else bool(getattr(r, "converged", True))
                par, se, cov = np.asarray(r.params), np.asarray(r.bse), np.asarray(r.cov_params())
            except Exception:
                converged, par, se, cov = False, None, None, None
        names_w = [type(w.message).__name__ for w in records if not issubclass(w.category, (FutureWarning, DeprecationWarning))]
        ok = converged and par is not None and np.isfinite(par).all() and np.isfinite(se).all()
        fits.append({"converged": ok, "warnings": ";".join(names_w)})
        if ok:
            coefs.append(par)
            ses.append(se)
            covs.append(cov)
    names = ["intercept"] + list(X.columns)
    if len(coefs) == 1:
        q, se_, dof = coefs[0], ses[0], np.full(len(coefs[0]), 1e6)
    else:
        q, se_, dof = rubin(np.array(coefs), np.array(ses))
    t = stats.t.ppf(0.975, dof)
    OR = pd.DataFrame({"analysis": label, "term": names, "OR": np.exp(q), "lower": np.exp(q - t * se_), "upper": np.exp(q + t * se_),
                       "p": 2 * (1 - stats.t.cdf(np.abs(q / se_), dof))})
    idx = [names.index(c) for c in exposure_cols]
    if len(coefs) == 1:
        W = float(q[idx] @ np.linalg.solve(covs[0][np.ix_(idx, idx)], q[idx]))
        D1 = (W / len(idx), len(idx), np.inf, float(stats.chi2.sf(W, len(idx))))
    else:
        D1 = d1_test(coefs, covs, idx)
    extra = None
    if extra_test and len(coefs) > 1:
        extra = d1_test(coefs, covs, [names.index(c) for c in extra_test[1]])
    fits = pd.DataFrame(fits)
    rhat_max = (f"{max(max(rhat([t_[v]['mean'] for t_ in traces]), rhat([t_[v]['sd'] for t_ in traces])) for v in traces[0]):.3f}"
                if traces[0] else "-")
    fit = {"analysis": label, "n": len(X), "events": int(y.sum()), "df": len(X.columns), "events_per_df": round(y.sum() / len(X.columns), 1),
           "converged": f"{int(fits['converged'].sum())}/{M}", "warnings": int((fits["warnings"] != "").sum()),
           "imputation": "none (complete case)" if complete_case else f"FCS m={M_IMP}, predictors: design + outcome; max R-hat {rhat_max}",
           "se": "cluster-robust by patient (GLM)" if cluster is not None else "model-based",
           "rubin_df_min": round(float(np.min(dof)), 1) if len(coefs) > 1 else "-"}
    return OR, D1, extra, fit, coefs, covs


def record(OR, D1, fit, extra=None, extra_name=None):
    OR_ALL.append(OR)
    FITS.append(fit)
    TESTS.append({"analysis": fit["analysis"], "test": "exposure (joint)", "statistic": D1[0], "df": D1[1], "p": D1[3]})
    if extra is not None:
        TESTS.append({"analysis": fit["analysis"], "test": extra_name, "statistic": extra[0], "df": extra[1], "p": extra[3]})


# ---------------------------------------------------------------- alternative cohorts and exposures
OR, D1, _, fit, _, _ = logistic_mi(design(L, rate_dummies(L)), y_L, "primary model structure (landmark cohort, n=1282)", RATE_COLS, 0)
record(OR, D1, fit)

A = df[(df["episode_sensitivity"] == 1) & (df["landmark_alive"] == 1)].copy()
OR, D1, _, fit, _, _ = logistic_mi(design(A, rate_dummies(A)), A["death_30d"].values.astype(float),
                                   f"all eligible episodes (patient-clustered SE), n={len(A)}, patients={A['patient_id'].nunique()}", RATE_COLS, 1,
                                   cluster=A["patient_id"].astype("category").cat.codes.values)
record(OR, D1, fit)

for col, label, run_id in [("measured_sodium_cohort", "eligibility on measured sodium (no glucose correction)", 2),
                           ("hillier_cohort", "Hillier glucose correction (2.4)", 3)]:
    S = df[(df[col] == 1) & (df["landmark_alive"] == 1)].copy()
    OR, D1, _, fit, _, _ = logistic_mi(design(S, rate_dummies(S)), S["death_30d"].values.astype(float), f"{label}, n={len(S)}", RATE_COLS, run_id)
    record(OR, D1, fit)

for threshold, run_id in [(8, 4), (10, 5)]:
    e = pd.DataFrame({f"rate>{threshold}": (L["rate_24h"] > threshold).astype(float)}, index=L.index)
    OR, D1, _, fit, _, _ = logistic_mi(design(L, e), y_L, f"binary threshold: rate >{threshold} vs <={threshold} mmol/L/24 h (n=1282)", [f"rate>{threshold}"], run_id)
    record(OR, D1, fit)

# 48-hour rate: measurements at 36-60 h, so the risk set is patients alive at 60 h
K2 = L[L["rate_48h"].notna()].copy()
K2A = K2[~(K2["survival_days_upper"] < 2.5)].copy()
e = pd.DataFrame({"rate48>18": (K2A["rate_48h"] > 18).astype(float)}, index=K2A.index)
OR, D1, _, fit, _, _ = logistic_mi(design(K2A, e), K2A["death_30d"].values.astype(float),
                                   f"48-hour rate >18 vs <=18 mmol/L/48 h, 60-hour landmark, n={len(K2A)} ({int((K2['survival_days_upper'] < 2.5).sum())} certain deaths before 60 h excluded)",
                                   ["rate48>18"], 6)
record(OR, D1, fit)

NW = L[L["sodium_24h_time"].between(18, 30)].copy()
OR, D1, _, fit, _, _ = logistic_mi(design(NW, rate_dummies(NW)), NW["death_30d"].values.astype(float), f"24-hour sodium measured at 18-30 h only, n={len(NW)}", RATE_COLS, 8)
record(OR, D1, fit)

e = pd.DataFrame({"delta_sodium_24h": L["delta_sodium_24h"].astype(float)}, index=L.index)
OR, D1, _, fit, _, _ = logistic_mi(design(L, e), y_L, "raw change in sodium 0-24 h, linear, OR per 1 mmol/L (n=1282)", ["delta_sodium_24h"], 9)
record(OR, D1, fit)

LU = L[L["landmark_uncertain"] == 0].copy()
OR, D1, _, fit, _, _ = logistic_mi(design(LU, rate_dummies(LU)), LU["death_30d"].values.astype(float), f"death interval crossing the landmark excluded, n={len(LU)}", RATE_COLS, 10)
record(OR, D1, fit)

OR, D1, _, fit, _, _ = logistic_mi(design(L, rate_dummies(L)), y_L, "complete case (no imputation)", RATE_COLS, 11, complete_case=True)
record(OR, D1, fit)

AD = L[L["admitted"] == 1].copy()
OR, D1, _, fit, _, _ = logistic_mi(design(AD, rate_dummies(AD)), AD["death_30d"].values.astype(float), f"confirmed hospital admission only, n={len(AD)}", RATE_COLS, 12)
record(OR, D1, fit)


# ---------------------------------------------------------------- prespecified subgroups
SUBGROUP_OR, SUBGROUP_TEST = [], []


def subgroup(d, col, reference, others, label, run_id, drop_core=()):
    """Single model: rate + subgroup indicators + rate x subgroup; within-subgroup >10 vs 4-8 contrasts."""
    y = d["death_30d"].values.astype(float)
    g = d[col].astype(str)
    hz = rate_dummies(d)
    G = pd.DataFrame({f"g_{lvl}": (g == lvl).astype(float) for lvl in others}, index=d.index)
    inter = pd.DataFrame({f"{h}_x_{gc}": hz[h] * G[gc] for h in RATE_COLS for gc in G.columns}, index=d.index)
    X = design(d, pd.concat([hz, G, inter], axis=1)).drop(columns=list(drop_core))
    OR, D1, extra, fit, _, _ = logistic_mi(X, y, f"subgroup: {label}", RATE_COLS, run_id, extra_test=("rate x subgroup interaction", list(inter.columns)))
    FITS.append(fit)
    SUBGROUP_TEST.append({"subgroup": label, "interaction_test_D1": f"F={extra[0]:.2f}, df={extra[1]}", "p": extra[3]})
    rng = np.random.default_rng([SEED, run_id])
    coefs, covs = [], []
    for _ in range(M_IMP):
        Xi, _ = impute_frame(X, y[:, None], rng, LIMITS)
        r = sm.Logit(y, sm.add_constant(Xi, has_constant="add")).fit(disp=0, maxiter=2000)
        if not r.mle_retvals.get("converged", True):
            continue
        coefs.append(np.asarray(r.params))
        covs.append(np.asarray(r.cov_params()))
    names = ["intercept"] + list(X.columns)
    top = f"rate_{RATE_LEVELS[-1]}"
    for lvl in [reference] + others:
        mask = g == lvl
        c = np.zeros(len(names))
        c[names.index(top)] = 1
        if lvl != reference:
            c[names.index(f"{top}_x_g_{lvl}")] = 1
        q_, se_, lo_, hi_, dof_ = pooled_contrast(c, coefs, covs)
        SUBGROUP_OR.append({"subgroup": label, "level": lvl, "n": int(mask.sum()), "deaths": int(d.loc[mask, "death_30d"].sum()),
                            f"OR {RATE_LEVELS[-1]} vs {RATE_REF}": np.exp(q_), "lower": np.exp(lo_), "upper": np.exp(hi_), "rubin_df": round(dof_, 1), "valid_imputations": len(coefs)})


subgroup(L, "ods_high_risk", "0", ["1"], "osmotic demyelination high-risk stratum (0 = no risk factor, 1 = at least one)", 20)
subgroup(L, "prior_sodium", "normal", ["unknown", "low"], "prior sodium status", 21, drop_core=["prior_low", "prior_unknown"])
SUBGROUP_TEST = pd.DataFrame(SUBGROUP_TEST)
SUBGROUP_TEST["p_bh"] = multipletests(SUBGROUP_TEST["p"].values, method="fdr_bh")[1]
SUBGROUP_TEST.loc[len(SUBGROUP_TEST)] = ["patients receiving haemodialysis", "not estimable: no validated haemodialysis variable in the dataset", np.nan, np.nan]
SUBGROUP_OR = pd.DataFrame(SUBGROUP_OR)


# ---------------------------------------------------------------- spline dose-response (reference 6 mmol/L/24 h)
knots = np.percentile(L["rate_24h"], [5, 35, 65, 95])
basis = rcs(L["rate_24h"].values, knots)
e = pd.DataFrame({"rate_rcs1": basis[:, 0], "rate_rcs2": basis[:, 1], "rate_rcs3": basis[:, 2]}, index=L.index)
X = design(L, e)
rng = np.random.default_rng([SEED, 30])
coefs, covs = [], []
for _ in range(M_IMP):
    Xi, _ = impute_frame(X, y_L[:, None], rng, LIMITS)
    r = sm.Logit(y_L, sm.add_constant(Xi, has_constant="add")).fit(disp=0, maxiter=2000)
    if r.mle_retvals.get("converged", True):
        coefs.append(np.asarray(r.params))
        covs.append(np.asarray(r.cov_params()))
names = ["intercept"] + list(X.columns)
ix = [names.index(c) for c in e.columns]
grid = np.arange(-4, 24.5, 0.5)
ref_basis = rcs(np.array([6.0]), knots)[0]
rows = []
for g_ in grid:
    c = np.zeros(len(names))
    c[ix] = rcs(np.array([g_]), knots)[0] - ref_basis
    q_, se_, lo_, hi_, _ = pooled_contrast(c, coefs, covs)
    rows.append({"rate_mmol_L_24h": g_, "OR_vs_6": np.exp(q_), "lower": np.exp(lo_), "upper": np.exp(hi_)})
SPLINE = pd.DataFrame(rows)
d1_spline = d1_test(coefs, covs, ix)
d1_nonlinear = d1_test(coefs, covs, ix[1:])
TESTS.append({"analysis": "spline dose-response (rate RCS, 4 knots)", "test": "rate (3 df, D1)", "statistic": d1_spline[0], "df": d1_spline[1], "p": d1_spline[3]})
TESTS.append({"analysis": "spline dose-response (rate RCS, 4 knots)", "test": "non-linearity (2 df, D1)", "statistic": d1_nonlinear[0], "df": d1_nonlinear[1], "p": d1_nonlinear[3]})


# ---------------------------------------------------------------- discharge alive from index care: Fine-Gray
def discharge_fine_gray(Ld, t0, label, run_id):
    """Competing event: death during index care; censoring at day 30 (still in hospital); Geskus weights equal 1."""
    t_raw = Ld["discharge_day"].values.astype(float)
    death = (Ld["in_hospital_death"] == 1).values
    event = np.where(death, 2, 1)
    T = np.where(death, np.minimum(Ld["survival_days"].values, DISCHARGE_HORIZON), np.minimum(t_raw, DISCHARGE_HORIZON))
    event = np.where(T >= DISCHARGE_HORIZON, 0, event)
    assert (T >= t0).all(), "event before the start of the risk set"
    Xd = design(Ld, rate_dummies(Ld))
    cols = list(Xd.columns)
    extra = np.column_stack([(event == 1).astype(float), (event == 2).astype(float), T])
    time = np.maximum(np.where(event == 2, DISCHARGE_HORIZON, T) - t0, 1e-6)
    E = (event == 1).astype(int)
    rng = np.random.default_rng([SEED, run_id])
    coefs, ses, covs, fits = [], [], [], []
    for _ in range(M_IMP):
        Xi, _ = impute_frame(Xd, extra, rng, LIMITS)
        D = Xi.copy()
        D["T"] = time
        D["E"] = E
        D["w"] = 1.0
        with warnings.catch_warnings(record=True) as records:
            warnings.simplefilter("always")
            try:
                cph = CoxPHFitter().fit(D, duration_col="T", event_col="E", weights_col="w", robust=True)
                ok = True
            except ConvergenceError:
                cph, ok = None, False
        names_w = [type(w.message).__name__ for w in records if not issubclass(w.category, (FutureWarning, DeprecationWarning))]
        fits.append({"converged": ok, "warnings": ";".join(names_w)})
        if ok:
            coefs.append(cph.params_.values)
            ses.append(cph.standard_errors_.values)
            covs.append(sandwich(cph, D, cols).values)
    q, se_, dof = rubin(np.array(coefs), np.array(ses))
    t = stats.t.ppf(0.975, dof)
    FG = pd.DataFrame({"analysis": label, "term": cols, "sHR": np.exp(q), "lower": np.exp(q - t * se_), "upper": np.exp(q + t * se_),
                       "p": 2 * (1 - stats.t.cdf(np.abs(q / se_), dof))})
    D1 = d1_test(coefs, covs, [cols.index(c) for c in RATE_COLS])
    fits = pd.DataFrame(fits)
    FITS.append({"analysis": label, "n": len(Ld), "events": int((event == 1).sum()), "df": len(cols), "events_per_df": round((event == 1).sum() / len(cols), 1),
                 "converged": f"{int(fits['converged'].sum())}/{M_IMP}", "warnings": int((fits["warnings"] != "").sum()),
                 "imputation": f"FCS m={M_IMP}, predictors: design + event type (discharge/death) + observed time; competing deaths {int((event == 2).sum())}; censored at day 30 {int((event == 0).sum())}",
                 "se": "robust (lifelines); joint test with the full sandwich covariance and D1", "rubin_df_min": round(float(np.min(dof)), 1)})
    TESTS.append({"analysis": label, "test": "rate (3 df, D1, sandwich)", "statistic": D1[0], "df": D1[1], "p": D1[3]})
    return FG


early = L[L["discharge_day"] < LANDMARK_DAYS]
LA = L[L["discharge_day"] >= LANDMARK_DAYS].copy()
FG = discharge_fine_gray(LA, LANDMARK_DAYS,
                         f"discharge alive, Fine-Gray: alive and not yet discharged at the landmark, n={len(LA)} ({len(early)} early discharges outside the risk set; {int((early['in_hospital_death'] == 1).sum())} of them died in hospital)", 40)


# ---------------------------------------------------------------- output
OR_ALL = pd.concat(OR_ALL, ignore_index=True)
rows = []
for a in OR_ALL["analysis"].unique():
    x = OR_ALL[OR_ALL["analysis"] == a]
    for _, r in x[~x["term"].isin(["intercept"] + COVARIATES)].iterrows():
        rows.append({"analysis": a, "exposure_contrast": r["term"], "OR": r["OR"], "lower": r["lower"], "upper": r["upper"], "p": r["p"]})
SUMMARY = pd.DataFrame(rows)

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, "05_sensitivity.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    SUMMARY.to_excel(writer, sheet_name="sensitivity_summary", index=False)
    OR_ALL.to_excel(writer, sheet_name="or_all_terms", index=False)
    pd.DataFrame(TESTS).to_excel(writer, sheet_name="joint_tests", index=False)
    SUBGROUP_OR.to_excel(writer, sheet_name="subgroup_or", index=False)
    SUBGROUP_TEST.to_excel(writer, sheet_name="subgroup_interaction_bh", index=False)
    SPLINE.to_excel(writer, sheet_name="spline_dose_response", index=False)
    FG.to_excel(writer, sheet_name="discharge_fine_gray", index=False)
    pd.DataFrame(FITS).to_excel(writer, sheet_name="fit_summary", index=False)

pd.set_option("display.width", 250)
print("written:", out)
print(SUMMARY.round(3).to_string(index=False))
print(pd.DataFrame(TESTS).round(3).to_string(index=False))
print(SUBGROUP_OR.round(2).to_string(index=False))
print(SUBGROUP_TEST.round(3).to_string(index=False))
print(FG[FG["term"].str.startswith("rate")].round(3).to_string(index=False))
