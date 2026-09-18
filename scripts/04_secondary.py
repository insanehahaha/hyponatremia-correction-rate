"""Secondary outcomes (Table 4, Supplementary Tables).

A) 90-day and 1-year mortality: Cox models in the landmark cohort, time from presentation
   with entry at day 1.5 (time axis shifted by 1.5 days), censoring at the horizon; multiple
   imputation with the event indicator and the Nelson-Aalen cumulative hazard as auxiliary
   predictors; Schoenfeld tests in every imputation; RMST differences from pseudo-values.
B) 7-30-day mortality: 7-day landmark; exposure = rapid correction during days 0-7 (any of
   three criteria); logistic regression.
C) Days alive and out of hospital at 30 days (DAOH-30): proportional odds model, Brant test
   pooled with D1, partial proportional odds model for the variables that violate
   proportionality, median regression as sensitivity.
D) 30-day readmission among patients discharged alive: logistic regression.
E) Benjamini-Hochberg adjustment over the six secondary rate tests (including the
   neurological Fine-Gray test from scripts/03_neurological.py).

Input : analysis dataset, results/03_neurological.xlsx
Output: results/04_secondary.xlsx
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from lifelines import CoxPHFitter, KaplanMeierFitter, NelsonAalenFitter
from lifelines.exceptions import ConvergenceError
from lifelines.statistics import proportional_hazard_test
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyponatremia import data
from hyponatremia.config import AGE_BANDS, LANDMARK_DAYS, M_IMP, RATE_LEVELS, RESULTS, SEED_SECONDARY
from hyponatremia.design import ALL_COLS, CORE_COLS, INTERACTION_COLS, RATE_COLS, design_matrix
from hyponatremia.imputation import impute_frame, limits_by_name, rhat
from hyponatremia.models import fit_logit
from hyponatremia.ordinal import brant_components, brant_index, fit_partial_proportional_odds, fit_proportional_odds
from hyponatremia.pooling import d1_test, d1_vector, rubin

SEED = SEED_SECONDARY
CONVERGENCE, FITS, TESTS, OBSERVED, PH_SUMMARY = [], [], [], [], []


def convergence_rows(traces, label, setup):
    return [{"analysis": label, "imputation_predictors": setup, "variable": v,
             "rhat_mean": round(rhat([t[v]["mean"] for t in traces]), 3), "rhat_sd": round(rhat([t[v]["sd"] for t in traces]), 3),
             "truncated_last_iteration": int(sum(t[v]["truncated"] for t in traces))} for v in traces[0]]


def effect_table(names, q, se, dof, label, measure="HR"):
    t = stats.t.ppf(0.975, dof)
    return pd.DataFrame({"analysis": label, "term": names, measure: np.exp(q), "lower": np.exp(q - t * se), "upper": np.exp(q + t * se),
                         "p": 2 * (1 - stats.t.cdf(np.abs(q / se), dof))})


def standardize(X):
    mu = X.mean()
    sd = X.std().replace(0, 1.0)
    return (X - mu) / sd, mu, sd


df = data.load()
first = data.first_visits(df)
L = data.landmark_cohort(df)
LIMITS = limits_by_name(first)
rate_str = L["rate_category"].astype(str)


# ================================================================ A) Cox 90 days / 1 year
X20 = design_matrix(L)[ALL_COLS]                    # 17 core + 3 fully observed age x rate columns (imputation predictors)
SETUP_COX = "17 design columns + 3 age x rate columns + event indicator + Nelson-Aalen cumulative hazard"


def cox_analysis(tau, label, run_id):
    T = np.minimum(L["survival_days"].values, tau)
    E = np.where(L["survival_days"].values <= tau, (L["died"] == 1).astype(int).values, 0)
    H = NelsonAalenFitter().fit(T, event_observed=E).cumulative_hazard_at_times(T).values
    extra = np.column_stack([E, H])
    rng = np.random.default_rng([SEED, run_id])
    store = {"core": dict(c=[], v=[], s=[], fit=[]), "interaction": dict(c=[], v=[], s=[], fit=[])}
    traces, completed, ph_rows = [], [], []
    for i in range(M_IMP):
        Xi, tr = impute_frame(X20, extra, rng, LIMITS)
        traces.append(tr)
        completed.append(Xi)
        for key, cols in [("core", CORE_COLS), ("interaction", ALL_COLS)]:
            D = Xi[cols].copy()
            D["T"] = T - LANDMARK_DAYS
            D["E"] = E
            with warnings.catch_warnings(record=True) as records:
                warnings.simplefilter("always")
                try:
                    cph = CoxPHFitter().fit(D, duration_col="T", event_col="E")
                    ok = True
                except ConvergenceError:
                    cph, ok = None, False
            names = [type(w.message).__name__ for w in records if not issubclass(w.category, (FutureWarning, DeprecationWarning))]
            finite = bool(ok and np.isfinite(cph.params_.values).all() and np.isfinite(cph.variance_matrix_.values).all())
            store[key]["fit"].append({"imputation": i + 1, "model": key, "converged": ok, "finite": finite, "warnings": ";".join(names)})
            if not (ok and finite):
                continue                                # failed fits are not pooled
            store[key]["c"].append(cph.params_.values)
            store[key]["v"].append(cph.variance_matrix_.values)
            store[key]["s"].append(cph.standard_errors_.values)
            if key == "core":
                ph_rows.append(proportional_hazard_test(cph, D, time_transform="rank").summary["p"].rename(f"imputation {i + 1}"))
    q, se, dof = rubin(np.array(store["core"]["c"]), np.array(store["core"]["s"]))
    HR = effect_table(CORE_COLS, q, se, dof, label)
    d1_rate = d1_test(store["core"]["c"], store["core"]["v"], [CORE_COLS.index(c) for c in RATE_COLS])
    d1_int = d1_test(store["interaction"]["c"], store["interaction"]["v"], [ALL_COLS.index(c) for c in INTERACTION_COLS])
    PH = pd.concat(ph_rows, axis=1)
    for v in PH.index:
        PH_SUMMARY.append({"analysis": label, "term": v, "p_min": PH.loc[v].min(), "p_median": PH.loc[v].median(), "p_max": PH.loc[v].max(),
                           "imputations_p_below_0.05": int((PH.loc[v] < 0.05).sum()), "imputations": PH.shape[1]})
    fits = pd.DataFrame(store["core"]["fit"] + store["interaction"]["fit"])
    n_ok, n_ok_i = len(store["core"]["c"]), len(store["interaction"]["c"])
    violations = ", ".join(f"{v} ({int((PH.loc[v] < 0.05).sum())}/{PH.shape[1]})" for v in PH.index[(PH < 0.05).sum(axis=1) > 0])
    FITS.append({"analysis": label, "n": len(L), "events": int(E.sum()), "events_per_df": round(E.sum() / 17, 1),
                 "converged_core_interaction": f"{n_ok}/{M_IMP} / {n_ok_i}/{M_IMP}", "warnings": int((fits["warnings"] != "").sum()),
                 "rubin_df_min": round(float(np.min(dof)), 1),
                 "note": f"Schoenfeld (rank) p<0.05 for rate terms in {int((PH.loc[RATE_COLS] < 0.05).sum().sum())}/{3 * PH.shape[1]} imputation-terms; terms with violations: {violations}"})
    CONVERGENCE.extend(convergence_rows(traces, label, SETUP_COX))
    TESTS.append({"analysis": label, "test": "rate (3 df, D1)", "F": d1_rate[0], "df": d1_rate[1], "p": d1_rate[3]})
    TESTS.append({"analysis": label, "test": "age x rate (3 df, D1; not in the FDR family)", "F": d1_int[0], "df": d1_int[1], "p": d1_int[3]})
    for lvl in RATE_LEVELS:
        mask = (rate_str == lvl).values
        km = KaplanMeierFitter().fit(T[mask] - LANDMARK_DAYS, E[mask])
        s = float(km.survival_function_at_times(tau - LANDMARK_DAYS).iloc[0])
        ci = km.confidence_interval_survival_function_.iloc[-1]
        OBSERVED.append({"outcome": label, "group": lvl, "n": int(mask.sum()), "events": int(E[mask].sum()),
                         "observed_risk_pct": 100 * (1 - s), "lower": 100 * (1 - ci.iloc[1]), "upper": 100 * (1 - ci.iloc[0])})
    return HR, T, E, completed


HR90, T90, E90, X90 = cox_analysis(90.0, "death 90 d (Cox)", 1)
HR365, T365, E365, X365 = cox_analysis(365.0, "death 1 y (Cox)", 2)


def rmst_pseudo(T, E, tau, label, completed):
    """Kaplan-Meier pseudo-values theta_i = n RMST - (n-1) RMST(-i) from day 1.5 to the horizon; OLS with HC1 errors."""
    t0 = T - LANDMARK_DAYS
    tau0 = tau - LANDMARK_DAYS
    n = len(t0)

    def rmst(tt, ee):
        sf = KaplanMeierFitter().fit(tt, ee).survival_function_
        sf = sf[sf.index <= tau0]
        positive = sf.index.values > 0
        times = np.r_[0.0, sf.index.values[positive], tau0]
        surv = np.r_[1.0, sf.iloc[:, 0].values[positive]]
        return float(np.sum(np.diff(times) * surv))

    full = rmst(t0, E)
    theta = np.array([n * full - (n - 1) * rmst(np.delete(t0, i), np.delete(E, i)) for i in range(n)])
    coefs, ses = [], []
    for Xi in completed:
        res = sm.OLS(theta, sm.add_constant(Xi[CORE_COLS])).fit(cov_type="HC1")
        coefs.append(res.params.values[1:])
        ses.append(res.bse.values[1:])
    q, se, dof = rubin(np.array(coefs), np.array(ses))
    t = stats.t.ppf(0.975, dof)
    return pd.DataFrame({"analysis": label, "term": CORE_COLS, "rmst_difference_days": q, "lower": q - t * se, "upper": q + t * se,
                         "p": 2 * (1 - stats.t.cdf(np.abs(q / se), dof))})


RMST = pd.concat([rmst_pseudo(T90, E90, 90.0, "death 90 d, RMST days 1.5-90", X90),
                  rmst_pseudo(T365, E365, 365.0, "death 1 y, RMST days 1.5-365", X365)])


# ================================================================ B) 7-30-day mortality
L7 = L[~(L["survival_days_upper"] < 7)].copy()
crossing7 = int(((L7["survival_days_lower"] < 7) & (L7["died"] == 1)).sum())
k1 = L7["max_rise_24h_7d"] > 10
k2 = L7["max_rise_48h_7d"] > 18
k3 = L7["rise_120_to_140_5d"] == 1
assert ((k1 | k2 | k3).astype(int).values == L7["rapid_correction_7d"].astype(int).values).all(), "exposure flag does not match its three criteria"
only = {"24h>10": int((k1 & ~k2 & ~k3).sum()), "48h>18": int((k2 & ~k1 & ~k3).sum()), "Na<120 to >140 within 5 d": int((k3 & ~k1 & ~k2).sum())}
X7 = design_matrix(L7)[[c for c in CORE_COLS if c not in RATE_COLS]].copy()
X7.insert(0, "rapid_correction_7d", L7["rapid_correction_7d"].astype(float).values)
y7 = L7["death_30d"].values.astype(float)
rng = np.random.default_rng([SEED, 3])
coefs, ses, traces, fits = [], [], [], []
for i in range(M_IMP):
    Xi, tr = impute_frame(X7, y7[:, None], rng, LIMITS)
    traces.append(tr)
    r = fit_logit(Xi.values, y7)
    fits.append({"converged": bool(r["converged"] and r["finite"]), "warnings": ";".join(r["warnings"])})
    if r["converged"] and r["finite"]:
        coefs.append(r["params"])
        ses.append(r["bse"])
q, se, dof = rubin(np.array(coefs), np.array(ses))
LABEL7 = "death 7-30 d (logistic, 7-day landmark; exposure rapid correction days 0-7)"
OR7 = effect_table(["intercept"] + list(X7.columns), q, se, dof, LABEL7, "OR")
fits = pd.DataFrame(fits)
n_exposed = int(L7["rapid_correction_7d"].sum())
FITS.append({"analysis": "death 7-30 d (7-day landmark)", "n": len(L7), "events": int(y7.sum()), "events_per_df": f"{y7.sum() / 15:.1f} (15 df)",
             "converged_core_interaction": f"{int(fits['converged'].sum())}/{M_IMP}", "warnings": int((fits["warnings"] != "").sum()),
             "rubin_df_min": round(float(np.min(dof)), 1),
             "note": f"excluded (death before day 7): {len(L) - len(L7)}; death interval crossing day 7 (kept): {crossing7}; exposed {n_exposed} = "
                     + ", ".join(f"only {k}: {v}" for k, v in only.items()) + f", several criteria: {n_exposed - sum(only.values())}"})
CONVERGENCE.extend(convergence_rows(traces, "death 7-30 d", "15 design columns (exposure + core covariates) + 30-day death indicator"))
TESTS.append({"analysis": "death 7-30 d (7-day landmark)", "test": "rapid correction days 0-7 (1 df, Wald t)", "F": float((q[1] / se[1]) ** 2), "df": 1,
              "p": float(OR7.loc[OR7["term"] == "rapid_correction_7d", "p"].iloc[0])})
for g, label in [(0, "no rapid correction days 0-7 (reference)"), (1, "rapid correction days 0-7")]:
    mask = (L7["rapid_correction_7d"] == g).values
    OBSERVED.append({"outcome": "death 7-30 d (7-day landmark)", "group": label, "n": int(mask.sum()), "events": int(y7[mask].sum()),
                     "observed_risk_pct": 100 * y7[mask].mean(), "lower": np.nan, "upper": np.nan})


# ================================================================ C) DAOH-30
daoh_raw = L["daoh30"].values
assert daoh_raw.max() <= 30 and daoh_raw.min() >= 0
daoh = np.clip(np.floor(daoh_raw).astype(int), 0, 30)
true_zero = int((daoh_raw == 0).sum())
category_zero = int((daoh == 0).sum())
X17 = design_matrix(L)[CORE_COLS]
names = list(X17.columns)
cutpoints = [k for k in range(1, int(daoh.max()) + 1) if min((daoh >= k).sum(), (daoh < k).sum()) >= 50]
rng = np.random.default_rng([SEED, 4])
coefs, covs, ses, traces, fits, coefs_q, ses_q, completed, brant = [], [], [], [], [], [], [], [], []
for i in range(M_IMP):
    Xi, tr = impute_frame(X17, daoh_raw[:, None], rng, LIMITS)
    traces.append(tr)
    completed.append(Xi)
    Z, mu, sd = standardize(Xi)
    with warnings.catch_warnings(record=True) as records:
        warnings.simplefilter("always")
        om = fit_proportional_odds(daoh, Z)
    ok = bool(om.mle_retvals.get("converged", False))
    k_ = Z.shape[1]
    beta = om.params.values[:k_] / sd.values
    V = om.cov_params().values[:k_, :k_] / np.outer(sd.values, sd.values)
    fits.append({"converged": ok, "warnings": ";".join(type(w.message).__name__ for w in records if not issubclass(w.category, (FutureWarning, DeprecationWarning)))})
    if ok and np.isfinite(beta).all():
        coefs.append(beta)
        covs.append(V)
        ses.append(np.sqrt(np.diag(V)))
    brant.append(brant_components(Xi, daoh, cutpoints))
    qr = sm.QuantReg(daoh_raw, sm.add_constant(Xi)).fit(q=0.5, max_iter=5000)
    coefs_q.append(qr.params.values[1:])
    ses_q.append(qr.bse.values[1:])
q, se, dof = rubin(np.array(coefs), np.array(ses))
ORd = effect_table(names, q, se, dof, "DAOH-30 (proportional odds; OR > 1 = higher DAOH category)", "OR")
d1_daoh = d1_test(coefs, covs, [names.index(c) for c in RATE_COLS])
fits = pd.DataFrame(fits)
p_brant = brant[0][2]
n_cut = len(cutpoints)
d_all = [d for d, _, _ in brant]
v_all = [V for _, V, _ in brant]
ix_rate = brant_index(names, RATE_COLS, n_cut, p_brant)
brant_all = d1_vector(d_all, v_all)
brant_rate = d1_vector([d[ix_rate] for d in d_all], [V[np.ix_(ix_rate, ix_rate)] for V in v_all])
BRANT = pd.DataFrame([{"test": "Brant, all covariates (equal coefficients across cutpoints)", "cutpoints": n_cut, "D1_F": brant_all[0], "df1": brant_all[1], "p": brant_all[3]},
                      {"test": "Brant, rate category terms only", "cutpoints": n_cut, "D1_F": brant_rate[0], "df1": brant_rate[1], "p": brant_rate[3]}])
BRANT["cutpoints (Y >= k)"] = ", ".join(map(str, cutpoints))
BRANT["method"] = "binary logistic model per cutpoint in every imputation; differences beta_k - beta_1 with the Brant (1990) joint sandwich covariance; pooled over 20 imputations with D1; cutpoints with >= 50 observations on both sides"

# Brant per variable block and partial proportional odds for the violating blocks
BLOCKS = {"rate (3)": RATE_COLS, "age spline (3)": ["age", "age_s1", "age_s2"], "prior sodium (2)": ["prior_low", "prior_unknown"]}
for c in names:
    if not any(c in v for v in BLOCKS.values()):
        BLOCKS[c] = [c]
rows = []
for block, cols in BLOCKS.items():
    ix = brant_index(names, cols, n_cut, p_brant)
    res = d1_vector([d[ix] for d in d_all], [V[np.ix_(ix, ix)] for V in v_all])
    rows.append({"variable_block": block, "df_components": res[1], "D1_F": res[0], "p": res[3]})
BRANT_VAR = pd.DataFrame(rows)
BRANT_VAR["violation_p_below_0.05"] = BRANT_VAR["p"] < 0.05
BRANT_VAR["cutpoints"] = n_cut
RELAX = [c for block, cols in BLOCKS.items() if float(BRANT_VAR.loc[BRANT_VAR["variable_block"] == block, "p"].iloc[0]) < 0.05 for c in cols]
print("Brant violations, relaxed in the partial proportional odds model:", RELAX, flush=True)
PPO = {}
for form, constrained in [("constrained", True), ("unconstrained", False)]:
    if not RELAX:
        break
    pc, ps, pv, prow = [], [], [], []
    for i, Xi in enumerate(completed):
        f = fit_partial_proportional_odds(Xi, daoh, RELAX, constrained=constrained)
        prow.append({"imputation": i + 1, "valid": f["valid"], "optimizer_success": f["optimizer_success"], "max_abs_gradient": f["gnorm"], "gradient_tolerance": f["gtol"],
                     "newton_steps": f["newton_steps"], "hessian_min_eigenvalue": f["min_eigenvalue"], "hessian_pd": f["hessian_pd"],
                     "negative_probabilities_over_cells": f"{f['n_negative']} / {f['n_cells']}", "loglik_ppo": f["loglik"], "loglik_po": f["loglik_po"],
                     "loglik_gain": f["loglik"] - f["loglik_po"], "n_parameters": f["n_params"], "message": f["message"]})
        if f["valid"]:
            bi = f["beta_idx"]
            pc.append(f["beta"][bi])
            ps.append(np.sqrt(np.diag(f["cov"])[bi]))
            pv.append(f["cov"][np.ix_(bi, bi)])
            bnames = [names[j] for j in bi]
        print(f"PPO {form} imputation {i + 1}: valid={f['valid']} gradient={f['gnorm']:.1e} loglik={f['loglik']:.1f}", flush=True)
    fit_table = pd.DataFrame(prow)
    if len(pc) >= 2:
        q_, se_, dof_ = rubin(np.array(pc), np.array(ps))
        t = stats.t.ppf(0.975, dof_)
        d1_ = d1_test(pc, pv, [bnames.index(c) for c in RATE_COLS])
        table = pd.DataFrame({"model": f"DAOH-30 partial proportional odds ({form})", "term": bnames, "OR": np.exp(q_), "lower": np.exp(q_ - t * se_),
                              "upper": np.exp(q_ + t * se_), "p": 2 * (1 - stats.t.cdf(np.abs(q_ / se_), dof_))})
        d1_text = f"F={d1_[0]:.2f}, p={d1_[3]:.3f}"
    else:
        table, d1_text = pd.DataFrame(), "-"
    K = int(daoh.max())
    summary = {"form": form, "relaxed_variables": ", ".join(RELAX),
               "gamma_structure": "gamma_k = g * s_k, linear in the cutpoint (1 parameter per variable)" if constrained else "cutpoint-specific gamma_k (K parameters per variable; common beta dropped)",
               "categories_cutpoints": f"{K + 1} / {K}", "n_parameters": int(fit_table["n_parameters"].iloc[0]), "valid_fits": f"{len(pc)}/{M_IMP}",
               "optimizer_success": int(fit_table["optimizer_success"].sum()), "hessian_pd": int(fit_table["hessian_pd"].sum()),
               "loglik_gain_range": f"{fit_table['loglik_gain'].min():.3f}-{fit_table['loglik_gain'].max():.3f}",
               "negative_probabilities_total_over_cells": f"{sum(int(x.split(' / ')[0]) for x in fit_table['negative_probabilities_over_cells'])} / {sum(int(x.split(' / ')[1]) for x in fit_table['negative_probabilities_over_cells'])}",
               "rate_joint_D1": d1_text}
    PPO[form] = (table, fit_table, summary)
PPO_SUMMARY = pd.DataFrame([s for _, _, s in PPO.values()]) if PPO else pd.DataFrame([{"note": "no Brant violation; partial proportional odds model not required"}])
FITS.append({"analysis": "DAOH-30 ordinal", "n": len(L), "events": f"true DAOH = 0: {true_zero}; model category 0 (<1 day): {category_zero}", "events_per_df": "-",
             "converged_core_interaction": f"{int(fits['converged'].sum())}/{M_IMP}", "warnings": int((fits["warnings"] != "").sum()), "rubin_df_min": round(float(np.min(dof)), 1),
             "note": f"scale 0-30 (whole days); highest observed {daoh_raw.max():.2f} -> category {int(daoh.max())}; Brant (MI-D1): all p={brant_all[3]:.3f}, rate terms p={brant_rate[3]:.3f}; cutpoints {n_cut}"})
CONVERGENCE.extend(convergence_rows(traces, "DAOH-30", "17 design columns + DAOH-30 (continuous)"))
TESTS.append({"analysis": "DAOH-30 ordinal", "test": "rate (3 df, D1)", "F": d1_daoh[0], "df": d1_daoh[1], "p": d1_daoh[3]})
qq, seq, dofq = rubin(np.array(coefs_q), np.array(ses_q))
t = stats.t.ppf(0.975, dofq)
MEDIAN = pd.DataFrame({"analysis": "DAOH-30 median regression (sensitivity)", "term": names, "median_difference_days": qq, "lower": qq - t * seq, "upper": qq + t * seq,
                       "p": 2 * (1 - stats.t.cdf(np.abs(qq / seq), dofq))})
for lvl in RATE_LEVELS:
    mask = (rate_str == lvl).values
    d = daoh_raw[mask]
    OBSERVED.append({"outcome": "DAOH-30", "group": lvl, "n": int(mask.sum()), "events": int((d == 0).sum()), "observed_risk_pct": np.nan, "lower": np.nan, "upper": np.nan,
                     "daoh_median_iqr": f"{np.median(d):.1f} ({np.percentile(d, 25):.1f}-{np.percentile(d, 75):.1f})", "daoh_zero_pct": 100 * (d == 0).mean()})


# ================================================================ D) 30-day readmission
LR = L[L["readmission_30d"].notna()].copy()
excluded = L[L["readmission_30d"].isna()]
Xr = design_matrix(LR)[CORE_COLS]
yr = LR["readmission_30d"].values.astype(float)
rng = np.random.default_rng([SEED, 5])
coefs, ses, covs, traces, fits = [], [], [], [], []
for i in range(M_IMP):
    Xi, tr = impute_frame(Xr, yr[:, None], rng, LIMITS)
    traces.append(tr)
    r = fit_logit(Xi.values, yr)
    fits.append({"converged": bool(r["converged"] and r["finite"]), "warnings": ";".join(r["warnings"])})
    if r["converged"] and r["finite"]:
        coefs.append(r["params"])
        ses.append(r["bse"])
        covs.append(r["cov"])
names_r = ["intercept"] + list(Xr.columns)
q, se, dof = rubin(np.array(coefs), np.array(ses))
ORr = effect_table(names_r, q, se, dof, "readmission 30 d (logistic; discharged alive from the index admission)", "OR")
d1_readm = d1_test(coefs, covs, [names_r.index(c) for c in RATE_COLS])
fits = pd.DataFrame(fits)
FITS.append({"analysis": "readmission 30 d", "n": len(LR), "events": int(yr.sum()), "events_per_df": round(yr.sum() / 17, 1),
             "converged_core_interaction": f"{int(fits['converged'].sum())}/{M_IMP}", "warnings": int((fits["warnings"] != "").sum()), "rubin_df_min": round(float(np.min(dof)), 1),
             "note": f"denominator: discharged alive {len(LR)} ({int((LR['death_30d'] == 1).sum())} died within 30 days after discharge, kept); "
                     f"not in denominator {len(excluded)}: in-hospital death {int((excluded['in_hospital_death'] == 1).sum())}, death within 30 days {int((excluded['death_30d'] == 1).sum())}"})
CONVERGENCE.extend(convergence_rows(traces, "readmission 30 d", "17 design columns + readmission indicator"))
TESTS.append({"analysis": "readmission 30 d", "test": "rate (3 df, D1)", "F": d1_readm[0], "df": d1_readm[1], "p": d1_readm[3]})
for lvl in RATE_LEVELS:
    mask = (LR["rate_category"].astype(str) == lvl).values
    OBSERVED.append({"outcome": "readmission 30 d", "group": lvl, "n": int(mask.sum()), "events": int(yr[mask].sum()), "observed_risk_pct": 100 * yr[mask].mean(),
                     "lower": np.nan, "upper": np.nan})


# ================================================================ E) descriptive by age band
rows = []
for name, lo, hi in AGE_BANDS:
    mask = ((L["age"] >= lo) & (L["age"] < hi)).values
    d = daoh_raw[mask]
    r_ = L.loc[mask, "readmission_30d"]
    rows.append({"age_band": name, "n": int(mask.sum()), "death_90d_n_pct": f"{int(E90[mask].sum())} ({100 * E90[mask].mean():.1f})",
                 "death_1y_n_pct": f"{int(E365[mask].sum())} ({100 * E365[mask].mean():.1f})",
                 "daoh_median_iqr": f"{np.median(d):.1f} ({np.percentile(d, 25):.1f}-{np.percentile(d, 75):.1f})",
                 "daoh_zero_n_pct": f"{int((d == 0).sum())} ({100 * (d == 0).mean():.1f})",
                 "readmission_n_denominator_pct": f"{int(r_.sum())}/{int(r_.notna().sum())} ({100 * r_.mean():.1f})"})
BY_AGE = pd.DataFrame(rows)


# ================================================================ F) Benjamini-Hochberg over the six secondary rate tests
neuro = pd.read_excel(os.path.join(RESULTS, "03_neurological.xlsx"), sheet_name="joint_tests")
p_neuro = float(neuro[(neuro["model"] == "Fine-Gray") & (neuro["test"].str.startswith("rate"))]["p"].iloc[0])
TESTS = pd.DataFrame(TESTS)
fdr = TESTS[~TESTS["test"].str.contains("age x rate")][["analysis", "test", "p"]].copy()
fdr = pd.concat([fdr, pd.DataFrame([{"analysis": "neurological deterioration 14 d (Fine-Gray)", "test": "rate (3 df, robust Wald)", "p": p_neuro}])], ignore_index=True)
m_tests = len(fdr)
order = np.argsort(fdr["p"].values)
adjusted = np.empty(m_tests)
running = 1.0
for rank, idx in zip(range(m_tests, 0, -1), order[::-1]):
    running = min(running, fdr["p"].values[idx] * m_tests / rank)
    adjusted[idx] = running
fdr["p_bh"] = np.minimum(adjusted, 1.0)
fdr["family"] = "six secondary rate tests; age x rate and subgroup tests are not part of this family"

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, "04_secondary.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    pd.DataFrame(OBSERVED).to_excel(writer, sheet_name="observed_by_rate", index=False)
    BY_AGE.to_excel(writer, sheet_name="observed_by_age", index=False)
    pd.concat([HR90, HR365]).to_excel(writer, sheet_name="cox_90d_1y", index=False)
    pd.DataFrame(PH_SUMMARY).to_excel(writer, sheet_name="cox_schoenfeld", index=False)
    RMST.to_excel(writer, sheet_name="rmst", index=False)
    OR7.to_excel(writer, sheet_name="death_7_30d", index=False)
    ORd.to_excel(writer, sheet_name="daoh_ordinal", index=False)
    BRANT.to_excel(writer, sheet_name="daoh_brant", index=False)
    BRANT_VAR.to_excel(writer, sheet_name="daoh_brant_by_variable", index=False)
    for form, (table, fit_table, _) in PPO.items():
        if len(table):
            table.to_excel(writer, sheet_name=f"daoh_ppo_{form}", index=False)
        fit_table.to_excel(writer, sheet_name=f"daoh_ppo_{form}_fit", index=False)
    PPO_SUMMARY.to_excel(writer, sheet_name="daoh_ppo_summary", index=False)
    MEDIAN.to_excel(writer, sheet_name="daoh_median", index=False)
    ORr.to_excel(writer, sheet_name="readmission", index=False)
    TESTS.to_excel(writer, sheet_name="joint_tests", index=False)
    fdr.to_excel(writer, sheet_name="bh_fdr", index=False)
    pd.DataFrame(FITS).to_excel(writer, sheet_name="fit_summary", index=False)
    pd.DataFrame(CONVERGENCE).to_excel(writer, sheet_name="imputation_convergence", index=False)

print("written:", out)
pd.set_option("display.width", 250)
for tbl in [HR90, HR365, OR7, ORd, ORr]:
    print(tbl[tbl["term"].str.startswith(("rate", "rapid"))].round(3).to_string(index=False))
print(TESTS.round(3).to_string(index=False))
print(fdr.round(3).to_string(index=False))
print(BRANT_VAR.round(3).to_string(index=False))
print(pd.DataFrame(FITS).to_string(index=False))
