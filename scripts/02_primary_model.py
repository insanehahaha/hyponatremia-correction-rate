"""Primary analysis: correction rate and 30-day mortality (Table 2, Table 3 mortality panel, Figure 3A).

Multivariable logistic regression with multiple imputation (m = 20), Rubin's rules and D1
joint tests; standardized risks and risk differences with patient-level bootstrap
confidence intervals (imputation repeated in every resample). The analysis is run for the
landmark cohort (primary) and, with the same model, from two other time origins.

Input : analysis dataset
Output: results/02_primary_model.xlsx, results/fig_imputation_traces.png
Usage : python scripts/02_primary_model.py [B=2000] [n_jobs]
"""
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyponatremia import data
from hyponatremia.config import AGE_BANDS, AGE_GRID, B_BOOT, M_IMP, N_ITER, RATE_LEVELS, RATE_REF, RESULTS, SEED_PRIMARY
from hyponatremia.design import ALL_COLS, CORE_COLS, IMPUTED_IDX, INTERACTION_COLS, N_CORE, RATE_COLS, design_matrix, set_age, set_rate
from hyponatremia.imputation import drift, impute_array, limits_from, rhat
from hyponatremia.models import fit_logit, fit_logit_fast, predict
from hyponatremia.pooling import d1_test, percentile_ci, ratio_table

B = int(sys.argv[1]) if len(sys.argv) > 1 else B_BOOT
N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 2)
SEED = SEED_PRIMARY
REF_J = RATE_LEVELS.index(RATE_REF)


# ---------- standardized risks
def standardized_risk(b0, beta, X):
    """Population-average risk when every patient is assigned to each rate category in turn."""
    return np.array([predict(b0, beta, set_rate(X, lvl)).mean() for lvl in RATE_LEVELS])


def age_band_risk(b0, beta, X, age):
    """Standardized risk within each age band (members of the band only)."""
    return np.array([[predict(b0, beta, set_rate(X[(age >= lo) & (age < hi)], lvl)).mean() for lvl in RATE_LEVELS]
                     for _, lo, hi in AGE_BANDS])


def age_grid_risk(b0, beta, X):
    """Standardized risk with every patient's age fixed at each grid value."""
    out = np.zeros((len(AGE_GRID), len(RATE_LEVELS)))
    for i, a in enumerate(AGE_GRID):
        Xa = set_age(X, a)
        for j, lvl in enumerate(RATE_LEVELS):
            out[i, j] = predict(b0, beta, set_rate(Xa, lvl)).mean()
    return out


# ---------- bootstrap
def bootstrap_replicate(bb, run_id, X, y, age, limits, full):
    rng = np.random.default_rng([SEED, run_id, bb + 1])
    n = len(y)
    ii = rng.integers(0, n, n)
    Xb, yb, ageb = X[ii], y[ii], age[ii]
    out = dict(bb=bb, status="ok", reason="", nonconverged_core=0, nonconverged_int=0, warnings=0, extreme=0, max_iter=0)
    if yb.sum() < 5:
        out.update(status="failed", reason="too few events")
        return out
    try:
        risks, bands, grids = [], [], []
        for _ in range(M_IMP):
            Xi, _ = impute_array(Xb, yb, rng, limits)
            core = fit_logit_fast(Xi[:, :N_CORE], yb)
            inter = fit_logit_fast(Xi, yb)
            out["nonconverged_core"] += int(not core["converged"])
            out["nonconverged_int"] += int(not inter["converged"])
            out["warnings"] += len(core["warnings"]) + len(inter["warnings"])
            out["extreme"] += core["extreme"] + inter["extreme"]
            out["max_iter"] = max(out["max_iter"], core["n_iter"], inter["n_iter"])
            if not (core["finite"] and inter["finite"]):
                raise ValueError("non-finite coefficient")
            risks.append(standardized_risk(core["b0"], core["beta"], Xi))
            if full:
                bands.append(age_band_risk(inter["b0"], inter["beta"], Xi, ageb))
                grids.append(age_grid_risk(inter["b0"], inter["beta"], Xi))
        if out["nonconverged_core"] or out["nonconverged_int"]:
            out.update(status="failed", reason="not converged")
            return out
        out["risk"] = np.mean(risks, 0)
        if full:
            out["band"] = np.mean(bands, 0)
            out["grid"] = np.mean(grids, 0)
    except Exception as err:
        out.update(status="failed", reason=f"exception: {type(err).__name__}: {err}"[:150])
    return out


# ---------- main analysis for one time origin
def analyse(df, label, run_id, limits, full=True):
    X0 = design_matrix(df)
    X = X0.values.astype(float)
    y = df["death_30d"].values.astype(float)
    age = df["age"].values.astype(float)
    n, events = len(df), int(y.sum())
    print(f"\n== {label}: n={n}, events={events}", flush=True)
    rng = np.random.default_rng([SEED, run_id, 0])
    coefs, ses, covs, preds, risks, bands, grids, traces, fits, completed = [], [], [], [], [], [], [], [], [], []
    coefs_i, ses_i, covs_i = [], [], []
    for i in range(M_IMP):
        Xi, tr = impute_array(X, y, rng, limits, trace=True)
        traces.append(tr)
        completed.append(Xi)
        core = fit_logit(Xi[:, :N_CORE], y)
        inter = fit_logit(Xi, y)
        fits.append({"imputation": i + 1, "core_converged": core["converged"], "core_iterations": core["n_iter"], "core_finite": core["finite"],
                     "core_extreme_predictions": core["extreme"], "core_warnings": ";".join(core["warnings"]),
                     "interaction_converged": inter["converged"], "interaction_iterations": inter["n_iter"], "interaction_finite": inter["finite"],
                     "interaction_extreme_predictions": inter["extreme"], "interaction_warnings": ";".join(inter["warnings"])})
        coefs.append(core["params"]); ses.append(core["bse"]); covs.append(core["cov"]); preds.append(core["pred"])
        coefs_i.append(inter["params"]); ses_i.append(inter["bse"]); covs_i.append(inter["cov"])
        risks.append(standardized_risk(core["b0"], core["beta"], Xi))
        if full:
            bands.append(age_band_risk(inter["b0"], inter["beta"], Xi, age))
            grids.append(age_grid_risk(inter["b0"], inter["beta"], Xi))
    fits = pd.DataFrame(fits)
    names = ["intercept"] + CORE_COLS
    names_i = ["intercept"] + ALL_COLS
    OR, q, se, dof = ratio_table(coefs, ses, names)
    OR_i, _, _, _ = ratio_table(coefs_i, ses_i, names_i)
    d1_rate = d1_test(coefs, covs, [names.index(c) for c in RATE_COLS])
    d1_int = d1_test(coefs_i, covs_i, [names_i.index(c) for c in INTERACTION_COLS])
    risk_pt = np.mean(risks, 0)
    band_pt = np.mean(bands, 0) if full else None
    grid_pt = np.mean(grids, 0) if full else None

    # imputation convergence (20 chains x N_ITER iterations, final truncated values)
    conv = []
    for j, name in IMPUTED_IDX.items():
        if traces[0]["n_missing"].get(j, 0) == 0:
            continue
        chain_mean = [t["mean"][j] for t in traces]
        chain_sd = [t["sd"][j] for t in traces]
        miss = np.isnan(X[:, j])
        observed = X[~miss, j]
        final = np.concatenate([Xi[miss, j] for Xi in completed])
        conv.append({"variable": name, "n_missing": int(miss.sum()),
                     "observed_mean_sd": f"{observed.mean():.2f} ({observed.std():.2f})",
                     "imputed_mean_sd": f"{final.mean():.2f} ({final.std():.2f})",
                     "imputed_min_max": f"{final.min():.1f}-{final.max():.1f}",
                     "untruncated_draw_mean_sd_last_iter": f"{np.mean([t['mean_raw'][j][-1] for t in traces]):.2f} ({np.mean([t['sd_raw'][j][-1] for t in traces]):.2f})",
                     "rhat_mean": round(rhat(chain_mean), 3), "rhat_sd": round(rhat(chain_sd), 3),
                     "drift_second_half_sd_units": round(drift(chain_mean) / observed.std(), 4),
                     "truncated_last_iter_over_total": f"{int(sum(t['truncated'][j] for t in traces))} / {int(miss.sum()) * M_IMP}",
                     "limits": f"[{limits[j][0]:.1f}, {limits[j][1]:.1f}]"})
    conv = pd.DataFrame(conv)

    # apparent performance and influence (same data, no optimism correction)
    p_hat = np.mean(preds, axis=0)
    auc = float(stats.mannwhitneyu(p_hat[y == 1], p_hat[y == 0]).statistic / (events * (n - events)))
    brier = float(np.mean((p_hat - y) ** 2))
    lp = np.log(p_hat / (1 - p_hat))
    cal = sm.Logit(y, sm.add_constant(lp)).fit(disp=0)
    decile = pd.qcut(p_hat, 10, labels=False, duplicates="drop")
    calibration = (pd.DataFrame({"decile": decile, "expected": p_hat, "observed": y})
                   .groupby("decile").agg(n=("observed", "size"), expected=("expected", "mean"), observed=("observed", "mean")).reset_index())
    Xc = X0[CORE_COLS].dropna()
    vif = []
    for c in Xc.columns:
        others = Xc.drop(columns=[c]).values
        A = np.column_stack([np.ones(len(others)), others])
        fitted = A @ np.linalg.lstsq(A, Xc[c].values, rcond=None)[0]
        r2 = 1 - np.sum((Xc[c].values - fitted) ** 2) / np.sum((Xc[c].values - Xc[c].mean()) ** 2)
        vif.append((c, 1 / (1 - r2 + 1e-12)))
    glm = sm.GLM(y, sm.add_constant(completed[0][:, :N_CORE], has_constant="add"), family=sm.families.Binomial()).fit()
    infl = glm.get_influence()
    cook = infl.cooks_distance[0]
    dfbetas = infl.dfbetas
    top1 = cook >= np.quantile(cook, 0.99)
    glm_trim = sm.GLM(y[~top1], sm.add_constant(completed[0][~top1, :N_CORE], has_constant="add"), family=sm.families.Binomial()).fit()
    rate_j = [1 + CORE_COLS.index(c) for c in RATE_COLS]
    influence = pd.DataFrame(
        [{"measure": "largest Cook's distance", "value": round(float(cook.max()), 4)},
         {"measure": f"observations with Cook's D > 4/n (={4 / n:.4f})", "value": int((cook > 4 / n).sum())},
         {"measure": "observations with Cook's D > 0.5", "value": int((cook > 0.5).sum())},
         {"measure": f"observations with |DFBETAS| > 2/sqrt(n) (={2 / np.sqrt(n):.3f}), rate terms", "value": " / ".join(str(int((np.abs(dfbetas[:, j]) > 2 / np.sqrt(n)).sum())) for j in rate_j)},
         {"measure": "largest |DFBETAS|, rate terms", "value": round(float(np.abs(dfbetas[:, rate_j]).max()), 3)},
         {"measure": "largest |DFBETAS|, all terms", "value": round(float(np.abs(dfbetas).max()), 3)},
         {"measure": "observations removed (top 1% Cook's D)", "value": int(top1.sum())}]
        + [{"measure": f"{c} OR, first imputation: all / top 1% removed", "value": f"{np.exp(glm.params[j]):.2f} / {np.exp(glm_trim.params[j]):.2f}"} for c, j in zip(RATE_COLS, rate_j)])
    summary = {"n": n, "events": events, "events_per_df": round(events / N_CORE, 1), "auc_apparent": round(auc, 3), "brier_apparent": round(brier, 4),
               "calibration_slope": round(float(cal.params[1]), 3), "calibration_intercept": round(float(cal.params[0]), 3),
               "max_vif_excl_spline": round(max(v for c, v in vif if not c.startswith("age_s")), 2), "max_vif_all": round(max(v for _, v in vif), 2),
               "rubin_df_min": round(float(np.min(dof)), 1), "d1_rate": f"F={d1_rate[0]:.2f}, p={d1_rate[3]:.4f}", "d1_age_x_rate": f"F={d1_int[0]:.2f}, p={d1_int[3]:.4f}",
               "core_converged": f"{int(fits['core_converged'].sum())}/{M_IMP}", "interaction_converged": f"{int(fits['interaction_converged'].sum())}/{M_IMP}",
               "extreme_predictions": int(fits["core_extreme_predictions"].sum() + fits["interaction_extreme_predictions"].sum()),
               "fits_with_warnings": int((fits["core_warnings"] != "").sum() + (fits["interaction_warnings"] != "").sum()),
               "imputation_rhat_max": (round(float(conv[["rhat_mean", "rhat_sd"]].max().max()), 3) if len(conv) else np.nan)}

    # bootstrap
    t0 = time.time()
    print(f"   bootstrap B={B}, n_jobs={N_JOBS} ...", flush=True)
    reps = Parallel(n_jobs=N_JOBS, batch_size=8)(delayed(bootstrap_replicate)(bb, run_id, X, y, age, limits, full) for bb in range(B))
    print(f"   bootstrap done ({time.time() - t0:.0f} s)", flush=True)
    ok = [r for r in reps if r["status"] == "ok"]
    bad = [r for r in reps if r["status"] != "ok"]
    boot = {"analysis": label, "B_requested": B, "B_completed": len(ok), "B_failed": len(bad),
            "failed_too_few_events": sum(r["reason"] == "too few events" for r in bad), "failed_not_converged": sum(r["reason"] == "not converged" for r in bad),
            "failed_exception": sum(r["reason"].startswith("exception") for r in bad),
            "completed_with_warnings": sum(r["warnings"] > 0 for r in ok), "completed_with_extreme_predictions": sum(r["extreme"] > 0 for r in ok),
            "lbfgs_max_iterations": max((r["max_iter"] for r in ok), default=0),
            "original_core_converged": f"{int(fits['core_converged'].sum())}/{M_IMP}", "original_interaction_converged": f"{int(fits['interaction_converged'].sum())}/{M_IMP}",
            "original_finite": bool(fits["core_finite"].all() and fits["interaction_finite"].all()),
            "original_max_newton_iterations": int(max(fits["core_iterations"].max(), fits["interaction_iterations"].max()))}
    failed = pd.DataFrame([{"analysis": label, "replicate": r["bb"], "reason": r["reason"]} for r in bad])
    rv = np.array([r["risk"] for r in ok])
    lo, hi = percentile_ci(rv)
    lo_d, hi_d = percentile_ci(rv - rv[:, [REF_J]])
    risk = pd.DataFrame([{"rate_category": lvl, "risk_pct": 100 * risk_pt[j], "risk_lower": 100 * lo[j], "risk_upper": 100 * hi[j],
                          "rd_vs_ref_pct": 100 * (risk_pt[j] - risk_pt[REF_J]), "rd_lower": 100 * lo_d[j], "rd_upper": 100 * hi_d[j]}
                         for j, lvl in enumerate(RATE_LEVELS)])
    band = grid = None
    if full:
        bv = np.array([r["band"] for r in ok])
        rows = []
        for k, (name, lo_a, hi_a) in enumerate(AGE_BANDS):
            mask = (age >= lo_a) & (age < hi_a)
            for j, lvl in enumerate(RATE_LEVELS):
                l1, u1 = percentile_ci(bv[:, k, j])
                l2, u2 = percentile_ci(bv[:, k, j] - bv[:, k, REF_J])
                rows.append({"age_band": name, "n": int(mask.sum()), "deaths": int(y[mask].sum()), "rate_category": lvl,
                             "risk_pct": 100 * band_pt[k, j], "risk_lower": 100 * l1, "risk_upper": 100 * u1,
                             "rd_vs_ref_pct": 100 * (band_pt[k, j] - band_pt[k, REF_J]), "rd_lower": 100 * l2, "rd_upper": 100 * u2})
        band = pd.DataFrame(rows)
        gv = np.array([r["grid"] for r in ok])
        grid = pd.DataFrame({"age": AGE_GRID})
        for j, lvl in enumerate(RATE_LEVELS):
            grid[lvl] = 100 * grid_pt[:, j]
            grid[f"{lvl} lower"] = 100 * np.percentile(gv[:, :, j], 2.5, axis=0)
            grid[f"{lvl} upper"] = 100 * np.percentile(gv[:, :, j], 97.5, axis=0)
    summary["bootstrap_completed"] = f"{len(ok)}/{B}"
    return dict(OR=OR, OR_i=OR_i, risk=risk, band=band, grid=grid, summary=summary, calibration=calibration,
                vif=pd.DataFrame(vif, columns=["term", "VIF"]), influence=influence, convergence=conv, fits=fits,
                boot=boot, failed=failed, traces=traces)


def with_label(table, label):
    table = table.copy()
    table.insert(0, "analysis", label)
    return table


if __name__ == "__main__":
    df = data.load()
    first = data.first_visits(df)
    limits = limits_from(first)                     # fixed truncation range: first eligible visits
    cohort = first[first["landmark_alive"] == 1].copy()
    at_sample = first[first["alive_at_sample"] == 1].copy()
    runs = [("landmark (primary)", analyse(cohort, "landmark (primary)", 1, limits, full=True)),
            ("time of 24-hour sodium measurement", analyse(at_sample, "time of 24-hour sodium measurement", 2, limits, full=False)),
            ("ED presentation", analyse(first, "ED presentation", 3, limits, full=False))]
    primary = runs[0][1]

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "02_primary_model.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        pd.DataFrame([{"analysis": lab, **r["summary"]} for lab, r in runs]).to_excel(writer, sheet_name="summary", index=False)
        pd.concat([with_label(r["OR"], lab) for lab, r in runs]).to_excel(writer, sheet_name="table2_or", index=False)
        pd.concat([r["risk"].assign(analysis=lab) for lab, r in runs]).to_excel(writer, sheet_name="table2_risk", index=False)
        primary["band"].to_excel(writer, sheet_name="table3_age_bands", index=False)
        primary["grid"].to_excel(writer, sheet_name="figure3_age_grid", index=False)
        pd.concat([with_label(r["OR_i"], lab) for lab, r in runs]).to_excel(writer, sheet_name="interaction_model_or", index=False)
        pd.concat([with_label(r["convergence"], lab) for lab, r in runs]).to_excel(writer, sheet_name="imputation_convergence", index=False)
        pd.DataFrame([r["boot"] for _, r in runs]).to_excel(writer, sheet_name="fit_summary", index=False)
        pd.concat([with_label(r["fits"], lab) for lab, r in runs]).to_excel(writer, sheet_name="fit_original_data", index=False)
        pd.concat([r["failed"] for _, r in runs]).to_excel(writer, sheet_name="bootstrap_failed", index=False)
        primary["calibration"].to_excel(writer, sheet_name="calibration", index=False)
        primary["vif"].to_excel(writer, sheet_name="vif", index=False)
        primary["influence"].to_excel(writer, sheet_name="influence", index=False)

    traces = primary["traces"]
    cols = [j for j in IMPUTED_IDX if traces[0]["n_missing"].get(j, 0) > 0]
    fig, axes = plt.subplots(2, len(cols), figsize=(4 * len(cols), 6), squeeze=False)
    for k, j in enumerate(cols):
        for t in traces:
            axes[0, k].plot(range(1, N_ITER + 1), t["mean"][j], lw=0.7, alpha=0.6)
            axes[1, k].plot(range(1, N_ITER + 1), t["sd"][j], lw=0.7, alpha=0.6)
        axes[0, k].set_title(f"{IMPUTED_IDX[j]}: imputed mean")
        axes[1, k].set_title(f"{IMPUTED_IDX[j]}: imputed SD")
        axes[1, k].set_xlabel("iteration")
    fig.suptitle(f"Imputation traces, {M_IMP} chains x {N_ITER} iterations (landmark cohort)")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "fig_imputation_traces.png"), dpi=150)
    plt.close(fig)

    print("\nwritten:", out)
    for lab, r in runs:
        print("\n", lab, r["summary"])
        print(r["OR"][r["OR"]["term"].str.startswith("rate")].round(3).to_string(index=False))
        print(r["risk"].round(1).to_string(index=False))
