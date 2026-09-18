"""Neurological deterioration within 14 days (Table 3 neurological panel, Figure 3B).

Risk set: landmark cohort patients alive, in hospital and event-free at 48 hours. Target
event: first physician-documented neurological deterioration; competing events: death and
event-free discharge; administrative censoring at day 14. Fine-Gray subdistribution
hazards are fitted as a weighted Cox model (Geskus weights; equal to 1 here because
censoring occurs only at the horizon), with cause-specific Cox models for comparison.
Standardized 14-day cumulative incidences use patient-level bootstrap intervals.

Model: rate category (3 df) + age spline (3 df) + index sodium + ICU in first 24 h = 8 df.

Input : analysis dataset
Output: results/03_neurological.xlsx
Usage : python scripts/03_neurological.py [B=2000] [n_jobs]
"""
import os
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyponatremia import data
from hyponatremia.config import AGE_BANDS, AGE_CENTER, AGE_GRID, AGE_SCALE, B_BOOT, NEURO_HORIZON, RATE_LEVELS, RATE_REF, RESULTS, SEED_NEURO
from hyponatremia.design import RATE_COLS, rcs
from hyponatremia.models import aalen_johansen, fit_cox, hazard_table, sandwich, wald
from hyponatremia.pooling import percentile_ci

B = int(sys.argv[1]) if len(sys.argv) > 1 else B_BOOT
N_JOBS = int(sys.argv[2]) if len(sys.argv) > 2 else max(1, (os.cpu_count() or 2) - 2)
SEED = SEED_NEURO
REF_J = RATE_LEVELS.index(RATE_REF)
EVENT = {"neurological": 1, "death": 2, "discharge": 3, "censored": 0}
CORE = RATE_COLS + ["age", "age_s1", "age_s2", "sodium_index", "icu_24h"]
INTERACTION = [f"{c}_x_age" for c in RATE_COLS]
FULL = CORE + INTERACTION


# ---------- design
def design(df):
    X = pd.DataFrame(index=df.index)
    for lvl in RATE_LEVELS:
        if lvl != RATE_REF:
            X[f"rate_{lvl}"] = (df["rate_category"].astype(str) == lvl).astype(float)
    spline = rcs(df["age"].values)
    X["age"], X["age_s1"], X["age_s2"] = spline[:, 0], spline[:, 1], spline[:, 2]
    X["sodium_index"] = df["sodium_index"].astype(float)
    X["icu_24h"] = df["icu_24h"].astype(float)
    X = with_interactions(X)
    X["T"] = df["neuro_event_day"].astype(float).values
    X["type"] = df["neuro_event_type"].map(EVENT).astype(int).values
    return X


def with_interactions(X):
    for c in RATE_COLS:
        X[f"{c}_x_age"] = X[c] * (X["age"] - AGE_CENTER) / AGE_SCALE
    return X


def set_rate(X, level):
    Xn = X.copy()
    Xn[RATE_COLS] = 0.0
    if level != RATE_REF:
        Xn[f"rate_{level}"] = 1.0
    return with_interactions(Xn)


def set_age(X, age):
    Xn = X.copy()
    spline = rcs(np.full(len(Xn), float(age)))
    Xn["age"], Xn["age_s1"], Xn["age_s2"] = spline[:, 0], spline[:, 1], spline[:, 2]
    return with_interactions(Xn)


# ---------- competing-risk data layouts
def fine_gray_data(X, target=1):
    """Geskus weighting: censoring survival G(t) equals 1 before the horizon, so weights are 1
    and patients with a competing event stay in the risk set until the horizon."""
    censored = (X["type"] == 0).astype(int)
    km = KaplanMeierFitter().fit(X["T"], event_observed=censored)
    g_before = float(km.survival_function_at_times(NEURO_HORIZON - 1e-9).iloc[0])
    assert abs(g_before - 1.0) < 1e-12, "censoring before the horizon: general Geskus weights required"
    D = X.copy()
    D["E"] = (D["type"] == target).astype(int)
    D["w"] = 1.0
    competing = (D["type"] != target) & (D["type"] != 0)
    D.loc[competing, "T"] = NEURO_HORIZON
    return D


def cause_specific_data(X, target=1):
    D = X.copy()
    D["E"] = (D["type"] == target).astype(int)
    D["w"] = 1.0
    return D


# ---------- standardized cumulative incidence
def cif(cph, X, cols):
    H = cph.predict_cumulative_hazard(X[cols], times=[NEURO_HORIZON]).iloc[0].values
    return 1 - np.exp(-H)


def standardized_cif(cph, X, cols):
    return np.array([cif(cph, set_rate(X, lvl), cols).mean() for lvl in RATE_LEVELS])


def age_band_cif(cph, X, cols):
    return np.array([[cif(cph, set_rate(X[(X["age"] >= lo) & (X["age"] < hi)], lvl), cols).mean() for lvl in RATE_LEVELS]
                     for _, lo, hi in AGE_BANDS])


def age_grid_cif(cph, X, cols):
    out = np.zeros((len(AGE_GRID), len(RATE_LEVELS)))
    for i, a in enumerate(AGE_GRID):
        Xa = set_age(X, a)
        for j, lvl in enumerate(RATE_LEVELS):
            out[i, j] = cif(cph, set_rate(Xa, lvl), cols).mean()
    return out


# ---------- bootstrap
def bootstrap_replicate(bb, X, full):
    rng = np.random.default_rng([SEED, 1, bb + 1])
    Xb = X.iloc[rng.integers(0, len(X), len(X))].reset_index(drop=True)
    out = dict(bb=bb, status="ok", reason="", warnings=0)
    if (Xb["type"] == 1).sum() < 5:
        out.update(status="failed", reason="too few events")
        return out
    try:
        D = fine_gray_data(Xb)
        c1, ok1, w1 = fit_cox(D, CORE, robust=False)
        c2, ok2, w2 = fit_cox(D, FULL, robust=False)
        out["warnings"] = len(w1) + len(w2)
        if not (ok1 and ok2):
            out.update(status="failed", reason="not converged")
            return out
        if not (np.isfinite(c1.params_).all() and np.isfinite(c2.params_).all()):
            out.update(status="failed", reason="non-finite coefficient")
            return out
        out["risk"] = standardized_cif(c1, Xb, CORE)
        if full:
            out["band"] = age_band_cif(c1, Xb, CORE)
            out["band_i"] = age_band_cif(c2, Xb, FULL)
            out["grid"] = age_grid_cif(c1, Xb, CORE)
            out["grid_i"] = age_grid_cif(c2, Xb, FULL)
    except Exception as err:
        out.update(status="failed", reason=f"exception: {type(err).__name__}: {err}"[:150])
    return out


def aj_row(label, X, mask=None):
    Xs = X if mask is None else X[mask]
    return {"group": label, "n": len(Xs), "neurological_events": int((Xs["type"] == 1).sum()),
            "cif14_neurological_pct": 100 * aalen_johansen(Xs["T"], Xs["type"], 1),
            "cif14_death_pct": 100 * aalen_johansen(Xs["T"], Xs["type"], 2),
            "cif14_discharge_pct": 100 * aalen_johansen(Xs["T"], Xs["type"], 3)}


if __name__ == "__main__":
    df = data.load()
    cohort = data.landmark_cohort(df)
    R = cohort[cohort["neuro_risk_set"] == 1].copy()
    covariates = ["age", "sodium_index", "icu_24h", "rate_category", "neuro_event_day", "neuro_event_type"]
    assert R[covariates].notna().all().all(), "missing covariate in the neurological model"
    X = design(R).reset_index(drop=True)
    n = len(X)
    n_events = int((X["type"] == 1).sum())
    counts = X["type"].value_counts()
    print(f"risk set n={n}; neurological {n_events}, death {counts.get(2, 0)}, discharge {counts.get(3, 0)}, censored {counts.get(0, 0)}")

    # denominators and non-parametric cumulative incidence
    R["age_band"] = data.age_band(R["age"])
    rate_str = R["rate_category"].astype(str)
    by_rate = pd.crosstab(rate_str, R["neuro_event_type"], margins=True, margins_name="total").reindex(RATE_LEVELS + ["total"])
    by_age = pd.crosstab(R["age_band"], R["neuro_event_type"], margins=True, margins_name="total")
    aj = pd.DataFrame([aj_row("all", X)]
                      + [aj_row(f"rate {lvl}", X, (rate_str == lvl).values) for lvl in RATE_LEVELS]
                      + [aj_row(f"age {name}", X, ((X["age"] >= lo) & (X["age"] < hi)).values) for name, lo, hi in AGE_BANDS])

    # models on the original data
    DF = fine_gray_data(X)
    DC = cause_specific_data(X)
    fg, ok_fg, w_fg = fit_cox(DF, CORE)
    fgi, ok_fgi, w_fgi = fit_cox(DF, FULL)
    cs, ok_cs, w_cs = fit_cox(DC, CORE)
    csi, ok_csi, w_csi = fit_cox(DC, FULL)
    cs_death, _, _ = fit_cox(cause_specific_data(X, 2), CORE)
    cs_discharge, _, _ = fit_cox(cause_specific_data(X, 3), CORE)
    HR = pd.concat([hazard_table(fg, "Fine-Gray sHR, neurological deterioration"),
                    hazard_table(cs, "cause-specific HR, neurological deterioration"),
                    hazard_table(cs_death, "cause-specific HR, death"),
                    hazard_table(cs_discharge, "cause-specific HR, event-free discharge"),
                    hazard_table(fgi, "Fine-Gray sHR with age x rate interaction (secondary)"),
                    hazard_table(csi, "cause-specific HR with age x rate interaction (secondary)")], ignore_index=True)
    V = {"fg": sandwich(fg, DF, CORE), "fgi": sandwich(fgi, DF, FULL), "cs": sandwich(cs, DC, CORE), "csi": sandwich(csi, DC, FULL)}
    joint = pd.DataFrame([{"model": m, "test": t, "wald_chi2_robust": w[0], "df": w[1], "p": w[2], "wald_chi2_model_based": wn[0], "p_model_based": wn[2]}
                          for m, t, w, wn in [("Fine-Gray", "rate (3 df)", wald(fg, RATE_COLS, V["fg"]), wald(fg, RATE_COLS)),
                                              ("Fine-Gray interaction", "age x rate (3 df)", wald(fgi, INTERACTION, V["fgi"]), wald(fgi, INTERACTION)),
                                              ("cause-specific Cox", "rate (3 df)", wald(cs, RATE_COLS, V["cs"]), wald(cs, RATE_COLS)),
                                              ("cause-specific Cox interaction", "age x rate (3 df)", wald(csi, INTERACTION, V["csi"]), wald(csi, INTERACTION))]])
    ph_fg = proportional_hazard_test(fg, DF[CORE + ["T", "E", "w"]], time_transform="rank").summary.reset_index().rename(columns={"index": "term"}).assign(model="Fine-Gray")
    ph_cs = proportional_hazard_test(cs, DC[CORE + ["T", "E", "w"]], time_transform="rank").summary.reset_index().rename(columns={"index": "term"}).assign(model="cause-specific Cox")
    PH = pd.concat([ph_fg, ph_cs], ignore_index=True)
    risk_pt = standardized_cif(fg, X, CORE)
    band_pt, band_pt_i = age_band_cif(fg, X, CORE), age_band_cif(fgi, X, FULL)
    grid_pt, grid_pt_i = age_grid_cif(fg, X, CORE), age_grid_cif(fgi, X, FULL)

    # sensitivity analyses of the event definition (Fine-Gray, original data)
    uncertain = (R["neuro_risk_set_uncertain"] == 0).values
    in_death_interval = (R["neuro_event_in_death_interval"] == 1).values
    rows = []
    for label, Xs in [(f"main (risk set n={n})", X),
                      (f"uncertain 48-hour status excluded (n={int(uncertain.sum())})", X[uncertain]),
                      ("events within the death interval counted as death", X.assign(type=np.where(in_death_interval & (X["type"] == 1), 2, X["type"])))]:
        c_, _, _ = fit_cox(fine_gray_data(Xs), CORE)
        for term in RATE_COLS:
            s = c_.summary.loc[term]
            rows.append({"sensitivity": label, "n": len(Xs), "neurological_events": int((Xs["type"] == 1).sum()), "term": term,
                         "sHR": s["exp(coef)"], "lower": s["exp(coef) lower 95%"], "upper": s["exp(coef) upper 95%"]})
    sensitivity = pd.DataFrame(rows)

    # bootstrap
    t0 = time.time()
    print(f"bootstrap B={B}, n_jobs={N_JOBS} ...", flush=True)
    reps = Parallel(n_jobs=N_JOBS, batch_size=8)(delayed(bootstrap_replicate)(bb, X, True) for bb in range(B))
    print(f"bootstrap done ({time.time() - t0:.0f} s)", flush=True)
    ok = [r for r in reps if r["status"] == "ok"]
    bad = [r for r in reps if r["status"] != "ok"]
    rv = np.array([r["risk"] for r in ok])
    lo, hi = percentile_ci(rv)
    lo_d, hi_d = percentile_ci(rv - rv[:, [REF_J]])
    risk = pd.DataFrame([{"rate_category": lvl, "cif14_pct": 100 * risk_pt[j], "cif_lower": 100 * lo[j], "cif_upper": 100 * hi[j],
                          "rd_vs_ref_pct": 100 * (risk_pt[j] - risk_pt[REF_J]), "rd_lower": 100 * lo_d[j], "rd_upper": 100 * hi_d[j]}
                         for j, lvl in enumerate(RATE_LEVELS)])
    rows = []
    for label, key, pt in [("Fine-Gray (8 df)", "band", band_pt), ("Fine-Gray with interaction (11 df; secondary)", "band_i", band_pt_i)]:
        bv = np.array([r[key] for r in ok])
        for k, (name, lo_a, hi_a) in enumerate(AGE_BANDS):
            mask = (X["age"] >= lo_a) & (X["age"] < hi_a)
            for j, lvl in enumerate(RATE_LEVELS):
                l1, u1 = percentile_ci(bv[:, k, j])
                l2, u2 = percentile_ci(bv[:, k, j] - bv[:, k, REF_J])
                rows.append({"model": label, "age_band": name, "n": int(mask.sum()), "neurological_events": int((mask & (X["type"] == 1)).sum()),
                             "rate_category": lvl, "cif14_pct": 100 * pt[k, j], "cif_lower": 100 * l1, "cif_upper": 100 * u1,
                             "rd_vs_ref_pct": 100 * (pt[k, j] - pt[k, REF_J]), "rd_lower": 100 * l2, "rd_upper": 100 * u2})
    band = pd.DataFrame(rows)

    def grid_frame(key, pt):
        gv = np.array([r[key] for r in ok])
        G = pd.DataFrame({"age": AGE_GRID})
        for j, lvl in enumerate(RATE_LEVELS):
            G[lvl] = 100 * pt[:, j]
            G[f"{lvl} lower"] = 100 * np.percentile(gv[:, :, j], 2.5, axis=0)
            G[f"{lvl} upper"] = 100 * np.percentile(gv[:, :, j], 97.5, axis=0)
        return G

    grid, grid_i = grid_frame("grid", grid_pt), grid_frame("grid_i", grid_pt_i)

    diagnostics = pd.DataFrame([
        ["risk set n / neurological / death / discharge / censored", f"{n} / {n_events} / {counts.get(2, 0)} / {counts.get(3, 0)} / {counts.get(0, 0)}"],
        ["events per df (8 df; 11 df)", f"{n_events / 8:.1f} / {n_events / 11:.1f}"],
        ["Fine-Gray weights", "Geskus IPCW; censoring only at day 14, G(t) = 1, all weights 1 (verified)"],
        ["converged (FG / FG interaction / CS / CS interaction)", f"{ok_fg} / {ok_fgi} / {ok_cs} / {ok_csi}; warnings: {';'.join(w_fg + w_fgi + w_cs + w_csi) or 'none'}"],
        ["bootstrap completed / requested", f"{len(ok)} / {B}; failed: too few events {sum(r['reason'] == 'too few events' for r in bad)}, "
                                            f"not converged {sum(r['reason'] == 'not converged' for r in bad)}, exception {sum(r['reason'].startswith('exception') for r in bad)}; "
                                            f"completed with warnings {sum(r['warnings'] > 0 for r in ok)}"],
        ["Schoenfeld test smallest p (FG / CS)", f"{ph_fg['p'].min():.3f} / {ph_cs['p'].min():.3f}"],
        ["robust SE / model-based SE, rate terms (FG)", " / ".join(f"{a:.3f}/{b:.3f}" for a, b in zip(fg.standard_errors_[RATE_COLS].values, np.sqrt(np.diag(fg.variance_matrix_.loc[RATE_COLS, RATE_COLS].values))))],
    ], columns=["measure", "value"])

    os.makedirs(RESULTS, exist_ok=True)
    out = os.path.join(RESULTS, "03_neurological.xlsx")
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        by_rate.to_excel(writer, sheet_name="events_by_rate")
        by_age.to_excel(writer, sheet_name="events_by_age")
        aj.to_excel(writer, sheet_name="aalen_johansen_cif14", index=False)
        HR.to_excel(writer, sheet_name="hazard_ratios", index=False)
        joint.to_excel(writer, sheet_name="joint_tests", index=False)
        risk.to_excel(writer, sheet_name="table3_cif14", index=False)
        band.to_excel(writer, sheet_name="age_band_cif14", index=False)
        grid.to_excel(writer, sheet_name="figure3_age_grid", index=False)
        grid_i.to_excel(writer, sheet_name="figure3_age_grid_interaction", index=False)
        PH.to_excel(writer, sheet_name="proportional_hazards", index=False)
        diagnostics.to_excel(writer, sheet_name="diagnostics", index=False)
        sensitivity.to_excel(writer, sheet_name="sensitivity", index=False)
        pd.DataFrame([{"replicate": r["bb"], "reason": r["reason"]} for r in bad]).to_excel(writer, sheet_name="bootstrap_failed", index=False)

    print("\nwritten:", out)
    print(by_rate.to_string())
    print(HR[HR["term"].str.startswith("rate")].round(3).to_string(index=False))
    print(joint.round(3).to_string(index=False))
    print(risk.round(1).to_string(index=False))
    print(diagnostics.to_string(index=False))
