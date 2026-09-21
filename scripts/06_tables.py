"""Manuscript tables (Word) from the analysis results.

Main tables (results/tables/Tables_1-4.docx):
  Table 1  characteristics by age group
  Table 2  correction rate and 30-day mortality by time origin
  Table 3  age-specific standardized risks from the interaction models
  Table 4  secondary outcomes
Supplementary tables (results/tables/Supplementary_Tables_S1-S4.docx):
  S1 characteristics by correction-rate category; S2 mortality between days 7 and 30 and
  secondary outcomes by age group; S3 sensitivity, subgroup and additional analyses;
  S4 components of neurological deterioration (requires results/neuro_components.xlsx, the
  event-level component coding from chart review, which is not distributed; S4 is skipped
  when the file is absent).

Input : analysis dataset, results/01-05 workbooks
Output: results/tables/*.docx
"""
import os
import re
import sys

import pandas as pd
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyponatremia import data
from hyponatremia.config import RATE_LEVELS, RESULTS
from hyponatremia.descriptive import median_iqr, n_percent
from hyponatremia.tables import DASH, EMPTY, add_table, ci, ci_to, is_empty, new_document, section_row

OUT = os.path.join(RESULTS, "tables")
os.makedirs(OUT, exist_ok=True)
R1 = pd.read_excel(os.path.join(RESULTS, "01_descriptive.xlsx"), sheet_name=None)
R2 = pd.read_excel(os.path.join(RESULTS, "02_primary_model.xlsx"), sheet_name=None)
R3 = pd.read_excel(os.path.join(RESULTS, "03_neurological.xlsx"), sheet_name=None)
R4 = pd.read_excel(os.path.join(RESULTS, "04_secondary.xlsx"), sheet_name=None)
R5 = pd.read_excel(os.path.join(RESULTS, "05_sensitivity.xlsx"), sheet_name=None)
L = data.landmark_cohort(data.load())
L["age_band"] = data.age_band(L["age"])

RATE = {"<4": "<4", "4-8": f"4{DASH}8 (reference)", ">8-10": ">8 to 10", ">10": ">10"}          # display labels
RATE_SHORT = {k: v.replace(" (reference)", "") for k, v in RATE.items()}
REF = "4-8"
BANDS = ["<65", "65-74", "75-84", ">=85"]
BAND = {"<65": "<65 years", "65-74": f"65{DASH}74 years", "75-84": f"75{DASH}84 years", ">=85": "≥85 years"}


def test_p(s):
    """'F=0.18, p=0.9077' -> 'p = 0.91'."""
    m = re.match(r"F=([\d.]+), p=([\d.]+)", str(s))
    return f"p = {float(m.group(2)):.2f}" if m else str(s)


def simplify(df, label_col, missing_col):
    """Drop the osmolality row and turn the missing-value column into footnote text."""
    df = df[df[label_col] != "Calculated serum osmolality, mOsm/kg"].copy()
    parts = []
    for _, r in df.iterrows():
        try:
            v = int(float(r[missing_col]))
        except (TypeError, ValueError):
            v = 0
        name = str(r[label_col])
        if v > 0 and not name.startswith(("24-hour", "Correction rate", "NRS-2002", "Glucose-corrected")):
            label = name.split(",")[0]
            label = label if label in ("eGFR", "C-reactive protein") else label[0].lower() + label[1:]
            parts.append(f"{label} {v}")
    df = df.drop(columns=[missing_col])
    return df, ("Missing values: " + ", ".join(parts) + ". ") if parts else ""


# ======================================================================= main tables
main = new_document()

# ---- Table 1: characteristics by age group
S1 = R1["table1_age"].copy()
cols1 = list(S1.columns)


def extra_row(label, f, col):
    r = {cols1[0]: label, cols1[1]: f(L[col])}
    for b, c in zip(BANDS, cols1[2:6]):
        r[c] = f(L.loc[L["age_band"] == b, col])
    r[cols1[6]] = int(L[col].isna().sum())
    r[cols1[7]] = ""
    return r


L["n_medications_5"] = (L["n_medications"] >= 5).astype(int)
L["nrs3"] = (L["nrs2002"] >= 3).astype(float).where(L["nrs2002"].notna())
extra_baseline = [extra_row("Active substances on medication reports, n", median_iqr, "n_medications"), extra_row("≥5 active substances", n_percent, "n_medications_5")]
extra_care = [extra_row("NRS-2002 score", median_iqr, "nrs2002"), extra_row("NRS-2002 score ≥3", n_percent, "nrs3")]
rate_rows = []
for k in RATE_LEVELS:
    r = {cols1[0]: f"Correction rate {RATE_SHORT[k]} mmol/L/24 h", cols1[1]: n_percent((L["rate_category"].astype(str) == k).astype(int))}
    for b, c in zip(BANDS, cols1[2:6]):
        r[c] = n_percent((L.loc[L["age_band"] == b, "rate_category"].astype(str) == k).astype(int))
    r[cols1[6]] = 0
    r[cols1[7]] = ""
    rate_rows.append(r)
rows = []
for _, r in S1.iterrows():
    rows.append(r.to_dict())
    if r[cols1[0]] == "Proton pump inhibitor before presentation":
        rows += extra_baseline
    if r[cols1[0]] == "Hypertonic (3%) saline":
        rows += extra_care
    if r[cols1[0]] == "Correction rate, mmol/L/24 h":
        rows += rate_rows
T1 = pd.DataFrame(rows)[cols1]
T1[cols1[6]] = T1[cols1[6]].apply(lambda v: "" if is_empty(v) else str(int(float(v))))
for c in cols1[1:6]:
    T1[c] = T1[c].apply(lambda v: "" if is_empty(v) else str(v))
n_band = {b: int((L["age_band"] == b).sum()) for b in BANDS}
T1.columns = ["Characteristic", f"All patients (n = {len(L)})"] + [f"{BAND[b]} (n = {n_band[b]})" for b in BANDS] + ["Missing, n", "Largest SMD"]
T1 = T1[~T1["Characteristic"].isin(["Glucose-corrected serum sodium, mmol/L", "Creatinine, mg/dL", "Active substances on medication reports, n", "NRS-2002 score"])].copy()
T1, missing1 = simplify(T1, "Characteristic", "Missing, n")
SHORT = {"Thiazide diuretic before presentation": "Thiazide diuretic", "Loop diuretic before presentation": "Loop diuretic", "SSRI or SNRI before presentation": "SSRI or SNRI",
         "Proton pump inhibitor before presentation": "Proton pump inhibitor", "Admitted to ward or intensive care unit": "Admitted to ward or ICU",
         "Intensive care unit admission within 24 h": "ICU admission within 24 h"}
SHORT.update({f"Correction rate {RATE_SHORT[k]} mmol/L/24 h": f"  {RATE_SHORT[k]}" for k in RATE_LEVELS})
T1["Characteristic"] = T1["Characteristic"].replace(SHORT)
n_nrs = int(L["nrs2002"].notna().sum())
add_table(main, T1, "Table 1. Characteristics of the study population by age group", widths=[1.85, 0.82, 0.78, 0.78, 0.78, 0.78, 0.55],
          footnote="Values are median (interquartile range) or n (%). " + missing1 + "Medications: prescriptions within 90 days or active medication reports within 1 year before presentation. "
    f"NRS-2002 recorded in {n_nrs} patients. "
    "Hypertonic saline: any 3% sodium chloride order within 48 h of the index measurement. To convert glucose to mmol/L, multiply by 0.0555. eGFR, estimated glomerular filtration rate; ICU, intensive care unit; NRS-2002, Nutritional Risk Screening 2002; SMD, largest absolute standardized mean difference versus the <65-year group; SNRI, serotonin–norepinephrine reuptake inhibitor; SSRI, selective serotonin reuptake inhibitor.",
          section=section_row)

# ---- Table 2: correction rate and 30-day mortality by time origin
orr = R2["table2_or"]
risk = R2["table2_risk"]
summary = R2["summary"].set_index("analysis")
ORIGINS = ["landmark (primary)", "time of 24-hour sodium measurement", "ED presentation"]
ORIGIN = {"landmark (primary)": "24-hour landmark (primary)", "time of 24-hour sodium measurement": "Time of 24-hour sodium measurement", "ED presentation": "Time of ED presentation"}
rows = []
for origin in ORIGINS:
    o = orr[orr["analysis"] == origin].set_index("term")
    r = risk[risk["analysis"] == origin].set_index("rate_category")
    n, ev = int(summary.loc[origin, "n"]), int(summary.loc[origin, "events"])
    name = ORIGIN[origin].replace(" (primary)", ", primary analysis")
    rows.append({"Correction rate, mmol/L/24 h": f"Time origin: {name} ({n} patients, {ev} deaths)"})
    for k in RATE_LEVELS:
        rr = r.loc[k]
        rows.append({"Correction rate, mmol/L/24 h": RATE[k],
                     "Adjusted OR (95% CI)": "1 (reference)" if k == REF else ci(o.loc[f"rate_{k}", "OR"], o.loc[f"rate_{k}", "lower"], o.loc[f"rate_{k}", "upper"]),
                     "Standardized 30-day mortality risk, % (95% CI)": f"{rr['risk_pct']:.1f} ({rr['risk_lower']:.1f}{DASH}{rr['risk_upper']:.1f})",
                     "Risk difference, percentage points (95% CI)": "0 (reference)" if k == REF else ci_to(rr["rd_vs_ref_pct"], rr["rd_lower"], rr["rd_upper"])})
p_rate = {o: test_p(summary.loc[o, "d1_rate"]) for o in ORIGINS}
p_int = {o: test_p(summary.loc[o, "d1_age_x_rate"]).replace("p = ", "") for o in ORIGINS}
JOINT = (f"24-hour landmark, correction rate {p_rate[ORIGINS[0]]}, age × correction rate p = {p_int[ORIGINS[0]]}; "
         f"time of 24-hour measurement, {p_rate[ORIGINS[1]]} and {p_int[ORIGINS[1]]}; "
         f"time of presentation, {p_rate[ORIGINS[2]]} and {p_int[ORIGINS[2]]}")
add_table(main, pd.DataFrame(rows), "Table 2. Sodium correction rate and 30-day mortality by time origin", widths=[1.5, 1.4, 1.75, 1.65],
          footnote="Adjusted odds ratios and standardized risks from multivariable logistic regression; covariates are listed in the Methods. Joint tests: " + JOINT + ". "
    "CI, confidence interval; ED, emergency department; OR, odds ratio.",
          section=section_row)

# ---- Table 3: age-specific risks from the interaction models
bands = R2["table3_age_bands"]
nb = R3["age_band_cif14"]
ntests = R3["joint_tests"]
p_fg_int = float(ntests[ntests["model"] == "Fine-Gray interaction"]["p"].iloc[0])
top = RATE_LEVELS[-1]
COLS3 = ["Age group", "n / events"] + [RATE[k] for k in RATE_LEVELS] + [f">10 vs 4{DASH}8, risk difference, percentage points (95% CI)"]
rows = [{"Age group": "A. 30-day mortality: standardized risk, % (95% CI)"}]
for b in bands["age_band"].unique():
    x = bands[bands["age_band"] == b].set_index("rate_category")
    row = {"Age group": BAND[b], "n / events": f"{int(x['n'].iloc[0])} / {int(x['deaths'].iloc[0])}"}
    for k in RATE_LEVELS:
        row[RATE[k]] = f"{x.loc[k, 'risk_pct']:.1f} ({x.loc[k, 'risk_lower']:.1f}{DASH}{x.loc[k, 'risk_upper']:.1f})"
    row[COLS3[-1]] = ci_to(x.loc[top, "rd_vs_ref_pct"], x.loc[top, "rd_lower"], x.loc[top, "rd_upper"])
    rows.append(row)
rows.append({"Age group": "B. Neurological deterioration up to day 14: standardized cumulative incidence, % (95% CI)"})
nbi = nb[nb["model"] == "Fine-Gray with interaction (11 df; secondary)"]
for b in nbi["age_band"].unique():
    x = nbi[nbi["age_band"] == b].set_index("rate_category")
    row = {"Age group": BAND[b], "n / events": f"{int(x['n'].iloc[0])} / {int(x['neurological_events'].iloc[0])}"}
    for k in RATE_LEVELS:
        row[RATE[k]] = f"{x.loc[k, 'cif14_pct']:.1f} ({x.loc[k, 'cif_lower']:.1f}{DASH}{x.loc[k, 'cif_upper']:.1f})"
    row[COLS3[-1]] = ci_to(x.loc[top, "rd_vs_ref_pct"], x.loc[top, "rd_lower"], x.loc[top, "rd_upper"])
    rows.append(row)
add_table(main, pd.DataFrame(rows)[COLS3], "Table 3. Age-specific adjusted risks of 30-day mortality and neurological deterioration by correction rate",
          widths=[1.0, 0.7, 0.92, 0.92, 0.92, 0.92, 0.95],
          footnote=f"Correction-rate categories in mmol/L/24 h. Standardized risks within each age group; confidence intervals from patient-level bootstrap. "
    f"Joint tests for age × correction-rate interaction: mortality {test_p(summary.loc['landmark (primary)', 'd1_age_x_rate'])}, neurological deterioration p = {p_fg_int:.2f}. CI, confidence interval.",
          section=section_row)

# ---- Table 4: secondary outcomes
observed = R4["observed_by_rate"]
tests4 = R4["joint_tests"]


def p_of(analysis):
    return float(tests4[(tests4["analysis"] == analysis) & (~tests4["test"].str.contains("age x rate"))].iloc[0]["p"])


aj = R3["aalen_johansen_cif14"].set_index("group")
hr3 = R3["hazard_ratios"]
fg = hr3[hr3["model"] == "Fine-Gray sHR, neurological deterioration"].set_index("term")
cif = R3["table3_cif14"].set_index("rate_category")
p_neuro = float(ntests[(ntests["model"] == "Fine-Gray") & (ntests["test"].str.startswith("rate"))]["p"].iloc[0])
ppo = R4["daoh_ppo_constrained"].set_index("term")
ppo_summary = R4["daoh_ppo_summary"]
p_ppo = float(str(ppo_summary[ppo_summary["form"] == "constrained"].iloc[0]["rate_joint_D1"]).split("p=")[1])
A90, A365, A7, AREADM = "death 90 d (Cox)", "death 1 y (Cox)", "death 7-30 d (7-day landmark)", "readmission 30 d"
family = {A90: p_of(A90), A365: p_of(A365), A7: p_of(A7), "DAOH": p_ppo, AREADM: p_of(AREADM), "neuro": p_neuro}
p_bh = dict(zip(family, multipletests(list(family.values()), method="fdr_bh")[1]))
C4 = ["Outcome / correction rate, mmol/L/24 h", "n / events", "Observed", "Effect estimate (95% CI)"]
rows = [{C4[0]: f"Neurological deterioration, 48 h to day 14 (n = {int(aj.loc['all', 'n'])}); joint test p = {p_neuro:.2f}, BH-adjusted p = {p_bh['neuro']:.2f}"}]
for k in RATE_LEVELS:
    a = aj.loc[f"rate {k}"]
    r = cif.loc[k]
    std = f"{r['cif14_pct']:.1f} ({r['cif_lower']:.1f}{DASH}{r['cif_upper']:.1f})"
    eff = "1 (reference)" if k == REF else "sHR " + ci(fg.loc[f"rate_{k}", "HR"], fg.loc[f"rate_{k}", "lower"], fg.loc[f"rate_{k}", "upper"]) + "; RD " + ci_to(r["rd_vs_ref_pct"], r["rd_lower"], r["rd_upper"])
    rows.append({C4[0]: RATE[k], C4[1]: f"{int(a['n'])} / {int(a['neurological_events'])}", C4[2]: f"{a['cif14_neurological_pct']:.1f}%; standardized {std}", C4[3]: eff})
hr4 = R4["cox_90d_1y"]
for analysis, title in [(A90, "Mortality at 90 days (n = 1282)"), (A365, "Mortality at 1 year (n = 1282)")]:
    h = hr4[hr4["analysis"] == analysis].set_index("term")
    t = observed[observed["outcome"] == analysis].set_index("group")
    rows.append({C4[0]: f"{title}; joint test p = {p_of(analysis):.2f}, BH-adjusted p = {p_bh[analysis]:.2f}"})
    for k in RATE_LEVELS:
        x = t.loc[k]
        rows.append({C4[0]: RATE[k], C4[1]: f"{int(x['n'])} / {int(x['events'])}", C4[2]: f"{x['observed_risk_pct']:.1f}%",
                     C4[3]: "1 (reference)" if k == REF else "HR " + ci(h.loc[f"rate_{k}", "HR"], h.loc[f"rate_{k}", "lower"], h.loc[f"rate_{k}", "upper"])})
t = observed[observed["outcome"] == "DAOH-30"].set_index("group")
rows.append({C4[0]: f"DAOH-30 (n = 1282); joint test p = {p_ppo:.3f}, BH-adjusted p = {p_bh['DAOH']:.2f}"})
for k in RATE_LEVELS:
    x = t.loc[k]
    rows.append({C4[0]: RATE[k], C4[1]: f"{int(x['n'])} / {int(x['events'])}", C4[2]: f"median {x['daoh_median_iqr'].replace('-', DASH)} days",
                 C4[3]: "1 (reference)" if k == REF else "OR " + ci(ppo.loc[f"rate_{k}", "OR"], ppo.loc[f"rate_{k}", "lower"], ppo.loc[f"rate_{k}", "upper"])})
readm = R4["readmission"].set_index("term")
t = observed[observed["outcome"] == AREADM].set_index("group")
rows.append({C4[0]: f"ED revisit or readmission within 30 days (n = {int(t['n'].sum())}); joint test p = {p_of(AREADM):.2f}, BH-adjusted p = {p_bh[AREADM]:.2f}"})
for k in RATE_LEVELS:
    x = t.loc[k]
    rows.append({C4[0]: RATE[k], C4[1]: f"{int(x['n'])} / {int(x['events'])}", C4[2]: f"{x['observed_risk_pct']:.1f}%",
                 C4[3]: "1 (reference)" if k == REF else "OR " + ci(readm.loc[f"rate_{k}", "OR"], readm.loc[f"rate_{k}", "lower"], readm.loc[f"rate_{k}", "upper"])})
add_table(main, pd.DataFrame(rows)[C4], "Table 4. Sodium correction rate and secondary outcomes", widths=[1.9, 0.85, 1.75, 1.8],
          footnote="Reference category 4–8 mmol/L/24 h. Observed: Aalen–Johansen 14-day cumulative incidence (neurological deterioration), Kaplan–Meier estimate (mortality), "
    "median (interquartile range) days (DAOH-30), proportion (readmission); for DAOH-30, n / events counts patients with DAOH-30 = 0. "
    "DAOH-30 scored in whole days, 0 for death within 30 days; OR >1 indicates more days alive and out of hospital. "
    "BH-adjusted p: Benjamini–Hochberg adjustment across the six secondary tests (this table and Supplementary Table S2). "
    "Proportional hazards assessment, restricted mean survival time and cause-specific hazard ratios in Supplementary Table S3. "
    "BH, Benjamini–Hochberg; CI, confidence interval; DAOH-30, days alive and out of hospital within 30 days; ED, emergency department; HR, hazard ratio; OR, odds ratio; RD, risk difference; sHR, subdistribution hazard ratio.",
          section=section_row)
main_path = os.path.join(OUT, "Tables_1-4.docx")
main.save(main_path)
print("written:", main_path)


# ======================================================================= supplementary tables
supp = new_document()

# ---- S1: characteristics by correction-rate category
t1 = R1["tableS1_rate"].copy()
n_rate = {k: int((L["rate_category"].astype(str) == k).sum()) for k in RATE_LEVELS}
t1.columns = ["Characteristic", f"All patients (n = {len(L)})"] + [f"{RATE_SHORT[k]} (n = {n_rate[k]})" for k in RATE_LEVELS] + ["Missing, n", f"Largest SMD vs 4{DASH}8"]
t1["Missing, n"] = t1["Missing, n"].apply(lambda v: "" if is_empty(v) else str(int(float(v))))
t1, missing2 = simplify(t1, "Characteristic", "Missing, n")
add_table(supp, t1, "Supplementary Table S1. Characteristics of the study population by correction-rate category",
          "Values are median (interquartile range) or n (%). " + missing2 + "Medications: prescriptions within 90 days or active medication reports within 1 year before presentation. Hypertonic saline: any 3% sodium chloride order within 48 h of the index measurement. "
    "To convert glucose to mmol/L, multiply by 0.0555; to convert creatinine to µmol/L, multiply by 88.4. eGFR, estimated glomerular filtration rate; SMD, largest absolute standardized mean difference versus the 4–8 mmol/L/24 h group; SNRI, serotonin–norepinephrine reuptake inhibitor; SSRI, selective serotonin reuptake inhibitor.",
          section=section_row)

# ---- S2: mortality between days 7 and 30; secondary outcomes by age group
C5 = ["Outcome / correction rate, mmol/L/24 h", "n / events", "Observed", "Effect estimate (95% CI)", "Standardized 14-day cumulative incidence, % (95% CI); risk difference"]
o7 = R4["death_7_30d"].set_index("term").loc["rapid_correction_7d"]
t7 = observed[observed["outcome"] == A7].set_index("group")
rows = [{C5[0]: f"Mortality between day 7 and day 30 (7-day landmark cohort, n = {int(t7['n'].sum())}; exposure: rapid correction within the first 7 days, reference \"no\"; logistic odds ratio) — p = {p_of(A7):.3f}; BH-adjusted p = {p_bh[A7]:.2f}"}]
G7 = {"no rapid correction days 0-7 (reference)": "No rapid correction, days 0–7 (reference)", "rapid correction days 0-7": "Rapid correction, days 0–7"}
for g in t7.index:
    x = t7.loc[g]
    rows.append({C5[0]: G7[g], C5[1]: f"{int(x['n'])} / {int(x['events'])}", C5[2]: f"{x['observed_risk_pct']:.1f}%",
                 C5[3]: "1 (reference)" if "no rapid" in g else "OR " + ci(o7["OR"], o7["lower"], o7["upper"]), C5[4]: EMPTY})
by_age = R4["observed_by_age"]
rows.append({C5[0]: "Secondary outcomes by age group (descriptive): 90-day mortality n (%) / 1-year mortality n (%) / DAOH-30 median (IQR) / readmission n/denominator (%)"})
for _, r in by_age.iterrows():
    rows.append({C5[0]: BAND[r["age_band"]], C5[1]: f"{int(r['n'])}", C5[2]: f"{r['death_90d_n_pct']} / {r['death_1y_n_pct']}",
                 C5[3]: f"DAOH-30 {r['daoh_median_iqr'].replace('-', DASH)}", C5[4]: f"readmission {r['readmission_n_denominator_pct']}"})
add_table(supp, pd.DataFrame(rows)[C5], "Supplementary Table S2. Mortality between days 7 and 30 and secondary outcomes by age group",
          "Mortality between days 7 and 30: 77 patients whose death was certain before day 7 were excluded; 11 whose death interval included day 7 were retained. "
    "Exposure is rapid correction during the first 7 days as defined in the Methods. Age-group panel: descriptive only; readmission denominator is patients discharged alive. "
    "BH-adjusted p as in Table 4. BH, Benjamini–Hochberg; CI, confidence interval; DAOH-30, days alive and out of hospital within 30 days; IQR, interquartile range; OR, odds ratio.",
          section=section_row)

# ---- S3: sensitivity, subgroup and additional analyses
sens = R5["sensitivity_summary"]
tests5 = R5["joint_tests"].set_index("analysis")
fits5 = R5["fit_summary"].set_index("analysis")
NAME = {"primary model structure (landmark cohort, n=1282)": "Primary model structure (for comparison; no bootstrap)",
        "all eligible episodes (patient-clustered SE), n=1391, patients=1282": "All eligible episodes, each ≥30 days after the previous included episode and after its discharge (1391 episodes in 1282 patients; standard errors clustered by patient)",
        "eligibility on measured sodium (no glucose correction), n=1626": "Eligibility based on measured sodium (no glucose correction)",
        "Hillier glucose correction (2.4), n=1195": "Glucose correction with the Hillier factor (2.4)",
        "binary threshold: rate >8 vs <=8 mmol/L/24 h (n=1282)": "Binary threshold >8 vs ≤8 mmol/L/24 h",
        "binary threshold: rate >10 vs <=10 mmol/L/24 h (n=1282)": "Binary threshold >10 vs ≤10 mmol/L/24 h",
        "48-hour rate >18 vs <=18 mmol/L/48 h, 60-hour landmark, n=980 (6 certain deaths before 60 h excluded)": "48-h correction >18 vs ≤18 mmol/L/48 h (60-hour landmark; 6 patients with certain death before 60 h excluded)",
        "24-hour sodium measured at 18-30 h only, n=728": "24-hour sodium measured at 18–30 h only",
        "raw change in sodium 0-24 h, linear, OR per 1 mmol/L (n=1282)": "Raw change in sodium over 0–24 h (linear; OR per 1 mmol/L)",
        "death interval crossing the landmark excluded, n=1264": "Exclusion of 18 patients whose death interval included the 24-h landmark",
        "complete case (no imputation)": "Complete-case analysis (no imputation)",
        "confirmed hospital admission only, n=1225": "Patients with confirmed hospital admission only"}
CONTRAST = {"rate_<4": "<4", "rate_>8-10": ">8 to 10", "rate_>10": ">10", "rate>8": ">8", "rate>10": ">10", "rate48>18": ">18 mmol/L/48 h", "delta_sodium_24h": "per 1 mmol/L"}
C6 = ["Analysis / correction rate", "n / events", "Effect estimate (95% CI)", "Joint test p"]
rows = [{C6[0]: f"A. Sensitivity analyses for the primary outcome (30-day mortality) — adjusted OR; contrasts are <4, >8 to 10 and >10 vs 4{DASH}8 mmol/L/24 h unless a binary or continuous exposure is stated"}]
for a in sens["analysis"].unique():
    x = sens[sens["analysis"] == a]
    u = fits5.loc[a]
    p = tests5.loc[a, "p"]
    p = p if not isinstance(p, pd.Series) else p.iloc[0]
    est = "; ".join(f"{CONTRAST.get(r['exposure_contrast'], r['exposure_contrast'])}: {r['OR']:.2f} ({r['lower']:.2f}{DASH}{r['upper']:.2f})" for _, r in x.iterrows())
    rows.append({C6[0]: NAME[a], C6[1]: f"{int(u['n'])} / {int(u['events'])}", C6[2]: est, C6[3]: f"{p:.2f}"})
sub = R5["subgroup_or"]
sub_test = R5["subgroup_interaction_bh"]
ODS = "osmotic demyelination high-risk stratum (0 = no risk factor, 1 = at least one)"
SUB = {(ODS, "0"): "High-risk stratum for osmotic demyelination: no risk factor", (ODS, "1"): "High-risk stratum for osmotic demyelination: ≥1 risk factor",
       ("prior sodium status", "normal"): "Prior sodium status: normal", ("prior sodium status", "low"): "Prior sodium status: low", ("prior sodium status", "unknown"): "Prior sodium status: unknown"}
rows.append({C6[0]: f"B. Prespecified subgroups: adjusted OR for >10 vs 4{DASH}8 mmol/L/24 h (contrast from a single model with subgroup × correction-rate interaction; Rubin t intervals)"})
or_col = [c for c in sub.columns if c.startswith("OR ")][0]
LEVEL_ORDER = ["0", "1", "normal", "low", "unknown"]
sub = sub.iloc[sub["level"].astype(str).map(LEVEL_ORDER.index).argsort(kind="stable")]
for _, r in sub.iterrows():
    e = sub_test[sub_test["subgroup"] == r["subgroup"]].iloc[0]
    rows.append({C6[0]: SUB[(r["subgroup"], str(r["level"]))], C6[1]: f"{int(r['n'])} / {int(r['deaths'])}", C6[2]: ci(r[or_col], r["lower"], r["upper"]),
                 C6[3]: f"interaction p = {e['p']:.2f}; BH-adjusted p = {e['p_bh']:.2f}"})
rows.append({C6[0]: "Patients receiving haemodialysis", C6[1]: EMPTY, C6[2]: "not estimable: no validated haemodialysis variable in the locked dataset", C6[3]: EMPTY})
rmst = R4["rmst"]
med = R4["daoh_median"].set_index("term")
fgd = R5["discharge_fine_gray"]
fgd_name = fgd["analysis"].iloc[0]
fgd = fgd.set_index("term")
u_fgd = fits5.loc[fgd_name]
p_fgd = float(tests5.loc[fgd_name, "p"])
cs = hr3[hr3["model"] == "cause-specific HR, neurological deterioration"].set_index("term")
cs_death = hr3[hr3["model"] == "cause-specific HR, death"].set_index("term")
cs_disch = hr3[hr3["model"] == "cause-specific HR, event-free discharge"].set_index("term")
p_cs = float(ntests[(ntests["model"] == "cause-specific Cox") & (ntests["test"].str.startswith("rate"))]["p"].iloc[0])


def three(f):
    return "; ".join(f"{CONTRAST[c]}: {f(c)}" for c in ["rate_<4", "rate_>8-10", "rate_>10"])


rows.append({C6[0]: f"C. Additional survival and distribution analyses (reference 4{DASH}8 mmol/L/24 h)"})
for analysis, title in [("death 90 d, RMST days 1.5-90", "Restricted mean survival time difference, days (95% CI), 24 h to day 90"),
                        ("death 1 y, RMST days 1.5-365", "Restricted mean survival time difference, days (95% CI), 24 h to day 365")]:
    x = rmst[rmst["analysis"] == analysis].set_index("term")
    rows.append({C6[0]: title, C6[1]: f"1282 / {EMPTY}", C6[2]: three(lambda c: ci_to(x.loc[c, "rmst_difference_days"], x.loc[c, "lower"], x.loc[c, "upper"])), C6[3]: EMPTY})
rows.append({C6[0]: "DAOH-30 median regression: difference in medians, days (95% CI)", C6[1]: f"1282 / {EMPTY}",
             C6[2]: three(lambda c: ci_to(med.loc[c, "median_difference_days"], med.loc[c, "lower"], med.loc[c, "upper"])), C6[3]: EMPTY})
rows.append({C6[0]: "Discharge alive from index care, Fine–Gray sHR (95% CI); patients alive and not yet discharged at 24 h", C6[1]: f"{int(u_fgd['n'])} / {int(u_fgd['events'])}",
             C6[2]: three(lambda c: ci(fgd.loc[c, "sHR"], fgd.loc[c, "lower"], fgd.loc[c, "upper"])), C6[3]: f"{p_fgd:.2f}"})
rows.append({C6[0]: "Neurological deterioration, cause-specific HR (95% CI), 48-hour risk set", C6[1]: "1139 / 79", C6[2]: three(lambda c: ci(cs.loc[c, "HR"], cs.loc[c, "lower"], cs.loc[c, "upper"])), C6[3]: f"{p_cs:.2f}"})
rows.append({C6[0]: "Competing events, cause-specific HR (95% CI): death", C6[1]: "1139 / 96", C6[2]: three(lambda c: ci(cs_death.loc[c, "HR"], cs_death.loc[c, "lower"], cs_death.loc[c, "upper"])), C6[3]: EMPTY})
rows.append({C6[0]: "Competing events, cause-specific HR (95% CI): discharge without event", C6[1]: "1139 / 731", C6[2]: three(lambda c: ci(cs_disch.loc[c, "HR"], cs_disch.loc[c, "lower"], cs_disch.loc[c, "upper"])), C6[3]: EMPTY})
add_table(supp, pd.DataFrame(rows)[C6], "Supplementary Table S3. Sensitivity, subgroup and additional analyses",
          "A: primary model structure; for the 48-h correction rate the risk set is patients alive at 60 h, because the measurement falls at 36–60 h. "
    "B: contrasts from a single model with subgroup × correction-rate interaction; interaction p values adjusted within their own family of two tests. "
    "C: restricted mean survival time from Kaplan–Meier pseudo-values regressed on the covariates, same imputations as the Cox models; "
    "discharge alive analyzed with death during index care as competing event, censoring at day 30, time from 24 h; cause-specific models censor competing events at their own times. "
    "BH, Benjamini–Hochberg; CI, confidence interval; DAOH-30, days alive and out of hospital within 30 days; HR, hazard ratio; OR, odds ratio; sHR, subdistribution hazard ratio.",
          section=section_row)

# ---- S4: components of neurological deterioration (event-level coding from chart review)
components_path = os.path.join(RESULTS, "neuro_components.xlsx")
if os.path.exists(components_path):
    NB = pd.read_excel(components_path)
    n4 = {k: int((NB["rate_category"] == k).sum()) for k in RATE_LEVELS}
    C7 = ["Component or context"] + [f"{RATE_SHORT[k]} (n = {n4[k]})" for k in RATE_LEVELS] + [f"All events (n = {len(NB)})"]

    def s4row(label, mask):
        r = {C7[0]: label}
        for k, c in zip(RATE_LEVELS, C7[1:5]):
            m = mask & (NB["rate_category"] == k)
            r[c] = f"{int(m.sum())} ({100 * m.sum() / n4[k]:.0f})"
        r[C7[5]] = f"{int(mask.sum())} ({100 * mask.sum() / len(NB):.0f})"
        return r

    rows = [{C7[0]: "Component of deterioration (an event may have more than one)"},
            s4row("Decreased level of consciousness (GCS fall ≥2 points, stupor or coma)", NB["decreased_consciousness"] == 1),
            s4row("New confusion or delirium", NB["confusion_delirium"] == 1),
            s4row("Seizure", NB["seizure"] == 1),
            s4row("New focal neurological deficit", NB["focal_deficit"] == 1),
            s4row("New movement disorder", NB["movement_disorder"] == 1),
            {C7[0]: "Severity according to the European guideline symptom classification"},
            s4row("Severe (coma, GCS ≤8 or seizure)", NB["severity"] == "severe"),
            s4row("Moderately severe (confusion or somnolence)", NB["severity"] == "moderately severe"),
            s4row("Focal or movement findings only", NB["severity"] == "unclassified"),
            {C7[0]: "Context"},
            s4row("Structural brain lesion on imaging", NB["structural_lesion_imaging"] == 1),
            s4row("Infection, hypoxia or shock documented at the event", NB["infection_hypoxia_shock"] == 1),
            s4row("Known epilepsy", NB["known_epilepsy"] == 1),
            s4row("Brain imaging between day 2 and day 14", NB["imaging_day2_14"] == 1),
            s4row("GCS series only, no narrative note", NB["gcs_series_only"] == 1),
            s4row("Death within 30 days", NB["death_30d"] == 1)]
    add_table(supp, pd.DataFrame(rows)[C7], "Supplementary Table S4. Components of neurological deterioration by correction-rate category",
              footnote="Values are n (% of events in the column); an event may contribute to more than one component. "
    "Severity follows the European guideline symptom classification: severe, coma (GCS ≤8), seizure or stupor; moderately severe, confusion or somnolence. GCS, Glasgow Coma Scale.",
              section=section_row)
else:
    print("results/neuro_components.xlsx not found: Supplementary Table S4 skipped")
supp_path = os.path.join(OUT, "Supplementary_Tables_S1-S4.docx")
supp.save(supp_path)
print("written:", supp_path)
print({k: round(v, 3) for k, v in p_bh.items()})
