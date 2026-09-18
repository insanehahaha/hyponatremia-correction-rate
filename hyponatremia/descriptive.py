"""Descriptive statistics for the characteristics tables."""
import numpy as np
import pandas as pd

ROWS = [
    ("Baseline characteristics (at index sodium measurement)", None, "h"),
    ("Age, years", "age", "c"),
    ("Age ≥65 years", "age_65_plus", "b"),
    ("Female sex", "female", "b"),
    ("Index serum sodium, mmol/L", "sodium_index", "c"),
    ("Glucose-corrected serum sodium, mmol/L", "sodium_index_corrected", "c"),
    ("Serum sodium <115 mmol/L", "sodium_below_115", "b"),
    ("Calculated serum osmolality, mOsm/kg", "osmolality", "c"),
    ("Prior sodium status: low", "prior_low", "b"),
    ("Prior sodium status: unknown", "prior_unknown", "b"),
    ("Charlson comorbidity index", "charlson", "c"),
    ("Heart failure", "heart_failure", "b"),
    ("Chronic kidney disease", "ckd", "b"),
    ("Malignancy", "malignancy", "b"),
    ("Dementia", "dementia", "b"),
    ("Diabetes mellitus", "diabetes", "b"),
    ("Hypertension", "hypertension", "b"),
    ("Thiazide diuretic before presentation", "thiazide", "b"),
    ("Loop diuretic before presentation", "loop_diuretic", "b"),
    ("SSRI or SNRI before presentation", "ssri_snri", "b"),
    ("Proton pump inhibitor before presentation", "ppi", "b"),
    ("Potassium, mmol/L", "potassium", "c"),
    ("Creatinine, mg/dL", "creatinine", "c"),
    ("eGFR, mL/min/1.73 m²", "egfr", "c"),
    ("Albumin, g/L", "albumin", "c"),
    ("Glucose, mg/dL", "glucose", "c"),
    ("C-reactive protein, mg/L", "crp", "c"),
    ("Care and treatment (index care episode)", None, "h"),
    ("Admitted to ward or intensive care unit", "admitted", "b"),
    ("Intensive care unit admission within 24 h", "icu_24h", "b"),
    ("Hypertonic (3%) saline", "hypertonic_saline", "b"),
    ("Follow-up at 24 h", None, "h"),
    ("24-hour serum sodium, mmol/L", "sodium_24h", "c"),
    ("Correction rate, mmol/L/24 h", "rate_24h", "c"),
    ("Outcome", None, "h"),
    ("30-day mortality", "death_30d", "b"),
]


def median_iqr(s):
    s = s.dropna()
    if len(s) == 0:
        return "—"
    return f"{s.median():.1f} ({s.quantile(0.25):.1f}–{s.quantile(0.75):.1f})"


def n_percent(s):
    s = s.dropna()
    if len(s) == 0:
        return "—"
    n = int((s == 1).sum())
    return f"{n} ({100 * n / len(s):.1f})"


def smd_continuous(a, b):
    a, b = a.dropna(), b.dropna()
    pooled = np.sqrt((a.var() + b.var()) / 2)
    return abs(a.mean() - b.mean()) / pooled if pooled > 0 else 0.0


def smd_binary(a, b):
    a, b = a.dropna(), b.dropna()
    p1, p2 = a.mean(), b.mean()
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return abs(p1 - p2) / pooled if pooled > 0 else 0.0


def add_indicators(df):
    df = df.copy()
    df["female"] = (df["sex"] == "female").astype(int)
    df["sodium_below_115"] = (df["sodium_index"] < 115).astype(int)
    df["prior_low"] = (df["prior_sodium"] == "low").astype(int)
    df["prior_unknown"] = (df["prior_sodium"] == "unknown").astype(int)
    return df


def characteristics_table(df, group_col, groups, reference=None):
    """Characteristics by group: median (IQR) or n (%), missing count and largest SMD versus the reference group."""
    df = add_indicators(df)
    header = ["Characteristic", f"All (n={len(df)})"]
    header += [f"{g} (n={int((df[group_col] == g).sum())})" for g in groups]
    header += ["Missing, n"] + ([f"Largest SMD vs {reference}"] if reference else [])
    rows = []
    for label, col, kind in ROWS:
        if kind == "h":
            rows.append([label] + [""] * (len(header) - 1))
            continue
        if col not in df.columns:
            continue
        summarise = median_iqr if kind == "c" else n_percent
        row = [label, summarise(df[col])]
        row += [summarise(df.loc[df[group_col] == g, col]) for g in groups]
        row.append(int(df[col].isna().sum()))
        if reference:
            smd = smd_continuous if kind == "c" else smd_binary
            ref = df.loc[df[group_col] == reference, col]
            row.append(f"{max(smd(df.loc[df[group_col] == g, col], ref) for g in groups if g != reference):.2f}")
        rows.append(row)
    return pd.DataFrame(rows, columns=header)
