"""Study flow counts and characteristics tables (Table 1, Supplementary Table S1).

Input : analysis dataset (see hyponatremia/data.py)
Output: results/01_descriptive.xlsx
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hyponatremia import data
from hyponatremia.config import RESULTS, RATE_LEVELS, RATE_REF
from hyponatremia.descriptive import characteristics_table

df = data.load()
eligible = df[df["eligible"] == 1]
first = data.first_visits(df)
cohort = data.landmark_cohort(df)
cohort["age_band"] = data.age_band(cohort["age"])

table1 = characteristics_table(cohort, "age_band", ["<65", "65-74", "75-84", ">=85"], reference="<65")
table_s1 = characteristics_table(cohort, "rate_category", RATE_LEVELS, reference=RATE_REF)

excluded = df["exclusion_reason"].value_counts()
flow = pd.DataFrame([
    ("Adult ED visits with serum sodium <125 mmol/L", int((df["exclusion_reason"] != "yaş <18").sum())),
    ("Excluded: sodium <125 mmol/L not confirmed", int(excluded.get("indeks Na<125 biyokimya ile doğrulanamadı", 0))),
    ("Excluded: laboratory artefact", int(excluded.get("indeks Na artefakt (kural seti D-2b)", 0))),
    ("Excluded: glucose-corrected sodium >=125 mmol/L", int(excluded.get("glukoz düzeltmesi sonrası Na ≥125 (uygunluk dışı)", 0))),
    ("Eligible visits", len(eligible)),
    ("No 24-hour sodium measurement", int((eligible["rate_available"] == 0).sum())),
    ("Visits with a calculable correction rate", int(eligible["rate_available"].sum())),
    ("First eligible visit per patient", len(first)),
    ("Death before the landmark", int((first["landmark_alive"] == 0).sum())),
    ("Study population (landmark cohort)", len(cohort)),
    ("  age >=65 years", int(cohort["age_65_plus"].sum())),
    ("  30-day deaths", int(cohort["death_30d"].sum())),
    ("  neurological analysis risk set (48 h)", int((cohort["neuro_risk_set"] == 1).sum())),
], columns=["Step", "n"])

os.makedirs(RESULTS, exist_ok=True)
out = os.path.join(RESULTS, "01_descriptive.xlsx")
with pd.ExcelWriter(out, engine="openpyxl") as writer:
    flow.to_excel(writer, sheet_name="flow", index=False)
    table1.to_excel(writer, sheet_name="table1_age", index=False)
    table_s1.to_excel(writer, sheet_name="tableS1_rate", index=False)
print("written:", out)
print(flow.to_string(index=False))
