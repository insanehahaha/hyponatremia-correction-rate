# Sodium correction rate and outcomes in severe hyponatremia

Analysis code for the retrospective cohort study of adult emergency department patients with
serum sodium <125 mmol/L (2018-2024): 24-hour sodium correction rate, 30-day mortality,
neurological deterioration and secondary outcomes, with an age-specific analysis.

The code reproduces every table and figure of the manuscript from the analysis-ready dataset.
The dataset contains patient-level clinical data and is not distributed; see
`docs/data_dictionary.csv` for the variables the scripts expect.

## Layout

```
hyponatremia/            shared modules
  config.py              paths, exposure categories, spline knots, seeds, m, B
  data.py                loading and variable names
  design.py              design matrix (rate categories, age spline, covariates, interactions)
  imputation.py          multiple imputation by fully conditional specification
  pooling.py             Rubin's rules, D1 test, pooled contrasts
  models.py              logistic and Cox fitting, robust covariance, cumulative incidence
  ordinal.py             Brant test components, partial proportional odds model
  descriptive.py         characteristics tables
  tables.py              Word table formatting
scripts/
  01_descriptive.py      study flow, Table 1, Supplementary Table S1
  02_primary_model.py    primary logistic model, standardized risks, bootstrap (Tables 2-3, Figure 3A)
  03_neurological.py     Fine-Gray and cause-specific models (Tables 3-4, Figure 3B)
  04_secondary.py        90-day/1-year Cox, 7-30-day mortality, DAOH-30, readmission, FDR (Table 4, S2)
  05_sensitivity.py      sensitivity analyses, subgroups, spline dose-response, discharge model (S3, Figure 2)
  06_tables.py           Word tables
  07_figures.py          Figures 1-3
tests/
  compare_numbers.py     compare the numbers in two Excel sheets
  compare_docx.py        compare the table cells of two Word documents
docs/
  data_dictionary.csv    variables used by the scripts
```

## Running

```
pip install -r requirements.txt
set HYPONATREMIA_DATA=path\to\analysis_dataset.parquet     # Windows
export HYPONATREMIA_DATA=path/to/analysis_dataset.parquet  # Linux, macOS
python scripts/01_descriptive.py
python scripts/02_primary_model.py [B] [n_jobs]
python scripts/03_neurological.py [B] [n_jobs]
python scripts/04_secondary.py
python scripts/05_sensitivity.py
python scripts/06_tables.py
python scripts/07_figures.py
```

Results are written to `results/` (Excel workbooks, `results/tables/*.docx`,
`results/figures/*`). Scripts 02 and 03 take an optional bootstrap size `B` (default 2000) and
number of parallel workers; the full bootstrap takes about 10-15 minutes per script on a
desktop computer. Script 04 depends on the output of script 03 (Benjamini-Hochberg family);
scripts 06 and 07 depend on 01-05.

Supplementary Table S4 (components of neurological deterioration) needs
`results/neuro_components.xlsx`, the event-level coding produced during chart review; the table
is skipped when the file is absent.

## Methods in brief

- Exposure: 24-hour correction rate in four categories (<4, 4-8, >8-10, >10 mmol/L per 24 h),
  reference 4-8. Primary cohort: first eligible visit per patient, alive at the 24-hour landmark.
- Primary model: logistic regression for 30-day mortality with 17 degrees of freedom (rate
  categories, age as a restricted cubic spline with 4 knots, sex, index sodium, prior sodium
  status, Charlson index, eGFR, albumin, glucose, potassium, diuretic use, calendar year);
  age x rate interaction model for the age-specific estimates.
- Missing covariates (albumin, glucose, potassium, eGFR): Bayesian normal fully conditional
  specification, m = 20, 50 iterations, draws truncated to the observed range; Rubin's rules
  and the D1 multivariate Wald test for pooling.
- Standardized risks and risk differences by model-based standardization; confidence intervals
  from a patient-level bootstrap (B = 2000) in which the imputation is repeated in every
  resample.
- Neurological deterioration: Fine-Gray subdistribution hazards fitted as a weighted Cox model
  with death and event-free discharge as competing events, robust covariance, 14-day
  standardized cumulative incidence.
- Secondary outcomes: Cox models with delayed entry, restricted mean survival time from
  pseudo-values, proportional and partial proportional odds models for days alive and out of
  hospital, logistic models for 7-30-day mortality and readmission, Benjamini-Hochberg
  adjustment across the six secondary tests.

Random seeds are fixed in `hyponatremia/config.py`; bootstrap replicates are seeded as
`(seed, analysis, replicate)` so that results are reproducible regardless of the number of
workers.

## Software

Python 3.12 with the package versions in `requirements.txt` (numpy, pandas, scipy,
statsmodels, lifelines, scikit-learn, joblib, matplotlib, openpyxl, python-docx).

## License

MIT (see `LICENSE`).
