"""Paths and fixed analysis settings shared by all scripts."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.environ.get("HYPONATREMIA_DATA", os.path.join(ROOT, "data", "analysis_dataset.parquet"))
RESULTS = os.environ.get("HYPONATREMIA_RESULTS", os.path.join(ROOT, "results"))

# exposure
RATE_LEVELS = ["<4", "4-8", ">8-10", ">10"]          # mmol/L per 24 h
RATE_REF = "4-8"

# age
AGE_KNOTS = [41.0, 64.1, 74.4, 86.8]                  # restricted cubic spline knots (years)
AGE_BANDS = [("<65", 0, 65), ("65-74", 65, 75), ("75-84", 75, 85), (">=85", 85, 200)]
AGE_GRID = [30 + 2.5 * i for i in range(27)]          # 30, 32.5, ..., 95
AGE_CENTER, AGE_SCALE = 70.0, 10.0                    # linear age x rate interaction: (age - 70) / 10

# multiple imputation and bootstrap
M_IMP = 20
N_ITER = 50
B_BOOT = 2000
SEED_PRIMARY = 20260909
SEED_NEURO = 20260910
SEED_SECONDARY = 20260911
SEED_SENSITIVITY = 20260912

# follow-up horizons (days)
NEURO_HORIZON = 14.0
LANDMARK_DAYS = 1.5
