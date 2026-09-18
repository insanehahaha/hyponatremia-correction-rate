"""Load the analysis-ready dataset and expose it with English variable names.

The dataset is not distributed with the code. `docs/data_dictionary.csv` lists every
variable used here. Column names below map the original file to the names used in the
analysis code; values of the categorical variables are relabelled in the same way.
"""
import pandas as pd

from .config import DATA_PATH, RATE_LEVELS

COLUMNS = {
    "kayit_no": "record_id",
    "hasta_anahtari": "patient_id",
    "basvuru_yili": "year",
    "yas": "age",
    "yasli": "age_65_plus",
    "cinsiyet": "sex",
    "na_indeks": "sodium_index",
    "na_indeks_glukoz_duzeltilmis": "sodium_index_corrected",
    "ozmolalite_hesaplanan": "osmolality",
    "onceki_na_durumu": "prior_sodium",
    "charlson_skoru": "charlson",
    "km_kalp_yetmezligi": "heart_failure",
    "km_bobrek": "ckd",
    "km_malignite": "malignancy",
    "km_demans": "dementia",
    "km_diyabet": "diabetes",
    "hipertansiyon": "hypertension",
    "tiazid": "thiazide",
    "loop_diuretik": "loop_diuretic",
    "ssri_snri": "ssri_snri",
    "ppi": "ppi",
    "diuretik_herhangi": "any_diuretic",
    "polifarmasi": "n_medications",
    "nrs_2002": "nrs2002",
    "potasyum_indeks": "potassium",
    "kreatinin_indeks": "creatinine",
    "egfr_indeks": "egfr",
    "albumin_indeks": "albumin",
    "glukoz_indeks_eslesmis": "glucose",
    "crp_indeks": "crp",
    "yatis": "admitted",
    "ybu_bazal_24sa": "icu_24h",
    "hipertonik_salin": "hypertonic_saline",
    "hipertonik_salin_ilk_saat": "hypertonic_saline_first_hour",
    "na_24sa": "sodium_24h",
    "na_24_olcum_saati": "sodium_24h_time",
    "delta_na_24": "delta_sodium_24h",
    "hiz_24": "rate_24h",
    "hiz_24_kategori": "rate_category",
    "hiz_48": "rate_48h",
    "na_48_olcum_saati": "sodium_48h_time",
    "olum": "died",
    "mortalite_30gun": "death_30d",
    "hastane_ici_olum": "in_hospital_death",
    "sagkalim_gun_alt": "survival_days_lower",
    "sagkalim_gun_ust": "survival_days_upper",
    "sagkalim_gun_analiz36": "survival_days",
    "taburcu_gun": "discharge_day",
    "taburcu_kaynak": "discharge_source",
    "tekrar_basvuru_30gun": "readmission_30d",
    "daoh30": "daoh30",
    "hizli_duzeltme_0_7g": "rapid_correction_7d",
    "na_maks_artis_24sa_0_7g": "max_rise_24h_7d",
    "na_maks_artis_48sa_0_7g": "max_rise_48h_7d",
    "na_120_140_5g": "rise_120_to_140_5d",
    "ods_yuksek_risk": "ods_high_risk",
    "ods_derece": "ods_grade",
    "noro48_uygun": "neuro_risk_set",
    "noro48_belirsiz": "neuro_risk_set_uncertain",
    "noro_olay_turu": "neuro_event_type",
    "noro_olay_zaman_gun": "neuro_event_day",
    "noro_olay_olum_araliginda": "neuro_event_in_death_interval",
    "noro_hedef_olay": "neuro_event",
    "dislama_nedeni": "exclusion_reason",
    "kohort_dahil": "eligible",
    "k1_kohort": "rate_available",
    "birincil_analiz": "first_visit",
    "landmark36_uygun": "landmark_alive",
    "landmark36_belirsiz": "landmark_uncertain",
    "landmark_ornek_uygun": "alive_at_sample",
    "atak_duyarlilik": "episode_sensitivity",
    "olculen_na_kohortu": "measured_sodium_cohort",
    "hillier_kohortu": "hillier_cohort",
}

VALUES = {
    "sex": {"Kadın": "female", "Erkek": "male"},
    "prior_sodium": {"normal": "normal", "düşük": "low", "bilinmiyor": "unknown"},
    "rate_category": {"<4 yavaş": "<4", "4-8 hedefte": "4-8", ">8-10 sınırda": ">8-10", ">10 aşırı": ">10"},
    "neuro_event_type": {"nörolojik kötüleşme": "neurological", "ölüm": "death",
                         "olaysız taburculuk": "discharge", "sansür 14 g": "censored"},
}


def load(path=DATA_PATH):
    """Return the full visit-level dataset with English names."""
    df = pd.read_parquet(path)
    keep = [c for c in COLUMNS if c in df.columns]
    df = df[keep].rename(columns=COLUMNS)
    for col, mapping in VALUES.items():
        if col in df.columns:
            df[col] = df[col].map(lambda v: mapping.get(v, v))
    df["rate_category"] = pd.Categorical(df["rate_category"].astype(str), categories=RATE_LEVELS)
    return df


def landmark_cohort(df):
    """First eligible visit per patient, alive at the landmark (primary analysis cohort)."""
    return df[(df["first_visit"] == 1) & (df["landmark_alive"] == 1)].copy()


def first_visits(df):
    """First eligible visit per patient with a calculable correction rate."""
    return df[df["first_visit"] == 1].copy()


def age_band(age):
    """Age band labels used in the tables."""
    return pd.cut(age, [0, 65, 75, 85, 200], right=False, labels=["<65", "65-74", "75-84", ">=85"])
