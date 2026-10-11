"""Unit tests for the preprocessing steps in src.features and src.predict."""
import numpy as np
import pandas as pd
import pytest

from app.schemas import EXAMPLE_PATIENT
from src import config
from src.features import (CategoricalEncoder, MedianModeImputer, add_engineered_features,
                          encode_categoricals, reconstruct_body_measurements)
from src.pipeline import build_preprocessor
from src.predict import predict


def make_df(n: int = 1, **overrides) -> pd.DataFrame:
    rows = [{**EXAMPLE_PATIENT, **overrides} for _ in range(n)]
    return pd.DataFrame(rows)[config.RAW_FEATURES]


# ---------------------------------------------------------------------------
# Body measurement reconstruction
# ---------------------------------------------------------------------------
def test_reconstructs_bmi_from_height_and_weight():
    df = reconstruct_body_measurements(make_df(Height_cm=180.0, Weight_kg=81.0, BMI=None))
    assert df.loc[0, "BMI"] == pytest.approx(25.0, abs=0.05)


def test_reconstructs_height_and_weight_from_bmi():
    df = make_df(2, Height_cm=180.0, Weight_kg=81.0, BMI=25.0)
    df.loc[0, "Height_cm"] = np.nan
    df.loc[1, "Weight_kg"] = np.nan
    out = reconstruct_body_measurements(df)
    assert out.loc[0, "Height_cm"] == pytest.approx(180.0, abs=0.1)
    assert out.loc[1, "Weight_kg"] == pytest.approx(81.0, abs=0.1)


def test_leaves_values_missing_when_two_body_measurements_are_missing():
    out = reconstruct_body_measurements(make_df(Height_cm=None, Weight_kg=None, BMI=25.0))
    assert out[["Height_cm", "Weight_kg"]].isna().all(axis=None)


# ---------------------------------------------------------------------------
# Imputation
# ---------------------------------------------------------------------------
def test_imputer_uses_statistics_from_fit_data_only():
    train = make_df(3)
    train["HbA1c"] = [5.0, 6.0, 100.0]
    train["Diet_Quality"] = ["Poor", "Poor", "Healthy"]
    imputer = MedianModeImputer().fit(train)

    test = make_df(HbA1c=None, Diet_Quality=None)
    out = imputer.transform(test)
    assert out.loc[0, "HbA1c"] == 6.0
    assert out.loc[0, "Diet_Quality"] == "Poor"


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def test_engineered_features_values():
    df = add_engineered_features(make_df(
        Blood_Pressure_Systolic=120, Blood_Pressure_Diastolic=80,
        Triglycerides=150, HDL=50, Waist_Circumference_cm=95, Fasting_Blood_Sugar=90,
        Hypertension="No", Heart_Disease="Yes", Fatty_Liver="No", PCOS="No",
        Family_History_Diabetes="Yes", Exercise_Hours_Per_Week=2, Daily_Walking_Minutes=60,
    ))
    row = df.iloc[0]
    assert row["Mean_Arterial_Pressure"] == pytest.approx(93.33, abs=0.01)
    assert row["Pulse_Pressure"] == 40
    assert row["Atherogenic_Index"] == pytest.approx(3.0)
    assert row["Total_Active_Hours_Weekly"] == pytest.approx(9.0)
    # waist > 90 and triglycerides >= 150 only
    assert row["Metabolic_Syndrome_Risk_Score"] == 2
    assert row["Comorbidity_Count"] == 2


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------
def test_encoding_columns_do_not_depend_on_categories_present():
    a = encode_categoricals(make_df(Country="Mexico", Gender="Male"))
    b = encode_categoricals(make_df(Country="Japan", Gender="Female"))
    assert a.columns.tolist() == b.columns.tolist()


def test_reference_level_and_unknown_category_encode_to_all_zeros():
    country_cols = [f"Country_{c}" for c in config.NOMINAL_CATEGORIES["Country"][1:]]
    for country in ["Argentina", "Kenya"]:
        out = encode_categoricals(make_df(Country=country))
        assert out[country_cols].sum(axis=1).iloc[0] == 0


def test_ordinal_and_binary_encoding():
    out = encode_categoricals(make_df(Diet_Quality="Healthy", Stress_Level="Low", PCOS="Yes"))
    assert out.loc[0, "Diet_Quality"] == 2
    assert out.loc[0, "Stress_Level"] == 0
    assert out.loc[0, "PCOS"] == 1


def test_categorical_encoder_keeps_fitted_column_order():
    encoder = CategoricalEncoder().fit(make_df())
    out = encoder.transform(make_df(Country="Japan"))
    assert out.columns.tolist() == encoder.feature_names_out_


def test_preprocessor_output_is_numeric_and_complete():
    train = make_df(5)
    test = make_df(HbA1c=None, BMI=None)
    test[config.NUMERIC_COLS] = test[config.NUMERIC_COLS].astype(float)  # as predict() does
    out = build_preprocessor().fit(train).transform(test)
    assert out.notna().all(axis=None)
    assert all(np.issubdtype(t, np.number) for t in out.dtypes)


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------
def test_predict_rejects_records_missing_columns():
    record = {k: v for k, v in EXAMPLE_PATIENT.items() if k != "HbA1c"}
    with pytest.raises(ValueError, match="HbA1c"):
        predict(record, pipeline=object())
