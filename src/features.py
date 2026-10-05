"""Cleaning, feature engineering and encoding steps from notebooks 01-02,
packaged as scikit-learn transformers so they run identically at train and
inference time.

Every transformer takes and returns a pandas DataFrame.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src import config


# ---------------------------------------------------------------------------
# Stateless helper functions
# ---------------------------------------------------------------------------
def reconstruct_body_measurements(df: pd.DataFrame) -> pd.DataFrame:
    """Back-fill missing Height/Weight/BMI from the other two (BMI = kg / m^2)."""
    df = df.copy()
    body_cols = ["Height_cm", "Weight_kg", "BMI"]
    df[body_cols] = df[body_cols].astype(float)

    mask_h = df["Height_cm"].isna() & df["Weight_kg"].notna() & df["BMI"].notna()
    df.loc[mask_h, "Height_cm"] = (100 * np.sqrt(df.loc[mask_h, "Weight_kg"] / df.loc[mask_h, "BMI"])).round(1)

    mask_w = df["Weight_kg"].isna() & df["Height_cm"].notna() & df["BMI"].notna()
    df.loc[mask_w, "Weight_kg"] = (df.loc[mask_w, "BMI"] * (df.loc[mask_w, "Height_cm"] / 100) ** 2).round(1)

    mask_b = df["BMI"].isna() & df["Height_cm"].notna() & df["Weight_kg"].notna()
    df.loc[mask_b, "BMI"] = (df.loc[mask_b, "Weight_kg"] / (df.loc[mask_b, "Height_cm"] / 100) ** 2).round(1)

    return df


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add the 10 clinical domain features from notebook 02."""
    df = df.copy()

    # Glycemic ratios
    df["HbA1c_Glucose_Ratio"] = (df["HbA1c"] / df["Blood_Glucose"]).round(4)
    df["Glucose_Insulin_Ratio"] = (df["Fasting_Blood_Sugar"] / (df["Insulin_Level"] + 1e-5)).round(4)

    # Lipid & cardiovascular ratios
    df["Atherogenic_Index"] = (df["Triglycerides"] / df["HDL"]).round(4)
    df["Total_Cholesterol_HDL_Ratio"] = (df["Total_Cholesterol"] / df["HDL"]).round(4)
    df["LDL_HDL_Ratio"] = (df["LDL"] / df["HDL"]).round(4)
    df["Mean_Arterial_Pressure"] = (
        df["Blood_Pressure_Diastolic"] + (df["Blood_Pressure_Systolic"] - df["Blood_Pressure_Diastolic"]) / 3
    ).round(2)
    df["Pulse_Pressure"] = df["Blood_Pressure_Systolic"] - df["Blood_Pressure_Diastolic"]

    # Physical activity
    df["Total_Active_Hours_Weekly"] = (df["Exercise_Hours_Per_Week"] + df["Daily_Walking_Minutes"] * 7 / 60).round(2)

    # Metabolic syndrome components
    c1 = (df["Waist_Circumference_cm"] > 90).astype(int)
    c2 = (
        (df["Blood_Pressure_Systolic"] >= 130)
        | (df["Blood_Pressure_Diastolic"] >= 85)
        | (df["Hypertension"] == "Yes")
    ).astype(int)
    c3 = (df["Fasting_Blood_Sugar"] >= 100).astype(int)
    c4 = (df["Triglycerides"] >= 150).astype(int)
    c5 = (df["HDL"] < 45).astype(int)
    df["Metabolic_Syndrome_Risk_Score"] = c1 + c2 + c3 + c4 + c5

    comorb_cols = ["Hypertension", "Heart_Disease", "Fatty_Liver", "PCOS", "Family_History_Diabetes"]
    df["Comorbidity_Count"] = sum((df[c] == "Yes").astype(int) for c in comorb_cols)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Ordinal, binary and one-hot encoding with fixed category lists.

    Unlike pd.get_dummies, the output columns do not depend on which categories
    appear in `df`, so a single patient record encodes to the same columns as
    the training set. Unknown ordinal values become NaN; unknown nominal values
    become all-zero dummies (i.e. the reference level).
    """
    df = df.copy()

    for col, mapping in config.ORDINAL_MAPPINGS.items():
        df[col] = df[col].map(mapping)

    for col in config.BINARY_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    for col, categories in config.NOMINAL_CATEGORIES.items():
        for cat in categories[1:]:
            df[f"{col}_{cat}"] = (df[col] == cat).astype(int)
    return df.drop(columns=list(config.NOMINAL_CATEGORIES))


# ---------------------------------------------------------------------------
# Transformers
# ---------------------------------------------------------------------------
class BodyMeasurementReconstructor(BaseEstimator, TransformerMixin):
    """Deterministic Height/Weight/BMI back-calculation (notebook 01, section 4)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return reconstruct_body_measurements(X)


class MedianModeImputer(BaseEstimator, TransformerMixin):
    """Median imputation for numeric columns, mode for categorical columns.

    Statistics are learned in `fit`, so in the pipeline they come from the
    training split only.
    """

    def fit(self, X, y=None):
        self.medians_ = {c: X[c].median() for c in config.NUMERIC_COLS}
        self.modes_ = {c: X[c].mode()[0] for c in config.CATEGORICAL_COLS}
        return self

    def transform(self, X):
        X = X.copy()
        for col, value in {**self.medians_, **self.modes_}.items():
            X[col] = X[col].fillna(value)
        return X


class ClinicalFeatureEngineer(BaseEstimator, TransformerMixin):
    """Adds the domain ratios and composite scores (notebook 02, section 4)."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return add_engineered_features(X)


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """Encodes categoricals and fixes the output column order seen during fit."""

    def fit(self, X, y=None):
        self.feature_names_out_ = encode_categoricals(X.head(1)).columns.tolist()
        return self

    def transform(self, X):
        return encode_categoricals(X)[self.feature_names_out_]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_out_, dtype=object)
