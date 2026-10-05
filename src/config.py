"""Project-wide paths, column definitions and model settings."""
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "diabetes_risk_prediction_dataset.csv"
INTERIM_DATA_PATH = DATA_DIR / "interim" / "cleaned_diabetes_risk_dataset.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "engineered_diabetes_risk_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "diabetes_risk_pipeline.pkl"
METADATA_PATH = MODELS_DIR / "diabetes_risk_pipeline.json"

# ---------------------------------------------------------------------------
# Target
# ---------------------------------------------------------------------------
TARGET = "Diabetes_Risk"
CLASS_ORDER = ["Low", "Moderate", "High"]  # ordinal clinical order -> labels 0, 1, 2
LABEL_MAP = {label: i for i, label in enumerate(CLASS_ORDER)}

# Identifier and target-derived columns removed in notebook 01 (leakage)
DROP_COLS = [
    "Patient_ID",
    "Diabetes_Risk_Score",
    "AI_Health_Recommendation",
    "Doctor_Consultation_Needed",
]

# ---------------------------------------------------------------------------
# Raw input features (what a patient record must provide)
# ---------------------------------------------------------------------------
NUMERIC_COLS = [
    "Age",
    "Height_cm",
    "Weight_kg",
    "BMI",
    "Waist_Circumference_cm",
    "Blood_Glucose",
    "HbA1c",
    "Fasting_Blood_Sugar",
    "Insulin_Level",
    "Blood_Pressure_Systolic",
    "Blood_Pressure_Diastolic",
    "Total_Cholesterol",
    "HDL",
    "LDL",
    "Triglycerides",
    "Heart_Rate",
    "Exercise_Hours_Per_Week",
    "Daily_Walking_Minutes",
    "Sleep_Hours",
    "Daily_Water_Intake_L",
]

ORDINAL_MAPPINGS = {
    "Physical_Activity_Level": {"Low": 0, "Moderate": 1, "High": 2},
    "Diet_Quality": {"Poor": 0, "Average": 1, "Healthy": 2},
    "Sugar_Intake_Level": {"Low": 0, "Moderate": 1, "High": 2},
    "Stress_Level": {"Low": 0, "Moderate": 1, "High": 2},
    "Medication_Adherence": {"Poor": 0, "Average": 1, "Good": 2},
}

BINARY_COLS = ["Family_History_Diabetes", "Hypertension", "Heart_Disease", "Fatty_Liver", "PCOS"]

# Full category lists for one-hot encoding. The first category of each is the
# dropped reference level, matching pd.get_dummies(drop_first=True) in notebook 02.
NOMINAL_CATEGORIES = {
    "Gender": ["Female", "Male", "Other"],
    "Country": [
        "Argentina", "Australia", "Bangladesh", "Brazil", "Canada", "China", "Egypt",
        "France", "Germany", "India", "Indonesia", "Italy", "Japan", "Malaysia",
        "Mexico", "Nigeria", "Pakistan", "Russia", "Saudi Arabia", "South Africa",
        "South Korea", "Spain", "Turkey", "United Kingdom", "United States",
    ],
    "Smoking_Status": ["Current", "Former", "Never"],
    "Alcohol_Consumption": ["Frequently", "Never", "Occasionally"],
    "Work_Type": ["Business", "Government", "Private", "Retired", "Student"],
    "Residence_Type": ["Rural", "Urban"],
}

CATEGORICAL_COLS = list(ORDINAL_MAPPINGS) + BINARY_COLS + list(NOMINAL_CATEGORIES)
RAW_FEATURES = NUMERIC_COLS + CATEGORICAL_COLS

# ---------------------------------------------------------------------------
# Training & model
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# Best hyperparameters from the randomised search in notebook 03
MODEL_PARAMS = {
    "learning_rate": 0.08,
    "max_iter": 200,
    "max_leaf_nodes": 31,
    "max_depth": 4,
    "min_samples_leaf": 40,
    "l2_regularization": 0.0,
    "class_weight": "balanced",
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 20,
    "random_state": RANDOM_STATE,
}
