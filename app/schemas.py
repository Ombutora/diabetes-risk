"""Request and response models for the diabetes risk API.

Field names match the raw dataset columns so a validated record can be passed
straight to `src.predict.predict`. Numeric limits reject physiologically
implausible values; they are deliberately wider than the training ranges
(values outside those are accepted but flagged with a warning).
"""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

YesNo = Literal["Yes", "No"]
LowModHigh = Literal["Low", "Moderate", "High"]
RiskTier = Literal["Low", "Moderate", "High"]

EXAMPLE_PATIENT = {
    "Age": 32, "Gender": "Male", "Country": "Mexico",
    "Height_cm": 182.1, "Weight_kg": 65.8, "BMI": 19.8, "Waist_Circumference_cm": 71.2,
    "Blood_Glucose": 88.4, "HbA1c": 10.4, "Fasting_Blood_Sugar": 149.5, "Insulin_Level": 27.4,
    "Blood_Pressure_Systolic": 94, "Blood_Pressure_Diastolic": 61,
    "Total_Cholesterol": 138.7, "HDL": 40.1, "LDL": 152.3, "Triglycerides": 250.8, "Heart_Rate": 119,
    "Physical_Activity_Level": "Moderate", "Exercise_Hours_Per_Week": 2.2, "Daily_Walking_Minutes": 106.1,
    "Diet_Quality": "Healthy", "Sugar_Intake_Level": "Low", "Sleep_Hours": 7.9, "Stress_Level": "Moderate",
    "Smoking_Status": "Former", "Alcohol_Consumption": "Never",
    "Family_History_Diabetes": "Yes", "Hypertension": "No", "Heart_Disease": "Yes",
    "Fatty_Liver": "Yes", "PCOS": "No", "Medication_Adherence": "Good",
    "Work_Type": "Retired", "Residence_Type": "Rural", "Daily_Water_Intake_L": 4.4,
}


class PatientRecord(BaseModel):
    """One patient's raw clinical, lifestyle and demographic data.

    Optional fields may be omitted or null and are imputed with training-set
    medians/modes.
    """

    model_config = ConfigDict(extra="forbid", json_schema_extra={"examples": [EXAMPLE_PATIENT]})

    # Demographics
    Age: float | None = Field(None, ge=1, le=120, description="Years")
    Gender: Literal["Female", "Male", "Other"]
    Country: str = Field(..., min_length=1, max_length=64,
                         description="Country of residence; countries not in the training data are allowed but flagged")

    # Anthropometrics
    Height_cm: float | None = Field(None, ge=50, le=250)
    Weight_kg: float | None = Field(None, ge=10, le=350)
    BMI: float | None = Field(None, ge=8, le=100, description="kg/m²; derived from height and weight if omitted")
    Waist_Circumference_cm: float = Field(..., ge=30, le=250)

    # Glycemic markers
    Blood_Glucose: float | None = Field(None, ge=20, le=1000, description="Random blood glucose, mg/dL")
    HbA1c: float | None = Field(None, ge=3, le=20, description="%")
    Fasting_Blood_Sugar: float = Field(..., ge=20, le=1000, description="mg/dL")
    Insulin_Level: float = Field(..., ge=0, le=300, description="µIU/mL")

    # Cardiovascular & lipids
    Blood_Pressure_Systolic: float = Field(..., ge=50, le=300, description="mmHg")
    Blood_Pressure_Diastolic: float = Field(..., ge=30, le=200, description="mmHg")
    Total_Cholesterol: float | None = Field(None, ge=50, le=600, description="mg/dL")
    HDL: float | None = Field(None, ge=5, le=200, description="mg/dL")
    LDL: float | None = Field(None, ge=10, le=500, description="mg/dL")
    Triglycerides: float | None = Field(None, ge=20, le=3000, description="mg/dL")
    Heart_Rate: float = Field(..., ge=20, le=250, description="Beats per minute")

    # Lifestyle
    Physical_Activity_Level: LowModHigh | None = None
    Exercise_Hours_Per_Week: float | None = Field(None, ge=0, le=60)
    Daily_Walking_Minutes: float | None = Field(None, ge=0, le=1440)
    Diet_Quality: Literal["Poor", "Average", "Healthy"]
    Sugar_Intake_Level: LowModHigh
    Sleep_Hours: float | None = Field(None, ge=0, le=24)
    Stress_Level: LowModHigh
    Smoking_Status: Literal["Current", "Former", "Never"]
    Alcohol_Consumption: Literal["Frequently", "Never", "Occasionally"]
    Daily_Water_Intake_L: float = Field(..., ge=0, le=15)

    # Medical history
    Family_History_Diabetes: YesNo
    Hypertension: YesNo
    Heart_Disease: YesNo
    Fatty_Liver: YesNo
    PCOS: YesNo
    Medication_Adherence: Literal["Poor", "Average", "Good"] | None = None

    # Socio-economic
    Work_Type: Literal["Business", "Government", "Private", "Retired", "Student"]
    Residence_Type: Literal["Rural", "Urban"]


class BatchRequest(BaseModel):
    patients: list[PatientRecord] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    risk_tier: RiskTier
    probabilities: dict[RiskTier, float] = Field(..., description="Class probabilities, summing to 1")
    imputed_fields: list[str] = Field(default_factory=list,
                                      description="Fields that were missing and filled from training statistics")
    warnings: list[str] = Field(default_factory=list)


class BatchResponse(BaseModel):
    predictions: list[Prediction]


class HealthResponse(BaseModel):
    status: Literal["ok", "model_unavailable"]
    model_loaded: bool


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_type: str
    trained_at: str | None
    sklearn_version: str | None
    classes: list[str]
    raw_features: list[str]
    optional_features: list[str]
    test_metrics: dict[str, float]
