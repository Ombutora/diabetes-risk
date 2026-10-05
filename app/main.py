"""FastAPI service for diabetes risk prediction.

Run locally:
    uvicorn app.main:app --reload

The model path defaults to models/diabetes_risk_pipeline.pkl and can be
overridden with the MODEL_PATH environment variable.
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request

from app.schemas import (BatchRequest, BatchResponse, HealthResponse, ModelInfo,
                         PatientRecord, Prediction)
from src import config
from src.predict import load_pipeline, predict

MODEL_PATH = Path(os.getenv("MODEL_PATH", config.MODEL_PATH))


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.pipeline = load_pipeline(MODEL_PATH)
    except FileNotFoundError:
        app.state.pipeline = None
    metadata_path = MODEL_PATH.with_suffix(".json")
    app.state.metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    yield


app = FastAPI(
    title="Diabetes Risk Prediction API",
    description=(
        "Predicts a patient's diabetes risk tier (Low / Moderate / High) from clinical, "
        "lifestyle and demographic data. Decision-support only — not a diagnosis."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


def get_pipeline(request: Request):
    pipeline = request.app.state.pipeline
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Train it with `python -m src.train`.")
    return pipeline


def input_notes(record: dict) -> tuple[list[str], list[str]]:
    """Return (imputed_fields, warnings) for one validated record."""
    imputed, warnings = [], []
    body = ("Height_cm", "Weight_kg", "BMI")
    known_body = [f for f in body if record.get(f) is not None]

    for field in config.OPTIONAL_FEATURES:
        if record.get(field) is not None:
            continue
        if field in body and len(known_body) >= 2:
            warnings.append(f"{field} was not provided and was calculated from {' and '.join(known_body)}.")
        else:
            imputed.append(field)

    for field, (low, high) in config.TRAINING_RANGES.items():
        value = record.get(field)
        if value is not None and not low <= value <= high:
            warnings.append(f"{field}={value} is outside the training range [{low}, {high}]; "
                            "the prediction is an extrapolation.")

    # A warning rather than a 422: ~8% of the training data has diastolic >= systolic
    if record["Blood_Pressure_Diastolic"] >= record["Blood_Pressure_Systolic"]:
        warnings.append("Blood_Pressure_Diastolic is not lower than Blood_Pressure_Systolic; "
                        "please check the blood pressure reading.")

    if record["Country"] not in config.NOMINAL_CATEGORIES["Country"]:
        warnings.append(f"Country '{record['Country']}' was not in the training data and is treated "
                        f"as the reference country ({config.NOMINAL_CATEGORIES['Country'][0]}).")
    return imputed, warnings


def score(records: list[PatientRecord], pipeline) -> list[Prediction]:
    dicts = [r.model_dump() for r in records]
    results = predict(dicts, pipeline=pipeline)
    predictions = []
    for record, result in zip(dicts, results):
        imputed, warnings = input_notes(record)
        predictions.append(Prediction(**result, imputed_fields=imputed, warnings=warnings))
    return predictions


@app.get("/health", response_model=HealthResponse, tags=["service"])
def health(request: Request):
    loaded = request.app.state.pipeline is not None
    return HealthResponse(status="ok" if loaded else "model_unavailable", model_loaded=loaded)


@app.get("/model/info", response_model=ModelInfo, tags=["service"])
def model_info(request: Request):
    pipeline = get_pipeline(request)
    meta = request.app.state.metadata
    return ModelInfo(
        model_type=type(pipeline.named_steps["model"]).__name__,
        trained_at=meta.get("trained_at"),
        sklearn_version=meta.get("sklearn_version"),
        classes=config.CLASS_ORDER,
        raw_features=config.RAW_FEATURES,
        optional_features=config.OPTIONAL_FEATURES,
        test_metrics=meta.get("test_metrics", {}),
    )


@app.post("/predict", response_model=Prediction, tags=["prediction"])
def predict_one(patient: PatientRecord, request: Request):
    return score([patient], get_pipeline(request))[0]


@app.post("/predict/batch", response_model=BatchResponse, tags=["prediction"])
def predict_batch(batch: BatchRequest, request: Request):
    return BatchResponse(predictions=score(batch.patients, get_pipeline(request)))
