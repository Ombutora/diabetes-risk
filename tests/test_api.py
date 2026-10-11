"""Integration tests for the FastAPI service. Requires a trained pipeline."""
import pytest
from fastapi.testclient import TestClient

from app.main import MODEL_PATH, app
from app.schemas import EXAMPLE_PATIENT
from src import config

pytestmark = pytest.mark.skipif(not MODEL_PATH.exists(),
                                reason="No trained pipeline; run `python -m src.train`")


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:  # runs the lifespan, which loads the model
        yield c


def patient(**overrides) -> dict:
    return {**EXAMPLE_PATIENT, **overrides}


def test_health(client):
    assert client.get("/health").json() == {"status": "ok", "model_loaded": True}


def test_model_info(client):
    body = client.get("/model/info").json()
    assert body["model_type"] == "HistGradientBoostingClassifier"
    assert body["classes"] == config.CLASS_ORDER
    assert "macro_f1" in body["test_metrics"]


def test_predict_example(client):
    response = client.post("/predict", json=patient())
    assert response.status_code == 200
    body = response.json()
    assert body["risk_tier"] in config.CLASS_ORDER
    assert set(body["probabilities"]) == set(config.CLASS_ORDER)
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["risk_tier"] == max(body["probabilities"], key=body["probabilities"].get)
    assert body["imputed_fields"] == []
    assert body["warnings"] == []


def test_higher_glycemic_markers_do_not_lower_high_risk_probability(client):
    low = client.post("/predict", json=patient(HbA1c=5.0, Fasting_Blood_Sugar=85)).json()
    high = client.post("/predict", json=patient(HbA1c=11.0, Fasting_Blood_Sugar=200)).json()
    assert high["probabilities"]["High"] >= low["probabilities"]["High"]


def test_optional_fields_can_be_omitted(client):
    record = {k: v for k, v in patient().items() if k not in ("HbA1c", "Sleep_Hours", "BMI")}
    body = client.post("/predict", json=record).json()
    assert set(body["imputed_fields"]) == {"HbA1c", "Sleep_Hours"}
    assert any("BMI" in w and "calculated" in w for w in body["warnings"])


def test_out_of_training_range_value_is_flagged(client):
    body = client.post("/predict", json=patient(Triglycerides=600)).json()
    assert any("Triglycerides" in w and "training range" in w for w in body["warnings"])


def test_unknown_country_is_flagged(client):
    body = client.post("/predict", json=patient(Country="Kenya")).json()
    assert any("Kenya" in w for w in body["warnings"])


@pytest.mark.parametrize("overrides", [
    {"Age": 500},                     # implausible value
    {"Gender": "Unknown"},            # invalid category
    {"Fasting_Blood_Sugar": None},    # required field
    {"HbA1c": "high"},                # wrong type
    {"Unexpected_Field": 1},          # extra field
])
def test_invalid_input_is_rejected(client, overrides):
    assert client.post("/predict", json=patient(**overrides)).status_code == 422


def test_batch_matches_single_predictions(client):
    records = [patient(), patient(HbA1c=5.0, Fasting_Blood_Sugar=85)]
    batch = client.post("/predict/batch", json={"patients": records}).json()["predictions"]
    singles = [client.post("/predict", json=r).json() for r in records]
    assert batch == singles


def test_empty_batch_is_rejected(client):
    assert client.post("/predict/batch", json={"patients": []}).status_code == 422
