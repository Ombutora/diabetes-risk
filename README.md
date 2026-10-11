# Diabetes Risk Prediction

Predicts a patient's diabetes risk tier — **Low**, **Moderate** or **High** — from
clinical, lifestyle and demographic data, and serves the model as a REST API.

> **Decision-support only.** This model is trained on a synthetic/educational dataset
> and is not a medical device. It does not diagnose diabetes and must not replace
> clinical judgement or laboratory testing.

## Results

Held-out test set (10,000 patients, stratified 80/20 split):

| Metric | Score |
|---|---|
| Macro F1 (primary) | 0.845 |
| Weighted F1 | 0.941 |
| Balanced accuracy | 0.880 |
| ROC-AUC (one-vs-rest, macro) | 0.993 |
| High-risk under-triage rate | 6.6% |

*High-risk under-triage rate* is the share of truly High-risk patients predicted as
Low or Moderate — the most costly error for a screening tool.

Model: `HistGradientBoostingClassifier` with balanced class weights, tuned by
randomised search on macro F1 (see `notebooks/03_modeling.ipynb`).

## Project structure

```
app/            FastAPI service (main.py) and request/response schemas
src/            Reusable pipeline code
  config.py     Paths, column definitions, model hyperparameters
  data.py       Loading and train/test split
  features.py   Cleaning, feature engineering and encoding as sklearn transformers
  pipeline.py   End-to-end pipeline: raw record -> risk tier
  train.py      Train and save the pipeline + metadata
  evaluate.py   Metrics on the held-out split
  predict.py    Inference helpers
notebooks/      01 cleaning, 02 EDA & feature engineering, 03 modeling
models/         Trained pipeline (.pkl, not in git) and its metadata (.json)
reports/figures Model comparison, ROC/PR curves, confusion matrix, feature importance
tests/          Unit tests (features) and API integration tests
```

The cleaning, imputation, feature engineering and encoding from the notebooks are
packaged inside the saved scikit-learn pipeline, so the API accepts **raw** patient
records and applies exactly the same transformations used in training.

## Setup

Requires Python 3.11+ (developed on 3.13).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on macOS/Linux)
pip install -r requirements-dev.txt
```

Place the raw dataset at `data/raw/diabetes_risk_prediction_dataset.csv`.

## Train and evaluate

```bash
python -m src.train       # writes models/diabetes_risk_pipeline.pkl and .json
python -m src.evaluate    # scores the saved pipeline on the test split
pytest                    # run the tests
```

## Run the API

Locally:

```bash
uvicorn app.main:app --reload
```

With Docker (train first — the image copies the local `.pkl`):

```bash
docker build -t diabetes-risk .
docker run -p 8000:8000 diabetes-risk
```

Interactive docs are at <http://localhost:8000/docs>.

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Service and model status |
| GET | `/model/info` | Model type, training date, features, test metrics |
| POST | `/predict` | Predict one patient |
| POST | `/predict/batch` | Predict up to 1,000 patients |

Example:

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
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
  "Work_Type": "Retired", "Residence_Type": "Rural", "Daily_Water_Intake_L": 4.4
}'
```

```json
{
  "risk_tier": "Moderate",
  "probabilities": {"Low": 0.002, "Moderate": 0.8548, "High": 0.1432},
  "imputed_fields": [],
  "warnings": []
}
```

### Input handling

- **Optional fields** (e.g. `HbA1c`, `Age`, lipids, sleep) may be omitted or `null`;
  they are filled with training-set medians/modes and listed in `imputed_fields`.
  `BMI`, `Height_cm` or `Weight_kg` is calculated from the other two when possible.
- **Implausible values** (e.g. `Age: 500`) and unknown categories are rejected with HTTP 422.
- **Values outside the training range**, unseen countries and diastolic ≥ systolic
  blood pressure are accepted but reported in `warnings`, since those predictions
  are less reliable.

`MODEL_PATH` (env var) overrides the model location; `PORT` sets the container port.
