"""Inference helpers: load the saved pipeline and score raw patient records."""
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config


@lru_cache(maxsize=None)
def load_pipeline(path: Path = config.MODEL_PATH) -> Pipeline:
    if not Path(path).exists():
        raise FileNotFoundError(f"No trained pipeline at {path}. Run `python -m src.train` first.")
    return joblib.load(path)


def predict(records: dict | list[dict] | pd.DataFrame, pipeline: Pipeline | None = None) -> list[dict]:
    """Predict the risk tier for one or more raw patient records.

    Records use the raw column names (see `config.RAW_FEATURES`). Missing or
    None values are imputed. Returns one dict per record:
        {"risk_tier": "High", "probabilities": {"Low": 0.01, "Moderate": 0.12, "High": 0.87}}
    """
    if pipeline is None:
        pipeline = load_pipeline()

    if isinstance(records, dict):
        records = [records]
    df = pd.DataFrame(records)

    missing = [c for c in config.RAW_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    df = df[config.RAW_FEATURES].copy()
    # None in a single-row frame gives object dtype; numeric steps need floats.
    # Raises on non-numeric strings rather than silently imputing them.
    df[config.NUMERIC_COLS] = df[config.NUMERIC_COLS].apply(pd.to_numeric).astype(float)

    probas = pipeline.predict_proba(df)
    return [
        {
            "risk_tier": config.CLASS_ORDER[int(row.argmax())],
            "probabilities": {tier: round(float(p), 4) for tier, p in zip(config.CLASS_ORDER, row)},
        }
        for row in probas
    ]
