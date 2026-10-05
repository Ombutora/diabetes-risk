"""End-to-end scikit-learn pipeline: raw patient record -> risk tier."""
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from src import config
from src.features import (BodyMeasurementReconstructor, CategoricalEncoder,
                          ClinicalFeatureEngineer, MedianModeImputer)


def build_preprocessor() -> Pipeline:
    """Cleaning + feature engineering + encoding, without the model."""
    return Pipeline([
        ("reconstruct", BodyMeasurementReconstructor()),
        ("impute", MedianModeImputer()),
        ("engineer", ClinicalFeatureEngineer()),
        ("encode", CategoricalEncoder()),
    ])


def build_pipeline(model_params: dict | None = None) -> Pipeline:
    """Preprocessing steps followed by the tuned HistGradientBoostingClassifier."""
    params = {**config.MODEL_PARAMS, **(model_params or {})}
    return Pipeline([
        *build_preprocessor().steps,
        ("model", HistGradientBoostingClassifier(**params)),
    ])
