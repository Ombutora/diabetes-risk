"""Loading the raw dataset and splitting it into features / target."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def load_raw(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV and drop identifier / target-leakage columns."""
    df = pd.read_csv(path)
    return df.drop(columns=[c for c in config.DROP_COLS if c in df.columns])


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Return raw feature columns and the ordinal-encoded target (Low=0, Moderate=1, High=2)."""
    unknown = set(df[config.TARGET].dropna().unique()) - set(config.CLASS_ORDER)
    if unknown:
        raise ValueError(f"Unexpected {config.TARGET} labels: {sorted(unknown)}")
    if df[config.TARGET].isna().any():
        raise ValueError(f"{config.TARGET} contains missing values")

    X = df[config.RAW_FEATURES].copy()
    y = df[config.TARGET].map(config.LABEL_MAP).to_numpy()
    return X, y


def train_test_data(df: pd.DataFrame | None = None):
    """Stratified train/test split, identical to the split used in notebook 03."""
    if df is None:
        df = load_raw()
    X, y = split_features_target(df)
    return train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
