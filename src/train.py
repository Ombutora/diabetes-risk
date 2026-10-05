"""Train the end-to-end pipeline on the raw dataset and save it.

Usage:
    python -m src.train
    python -m src.train --data path/to/raw.csv --output models/my_pipeline.pkl
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import sklearn

from src import config
from src.data import load_raw, train_test_data
from src.evaluate import compute_metrics, report
from src.pipeline import build_pipeline


def train(data_path: Path = config.RAW_DATA_PATH, output_path: Path = config.MODEL_PATH) -> dict:
    X_train, X_test, y_train, y_test = train_test_data(load_raw(data_path))
    print(f"Training on {len(X_train):,} rows, evaluating on {len(X_test):,} rows")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, pipeline.predict_proba(X_test))
    print(report(y_test, y_pred))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sklearn_version": sklearn.__version__,
        "data_path": str(data_path),
        "target": config.TARGET,
        "classes": config.CLASS_ORDER,
        "raw_features": config.RAW_FEATURES,
        "model_features": pipeline.named_steps["encode"].feature_names_out_,
        "model_params": config.MODEL_PARAMS,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "test_metrics": {k: round(v, 4) for k, v in metrics.items()},
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved pipeline to {output_path}")
    print(f"Saved metadata to {metadata_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the diabetes risk pipeline.")
    parser.add_argument("--data", type=Path, default=config.RAW_DATA_PATH, help="Raw CSV path")
    parser.add_argument("--output", type=Path, default=config.MODEL_PATH, help="Output .pkl path")
    args = parser.parse_args()
    train(args.data, args.output)


if __name__ == "__main__":
    main()
