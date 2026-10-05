"""Evaluation metrics for the 3-tier risk classifier.

Run `python -m src.evaluate` to score the saved pipeline on the held-out test split.
"""
import numpy as np
from sklearn.metrics import (balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, roc_auc_score)

from src import config


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    """Headline metrics. Macro F1 is the primary metric (see notebook 03)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(config.CLASS_ORDER)))
    high = config.LABEL_MAP["High"]
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((y_true == y_pred).mean()),
        "roc_auc_ovr_macro": float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")),
        # Share of truly High-risk patients predicted as Low or Moderate
        "high_risk_under_triage_rate": float(cm[high, :high].sum() / cm[high].sum()),
    }


def report(y_true, y_pred) -> str:
    return classification_report(y_true, y_pred, target_names=config.CLASS_ORDER, digits=4)


def main() -> None:
    from src.data import train_test_data
    from src.predict import load_pipeline

    _, X_test, _, y_test = train_test_data()
    pipeline = load_pipeline()
    y_pred = pipeline.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, pipeline.predict_proba(X_test))

    for name, value in metrics.items():
        print(f"{name:<30} {value:.4f}")
    print()
    print(report(y_test, y_pred))


if __name__ == "__main__":
    main()
