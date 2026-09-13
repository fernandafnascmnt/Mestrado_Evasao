"""Scoring of fitted models and aggregation of repeated measurements."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
    roc_auc_score,
)

from . import config

METRICS = ("accuracy", "precision", "recall", "f1", "roc_auc")

SCORING = {
    "accuracy": "accuracy",
    "precision": make_scorer(precision_score, pos_label=config.POSITIVE_LABEL, zero_division=0),
    "recall": make_scorer(recall_score, pos_label=config.POSITIVE_LABEL, zero_division=0),
    "f1": make_scorer(f1_score, pos_label=config.POSITIVE_LABEL, zero_division=0),
    "roc_auc": "roc_auc",
}
"""Scorers evaluated during the grid search.

Precision, Recall and F1 are computed for the dropout class, which is the
minority class and the one of practical interest.
"""


def score_on_holdout(estimator, features: pd.DataFrame, target: pd.Series) -> dict[str, float]:
    """Evaluate a fitted estimator on the independent test set."""
    predictions = estimator.predict(features)
    scores = _decision_scores(estimator, features)
    tn, fp, fn, tp = confusion_matrix(target, predictions, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(target, predictions)),
        "precision": float(
            precision_score(target, predictions, pos_label=config.POSITIVE_LABEL, zero_division=0)
        ),
        "recall": float(
            recall_score(target, predictions, pos_label=config.POSITIVE_LABEL, zero_division=0)
        ),
        "f1": float(
            f1_score(target, predictions, pos_label=config.POSITIVE_LABEL, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(target, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def _decision_scores(estimator, features: pd.DataFrame) -> np.ndarray:
    """Continuous scores for ROC-AUC, from probabilities where available."""
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(features)[:, 1]
    return estimator.decision_function(features)


def summarise_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Mean, standard deviation and range of each metric per algorithm."""
    rows = []
    for model, group in results.groupby("model"):
        row: dict[str, object] = {"model": model, "n_repetitions": len(group)}
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1) if len(group) > 1 else 0.0
            row[f"{metric}_min"] = group[metric].min()
            row[f"{metric}_max"] = group[metric].max()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("f1_mean", ascending=False).reset_index(drop=True)


def best_parameter_frequency(results: pd.DataFrame) -> pd.DataFrame:
    """How often each configuration was selected across the repetitions."""
    counts = (
        results.groupby(["model", "best_params"]).size().reset_index(name="count")
    )
    totals = results.groupby("model").size().rename("total")
    counts = counts.join(totals, on="model")
    counts["frequency"] = counts["count"] / counts["total"]
    return (
        counts.drop(columns="total")
        .sort_values(["model", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )


def aggregate_confusion_matrix(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sum the per-repetition confusion matrices and normalise by true class."""
    counts = np.array(
        [[group["tn"].sum(), group["fp"].sum()], [group["fn"].sum(), group["tp"].sum()]]
    )
    index = [f"true_{config.CLASS_NAMES[0]}", f"true_{config.CLASS_NAMES[1]}"]
    columns = [f"predicted_{config.CLASS_NAMES[0]}", f"predicted_{config.CLASS_NAMES[1]}"]
    absolute = pd.DataFrame(counts, index=index, columns=columns)
    normalized = pd.DataFrame(
        counts / counts.sum(axis=1, keepdims=True), index=index, columns=columns
    )
    return absolute, normalized
