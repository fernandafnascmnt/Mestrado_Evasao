"""Orchestration of the repeated hold-out experiment.

Protocol
--------
1. The target is separated and the post-outcome variable is dropped.
2. A single stratified 80/20 split isolates an independent test set.
3. Feature importance is estimated on the training partition alone and the
   fifteen highest-ranked variables are retained.
4. For every repetition, each algorithm is tuned by stratified 10-fold
   cross-validation on the training partition and evaluated once on the test
   set. Folds are reshuffled between repetitions; the test set never changes.
5. A single refit criterion is used, so all metrics reported for an algorithm
   in a given repetition come from the same fitted model.
6. The algorithms are compared with the Friedman and Nemenyi tests.
"""

from __future__ import annotations

import json
import logging
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from . import config, data, evaluation, features, models, statistics

LOGGER = logging.getLogger(__name__)

CV_SEED_OFFSET = 1000
MODEL_SEED_OFFSET = 2000
REDUCED_REPETITIONS = 1
REDUCED_CV_FOLDS = 3


@dataclass
class ExperimentSettings:
    """Everything the run needs beyond the data itself."""

    data_path: Path
    output_dir: Path
    codebook_path: Path | None = None
    repetitions: int = config.DEFAULT_REPETITIONS
    cv_folds: int = config.DEFAULT_CV_FOLDS
    n_jobs: int = -1
    reduced: bool = False

    @property
    def effective_repetitions(self) -> int:
        return REDUCED_REPETITIONS if self.reduced else self.repetitions

    @property
    def effective_cv_folds(self) -> int:
        return REDUCED_CV_FOLDS if self.reduced else self.cv_folds


def run(settings: ExperimentSettings) -> pd.DataFrame:
    """Execute the full protocol and write every artefact to disk."""
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    X, y = data.load_dataset(settings.data_path)
    descriptions = None
    if settings.codebook_path is not None:
        codebook = data.load_codebook(settings.codebook_path)
        descriptions = data.variable_descriptions(codebook)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )
    LOGGER.info(
        "Split: %d training (%s) and %d test (%s) records",
        len(X_train),
        dict(y_train.value_counts().sort_index()),
        len(X_test),
        dict(y_test.value_counts().sort_index()),
    )

    by_variable, by_indicator = features.rank_feature_importance(X_train, y_train, descriptions)
    selected = features.selected_features(by_variable)
    by_variable.to_csv(settings.output_dir / "feature_importance_by_variable.csv", index=False)
    by_indicator.to_csv(settings.output_dir / "feature_importance_by_indicator.csv", index=False)
    by_variable[by_variable["selected"]].to_csv(
        settings.output_dir / "selected_features.csv", index=False
    )
    LOGGER.info("Selected predictors: %s", ", ".join(selected))

    _write_run_configuration(settings, X, X_train, X_test, y_train, y_test, selected)
    _write_outlier_note(settings.output_dir)

    results = _run_repetitions(
        X_train[selected], X_test[selected], y_train, y_test, settings
    )

    _write_summaries(results, settings.output_dir)
    _write_statistical_comparisons(results, settings.output_dir)

    LOGGER.info("Artefacts written to %s", settings.output_dir.resolve())
    return results


def _run_repetitions(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    settings: ExperimentSettings,
) -> pd.DataFrame:
    """Tune and evaluate every algorithm once per repetition."""
    rows: list[dict[str, object]] = []
    started = time.time()
    detailed_path = settings.output_dir / "per_repetition_results.csv"

    for repetition in range(settings.effective_repetitions):
        LOGGER.info("Repetition %d/%d", repetition + 1, settings.effective_repetitions)
        cv = StratifiedKFold(
            n_splits=settings.effective_cv_folds,
            shuffle=True,
            random_state=CV_SEED_OFFSET + repetition,
        )

        for name in config.MODEL_NAMES:
            elapsed = time.time()
            pipeline, grid = models.build_model(
                name,
                random_state=MODEL_SEED_OFFSET + repetition,
                reduced=settings.reduced,
            )
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=grid,
                scoring=evaluation.SCORING,
                refit=config.REFIT_METRIC,
                cv=cv,
                n_jobs=settings.n_jobs,
                return_train_score=False,
                error_score=np.nan,
            )
            search.fit(X_train, y_train)
            scores = evaluation.score_on_holdout(search.best_estimator_, X_test, y_test)

            rows.append(
                {
                    "repetition": repetition,
                    "model": name,
                    "cv_best_score": float(search.best_score_),
                    "best_params": json.dumps(
                        _readable_params(search.best_params_), ensure_ascii=False, sort_keys=True
                    ),
                    "seconds": round(time.time() - elapsed, 3),
                    **scores,
                }
            )
            # Written after each model so a long run can be inspected, or
            # resumed from, while it is still in progress.
            pd.DataFrame(rows).to_csv(detailed_path, index=False)

            LOGGER.info(
                "  %-14s F1=%.4f Recall=%.4f Precision=%.4f Accuracy=%.4f",
                name,
                scores["f1"],
                scores["recall"],
                scores["precision"],
                scores["accuracy"],
            )

    total = time.time() - started
    (settings.output_dir / "runtime.txt").write_text(
        f"seconds: {total:.3f}\nhours: {total / 3600:.3f}\n", encoding="utf-8"
    )
    return pd.DataFrame(rows)


def _readable_params(params: dict[str, object]) -> dict[str, object]:
    """Strip pipeline prefixes and make the values JSON-serialisable."""
    readable: dict[str, object] = {}
    for key, value in params.items():
        name = key.replace("model__", "")
        if hasattr(value, "get_params"):
            readable[name] = type(value).__name__
        elif isinstance(value, np.generic):
            readable[name] = value.item()
        elif isinstance(value, tuple):
            readable[name] = list(value)
        else:
            readable[name] = value
    return readable


def _write_summaries(results: pd.DataFrame, output_dir: Path) -> None:
    evaluation.summarise_metrics(results).to_csv(output_dir / "metric_summary.csv", index=False)
    evaluation.best_parameter_frequency(results).to_csv(
        output_dir / "best_params_frequency.csv", index=False
    )

    for model, group in results.groupby("model"):
        slug = model.replace(" ", "_")
        absolute, normalized = evaluation.aggregate_confusion_matrix(group)
        absolute.to_csv(output_dir / f"confusion_counts_{slug}.csv")
        normalized.to_csv(output_dir / f"confusion_normalized_{slug}.csv")


def _write_statistical_comparisons(results: pd.DataFrame, output_dir: Path) -> None:
    metrics = (config.PRIMARY_STATISTICAL_METRIC, config.SECONDARY_STATISTICAL_METRIC)
    order = [
        name for name in config.MODEL_NAMES if name in set(results["model"])
    ]

    for metric in metrics:
        comparison = statistics.compare_models(results, metric)
        if comparison is None:
            LOGGER.warning("Not enough data to compare models on %s", metric)
            continue

        (output_dir / f"friedman_{metric}.txt").write_text(
            comparison.as_text(), encoding="utf-8"
        )
        comparison.average_ranks.to_csv(output_dir / f"average_ranks_{metric}.csv")
        comparison.pairwise.to_csv(output_dir / f"nemenyi_{metric}.csv", index=False)
        statistics.rank_difference_matrix(comparison.average_ranks, order).to_csv(
            output_dir / f"rank_differences_{metric}.csv"
        )
        statistics.significance_matrix(comparison.pairwise, order).to_csv(
            output_dir / f"significance_matrix_{metric}.csv"
        )


def _write_run_configuration(
    settings: ExperimentSettings,
    X: pd.DataFrame,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    selected: list[str],
) -> None:
    """Record the protocol and the software versions used."""
    record = {
        "sample": {
            "total": len(X),
            "training": len(X_train),
            "test": len(X_test),
            "training_class_counts": y_train.value_counts().sort_index().to_dict(),
            "test_class_counts": y_test.value_counts().sort_index().to_dict(),
        },
        "protocol": {
            "target": config.TARGET,
            "positive_class": f"{config.POSITIVE_LABEL} = {config.CLASS_NAMES[1]}",
            "excluded_features": list(config.EXCLUDED_FEATURES),
            "outer_split": (
                f"{int((1 - config.TEST_SIZE) * 100)}/{int(config.TEST_SIZE * 100)} stratified, "
                f"random_state={config.RANDOM_STATE}"
            ),
            "inner_cv": (
                f"StratifiedKFold(n_splits={settings.effective_cv_folds}, shuffle=True), "
                f"repeated {settings.effective_repetitions} times"
            ),
            "feature_selection": (
                "Random Forest importance fitted on the training partition after "
                "most-frequent imputation and one-hot encoding; indicator importances "
                "aggregated by source variable"
            ),
            "selector_n_estimators": config.SELECTOR_N_ESTIMATORS,
            "selector_class_weight": "balanced",
            "selected_features": selected,
            "refit_metric": config.REFIT_METRIC,
            "metrics": list(evaluation.METRICS),
            "significance_level": config.ALPHA,
            "reduced_run": settings.reduced,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (settings.output_dir / "run_configuration.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def _write_outlier_note(output_dir: Path) -> None:
    """Document why interquartile-range filtering is not applied."""
    note = (
        "# Outlier handling\n\n"
        "Interquartile-range filtering is not applied in this experiment.\n\n"
        "All predictors are categorised questionnaire answers. Their stored values "
        "are category codes, not measurements on an interval scale, so the quartiles "
        "of such a column carry no meaning and filtering by them would discard valid "
        "responses whose only peculiarity is belonging to an infrequent category.\n\n"
        "Should a continuous predictor be added to the dataset, outlier treatment "
        "should be reconsidered for that variable alone.\n"
    )
    (output_dir / "outlier_handling.md").write_text(note, encoding="utf-8")
