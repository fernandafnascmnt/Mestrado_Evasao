"""Experimental protocol constants.

Every design decision that affects the reported numbers is declared here so
that the protocol can be inspected without reading the implementation.
"""

from __future__ import annotations

TARGET = "var_0"
"""Column holding the class label."""

EXCLUDED_FEATURES = ("var_3",)
"""Columns removed before modelling.

``var_3`` records the semester in which the student completed or left the
programme. It is only observable after the outcome and would therefore leak
the label into the predictors.
"""

CLASS_MAPPING = {"Concludente/Egresso": 0, "Evadido/Desistente": 1}
"""Mapping from the raw label text to the binary target."""

POSITIVE_LABEL = 1
"""Dropout is the positive class: Precision, Recall and F1 refer to it."""

CLASS_NAMES = {0: "completer", 1: "dropout"}

N_SELECTED_FEATURES = 15
"""Number of original variables retained after the importance ranking."""

TEST_SIZE = 0.20
"""Proportion of the sample held out as an independent test set."""

RANDOM_STATE = 42
"""Seed of the outer split and of the feature-importance estimator."""

DEFAULT_REPETITIONS = 50
DEFAULT_CV_FOLDS = 10

REFIT_METRIC = "f1"
"""Single criterion used to select a configuration in each grid search.

Using one criterion for all metrics guarantees that the Accuracy, Precision,
Recall, F1 and ROC-AUC reported for a given algorithm come from the same
fitted model.
"""

PRIMARY_STATISTICAL_METRIC = "f1"
SECONDARY_STATISTICAL_METRIC = "accuracy"

MODEL_NAMES = (
    "Perceptron",
    "KNN",
    "Naive Bayes",
    "Decision Tree",
    "Random Forest",
    "MLP",
    "SVM",
)

MODEL_ABBREVIATIONS = {
    "Perceptron": "PER",
    "KNN": "KNN",
    "Naive Bayes": "NB",
    "Decision Tree": "DT",
    "Random Forest": "RF",
    "MLP": "MLP",
    "SVM": "SVM",
}

MISSING_TOKENS = frozenset({"", " ", "NULO", "NULL", "null", "None", "NONE", "nan", "NaN"})
"""Textual markers that the survey export uses for unanswered items."""

SELECTOR_N_ESTIMATORS = 500
"""Trees in the Random Forest used to rank feature importance."""

ALPHA = 0.05
"""Significance level of the Friedman and Nemenyi tests."""
