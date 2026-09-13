"""Estimator pipelines and hyperparameter search spaces.

Each entry returns a scikit-learn ``Pipeline`` together with the grid explored
by ``GridSearchCV``. Estimators whose decision boundary depends on feature
magnitude receive the scaling step; tree ensembles and the Naive Bayes family
do not. Class weighting is part of the search space wherever the estimator
supports it natively, so the choice between weighted and unweighted fitting is
made by cross-validation rather than imposed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .features import preprocessing_steps

Grid = dict[str, Any] | list[dict[str, Any]]

CLASS_WEIGHTS = [None, "balanced"]


def build_model(name: str, random_state: int, reduced: bool = False) -> tuple[Pipeline, Grid]:
    """Return the pipeline and search space for one algorithm.

    Parameters
    ----------
    name:
        One of :data:`dropout_prediction.config.MODEL_NAMES`.
    random_state:
        Seed of the estimator for this repetition.
    reduced:
        Collapse each grid to a single configuration. Intended only for
        verifying that the pipeline runs end to end; results obtained with it
        must not be reported.
    """
    builder = _BUILDERS.get(name)
    if builder is None:
        raise ValueError(f"Unknown model {name!r}")
    return builder(random_state, reduced)


def _perceptron(random_state: int, reduced: bool) -> tuple[Pipeline, Grid]:
    pipeline = Pipeline(
        preprocessing_steps(scale=True)
        + [("model", Perceptron(max_iter=2000, tol=1e-3, random_state=random_state))]
    )
    grid = {
        "model__penalty": ["l2"] if reduced else [None, "l2", "l1"],
        "model__alpha": [1e-4] if reduced else [1e-4, 1e-3],
        "model__eta0": [0.1] if reduced else [0.01, 0.1, 1.0],
        "model__class_weight": ["balanced"] if reduced else CLASS_WEIGHTS,
    }
    return pipeline, grid


def _knn(random_state: int, reduced: bool) -> tuple[Pipeline, Grid]:
    pipeline = Pipeline(
        preprocessing_steps(scale=True) + [("model", KNeighborsClassifier(n_jobs=1))]
    )
    grid = {
        "model__n_neighbors": [5] if reduced else [3, 5, 7, 9, 11],
        "model__weights": ["distance"] if reduced else ["uniform", "distance"],
        "model__metric": ["euclidean"] if reduced else ["euclidean", "manhattan"],
    }
    return pipeline, grid


def _naive_bayes(random_state: int, reduced: bool) -> tuple[Pipeline, Grid]:
    """Three likelihood models compete within the Naive Bayes family.

    After one-hot encoding the design matrix is binary, which makes the
    Bernoulli likelihood a natural candidate alongside the Gaussian variant
    used in earlier work.
    """
    pipeline = Pipeline(preprocessing_steps(scale=False) + [("model", GaussianNB())])
    if reduced:
        return pipeline, [{"model": [GaussianNB()], "model__var_smoothing": [1e-9]}]
    grid = [
        {"model": [GaussianNB()], "model__var_smoothing": np.logspace(-11, -7, 5)},
        {
            "model": [MultinomialNB()],
            "model__alpha": [0.1, 0.5, 1.0],
            "model__fit_prior": [True, False],
        },
        {
            "model": [BernoulliNB()],
            "model__alpha": [0.1, 0.5, 1.0],
            "model__fit_prior": [True, False],
        },
    ]
    return pipeline, grid


def _decision_tree(random_state: int, reduced: bool) -> tuple[Pipeline, Grid]:
    pipeline = Pipeline(
        preprocessing_steps(scale=False)
        + [("model", DecisionTreeClassifier(random_state=random_state))]
    )
    grid = {
        "model__criterion": ["gini"] if reduced else ["gini", "entropy"],
        "model__max_depth": [5] if reduced else [None, 3, 5, 10],
        "model__min_samples_split": [5] if reduced else [2, 5, 10],
        "model__min_samples_leaf": [2] if reduced else [1, 2],
        "model__class_weight": ["balanced"] if reduced else CLASS_WEIGHTS,
    }
    return pipeline, grid


def _random_forest(random_state: int, reduced: bool) -> tuple[Pipeline, Grid]:
    pipeline = Pipeline(
        preprocessing_steps(scale=False)
        + [("model", RandomForestClassifier(random_state=random_state, n_jobs=1))]
    )
    grid = {
        "model__criterion": ["gini"] if reduced else ["gini", "entropy"],
        "model__n_estimators": [100] if reduced else [100, 300],
        "model__max_depth": [5] if reduced else [None, 5],
        "model__min_samples_split": [5] if reduced else [2, 5],
        "model__min_samples_leaf": [1] if reduced else [1, 2],
        "model__max_features": ["sqrt"],
        "model__class_weight": ["balanced"] if reduced else CLASS_WEIGHTS,
    }
    return pipeline, grid


def _mlp(random_state: int, reduced: bool) -> tuple[Pipeline, Grid]:
    pipeline = Pipeline(
        preprocessing_steps(scale=True)
        + [("model", MLPClassifier(max_iter=1000, solver="adam", random_state=random_state))]
    )
    grid = {
        "model__hidden_layer_sizes": [(20,)] if reduced else [(20,), (50,), (30, 15)],
        "model__activation": ["relu"] if reduced else ["relu", "tanh"],
        "model__alpha": [1e-4] if reduced else [1e-4, 1e-3],
        "model__learning_rate_init": [0.001] if reduced else [0.001, 0.01],
    }
    return pipeline, grid


def _svm(random_state: int, reduced: bool) -> tuple[Pipeline, Grid]:
    pipeline = Pipeline(preprocessing_steps(scale=True) + [("model", SVC())])
    if reduced:
        return pipeline, [
            {
                "model__kernel": ["rbf"],
                "model__C": [1],
                "model__gamma": ["scale"],
                "model__class_weight": ["balanced"],
            }
        ]
    grid = [
        {
            "model__kernel": ["linear"],
            "model__C": [0.1, 1, 10],
            "model__class_weight": CLASS_WEIGHTS,
        },
        {
            "model__kernel": ["rbf"],
            "model__C": [0.1, 1, 10],
            "model__gamma": [0.1, 0.01, 0.001],
            "model__class_weight": CLASS_WEIGHTS,
        },
        {
            "model__kernel": ["poly"],
            "model__C": [1, 10],
            "model__gamma": [0.1, 0.01],
            "model__degree": [2, 3],
            "model__class_weight": CLASS_WEIGHTS,
        },
        {
            "model__kernel": ["sigmoid"],
            "model__C": [1, 10],
            "model__gamma": [0.1, 0.01],
            "model__class_weight": CLASS_WEIGHTS,
        },
    ]
    return pipeline, grid


_BUILDERS = {
    "Perceptron": _perceptron,
    "KNN": _knn,
    "Naive Bayes": _naive_bayes,
    "Decision Tree": _decision_tree,
    "Random Forest": _random_forest,
    "MLP": _mlp,
    "SVM": _svm,
}
