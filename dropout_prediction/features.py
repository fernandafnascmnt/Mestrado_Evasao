"""Preprocessing steps and training-only feature selection."""

from __future__ import annotations

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

from . import config


def make_one_hot_encoder() -> OneHotEncoder:
    """Encoder used both for selection and inside the modelling pipelines.

    A dense output is used because the dataset is small and several estimators
    (Naive Bayes variants, ``MinMaxScaler``, Random Forest) handle dense input
    more conveniently. Unseen categories are ignored so that a category absent
    from the training folds does not break the transform.
    """
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def preprocessing_steps(scale: bool) -> list[tuple[str, object]]:
    """Build the shared preprocessing steps of a modelling pipeline.

    Imputation and encoding live inside the pipeline so that they are refitted
    on the training folds of every cross-validation split.

    Parameters
    ----------
    scale:
        Whether to append Min-Max scaling. One-hot encoding already produces
        values in ``[0, 1]``; the step is kept for the scale-sensitive
        estimators so that the normalisation stage is explicit and so that the
        pipeline remains correct if continuous predictors are added later.
    """
    steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", make_one_hot_encoder()),
    ]
    if scale:
        steps.append(("scale", MinMaxScaler(feature_range=(0, 1))))
    return steps


def rank_feature_importance(
    features: pd.DataFrame,
    target: pd.Series,
    descriptions: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank predictors by Random Forest importance using training data only.

    One-hot encoding splits a questionnaire item into several indicator
    columns, so the importance of each item is recovered by summing the
    importances of the indicators it generated. The mapping from indicator to
    source variable is taken from the encoder categories rather than from the
    generated column names, which avoids parsing category labels that may
    themselves contain separators.

    Returns
    -------
    by_variable, by_indicator:
        Importances aggregated per original variable (sorted, with a
        ``selected`` flag) and the raw per-indicator importances.
    """
    imputer = SimpleImputer(strategy="most_frequent")
    imputed = pd.DataFrame(
        imputer.fit_transform(features),
        columns=features.columns,
        index=features.index,
    )

    encoder = make_one_hot_encoder()
    encoded = encoder.fit_transform(imputed)

    selector = RandomForestClassifier(
        n_estimators=config.SELECTOR_N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        class_weight="balanced",
        n_jobs=-1,
    )
    selector.fit(encoded, target)

    source_variables: list[str] = []
    for column, categories in zip(features.columns, encoder.categories_):
        source_variables.extend([column] * len(categories))

    by_indicator = pd.DataFrame(
        {
            "indicator": encoder.get_feature_names_out(features.columns),
            "source_variable": source_variables,
            "importance": selector.feature_importances_,
        }
    )
    if len(by_indicator) != len(source_variables):
        raise RuntimeError("Indicator columns could not be mapped to source variables")

    by_variable = (
        by_indicator.groupby("source_variable", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    by_variable["rank"] = by_variable.index + 1
    by_variable["selected"] = by_variable["rank"] <= config.N_SELECTED_FEATURES

    if descriptions:
        by_variable.insert(1, "description", by_variable["source_variable"].map(descriptions))
        by_indicator.insert(2, "description", by_indicator["source_variable"].map(descriptions))

    return by_variable, by_indicator


def selected_features(by_variable: pd.DataFrame) -> list[str]:
    """Extract the retained predictors, in decreasing order of importance."""
    return by_variable.loc[by_variable["selected"], "source_variable"].tolist()
