"""Loading and basic cleaning of the survey dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from . import config

N_SURVEY_COLUMNS = 32
"""Columns ``var_0``..``var_31``.

The export carries a few trailing administrative columns that are not part of
the questionnaire; they are discarded on load.
"""


def load_codebook(path: Path) -> pd.DataFrame:
    """Read the variable codebook indexed by variable name."""
    return pd.read_csv(path).set_index("variable")


def variable_descriptions(codebook: pd.DataFrame, language: str = "en") -> dict[str, str]:
    """Return a ``{variable: description}`` mapping in the requested language."""
    column = f"description_{language}"
    if column not in codebook.columns:
        raise ValueError(f"Codebook has no column {column!r}")
    return codebook[column].to_dict()


def load_dataset(path: Path) -> tuple[pd.DataFrame, pd.Series]:
    """Load the survey export and return the predictor matrix and the target.

    Missing-value markers are converted to ``NaN`` here; imputation itself is
    deferred to the modelling pipeline so that it is fitted on training data
    only.
    """
    frame = pd.read_csv(path, sep=";").iloc[:, :N_SURVEY_COLUMNS].copy()

    expected = [f"var_{i}" for i in range(N_SURVEY_COLUMNS)]
    missing = [column for column in expected if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing the columns {missing}")

    target = frame[config.TARGET].map(config.CLASS_MAPPING)
    if target.isna().any():
        unknown = frame.loc[target.isna(), config.TARGET].drop_duplicates().tolist()
        raise ValueError(f"Unrecognised class labels in {config.TARGET}: {unknown}")

    features = frame.drop(columns=[config.TARGET, *config.EXCLUDED_FEATURES])
    features = features.apply(_normalise_missing)

    return features, target.astype(int)


def _normalise_missing(column: pd.Series) -> pd.Series:
    """Strip whitespace and map missing-value markers to ``NaN``."""

    def clean(value: object) -> object:
        if pd.isna(value):
            return np.nan
        text = str(value).strip()
        return np.nan if text in config.MISSING_TOKENS else text

    return column.map(clean)
