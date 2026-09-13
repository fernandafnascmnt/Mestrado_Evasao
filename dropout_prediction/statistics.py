"""Friedman omnibus test and Nemenyi post-hoc comparison.

The procedure follows Demsar's protocol for comparing several classifiers over
multiple runs: algorithms are ranked within each repetition, the Friedman test
checks whether the mean ranks differ, and the Nemenyi test identifies which
pairs differ once the null hypothesis is rejected.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, studentized_range

from . import config


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of the omnibus test and the pairwise comparisons."""

    metric: str
    statistic: float
    p_value: float
    n_repetitions: int
    n_models: int
    average_ranks: pd.Series
    pairwise: pd.DataFrame
    critical_distance: float

    def as_text(self) -> str:
        return (
            f"metric: {self.metric}\n"
            f"Friedman statistic: {self.statistic:.6f}\n"
            f"p-value: {self.p_value:.6e}\n"
            f"repetitions (N): {self.n_repetitions}\n"
            f"models (k): {self.n_models}\n"
            f"critical distance (alpha={config.ALPHA}): {self.critical_distance:.4f}\n"
        )


def compare_models(results: pd.DataFrame, metric: str) -> ComparisonResult | None:
    """Rank the algorithms on ``metric`` and test the differences.

    Returns ``None`` when there are too few repetitions or algorithms for the
    tests to be defined.
    """
    scores = results.pivot(index="repetition", columns="model", values=metric)
    n_repetitions, n_models = scores.shape
    if n_repetitions < 2 or n_models < 3:
        return None

    statistic, p_value = friedmanchisquare(*(scores[column].values for column in scores.columns))

    ranks = scores.apply(
        lambda row: rankdata(-row.values, method="average"), axis=1, result_type="expand"
    )
    ranks.columns = scores.columns
    average_ranks = ranks.mean(axis=0).sort_values().rename("average_rank")

    pairwise, critical_distance = _nemenyi(average_ranks, n_models, n_repetitions)

    return ComparisonResult(
        metric=metric,
        statistic=float(statistic),
        p_value=float(p_value),
        n_repetitions=int(n_repetitions),
        n_models=int(n_models),
        average_ranks=average_ranks,
        pairwise=pairwise,
        critical_distance=critical_distance,
    )


def _nemenyi(
    average_ranks: pd.Series, n_models: int, n_repetitions: int
) -> tuple[pd.DataFrame, float]:
    """Pairwise Nemenyi comparisons and the critical distance.

    Two algorithms differ significantly when the absolute difference between
    their mean ranks exceeds ``CD = q_alpha * sqrt(k(k+1) / 6N)``, where
    ``q_alpha`` is the studentized range statistic divided by ``sqrt(2)``.
    """
    standard_error = np.sqrt(n_models * (n_models + 1) / (6.0 * n_repetitions))
    q_alpha = studentized_range.ppf(1 - config.ALPHA, n_models, np.inf) / np.sqrt(2.0)
    critical_distance = float(q_alpha * standard_error)

    models = list(average_ranks.index)
    rows = []
    for i, first in enumerate(models):
        for second in models[i + 1 :]:
            difference = abs(float(average_ranks[first] - average_ranks[second]))
            q_statistic = (difference / standard_error) * np.sqrt(2.0)
            p_value = float(studentized_range.sf(q_statistic, n_models, np.inf))
            rows.append(
                {
                    "model_a": first,
                    "model_b": second,
                    "rank_difference": difference,
                    "p_value": p_value,
                    "significant": p_value < config.ALPHA,
                }
            )
    return pd.DataFrame(rows), critical_distance


def rank_difference_matrix(average_ranks: pd.Series, order: list[str]) -> pd.DataFrame:
    """Square matrix of absolute rank differences, for tabulation."""
    values = np.abs(
        average_ranks[order].values[:, None] - average_ranks[order].values[None, :]
    )
    return pd.DataFrame(values, index=order, columns=order)


def significance_matrix(pairwise: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """Square boolean matrix of significant pairwise differences."""
    matrix = pd.DataFrame(False, index=order, columns=order)
    for row in pairwise.itertuples():
        matrix.loc[row.model_a, row.model_b] = bool(row.significant)
        matrix.loc[row.model_b, row.model_a] = bool(row.significant)
    return matrix
