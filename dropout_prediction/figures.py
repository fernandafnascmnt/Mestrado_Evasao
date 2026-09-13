"""Publication-quality figures built from the experiment artefacts.

Figures are produced at the size they occupy in the manuscript so that the
font sizes are correct without rescaling, and are written both as raster
images and as vector graphics.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config

BAR_COLOR = "#3B6FA0"
BAR_EDGE = "#22405C"

STYLE = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.7,
    "axes.edgecolor": "0.25",
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
}


def build_all(results_dir: Path, figures_dir: Path | None = None) -> Path:
    """Regenerate every figure from the artefacts in ``results_dir``."""
    figures_dir = figures_dir or results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(STYLE):
        importance = pd.read_csv(results_dir / "selected_features.csv")
        _feature_importance(importance, figures_dir / "fig2_feature_importance")

        summary = _ordered_summary(results_dir / "metric_summary.csv")
        _metric_pair(
            summary,
            [("recall", "Recall"), ("precision", "Precision")],
            figures_dir / "fig4_recall_precision",
        )
        _single_metric(summary, "f1", "F1 Score", figures_dir / "fig5_f1_score")
        _metric_pair(
            summary,
            [("roc_auc", "ROC_AUC"), ("accuracy", "Accuracy")],
            figures_dir / "fig6_rocauc_accuracy",
            upper_limit=100,
        )

        best = summary.index[0]
        matrix = pd.read_csv(
            results_dir / f"confusion_normalized_{best.replace(' ', '_')}.csv", index_col=0
        )
        _confusion_matrix(matrix, figures_dir / "fig7_confusion_matrix")

    return figures_dir


def _ordered_summary(path: Path) -> pd.DataFrame:
    """Metric summary indexed by model, in the order used by the tables."""
    summary = pd.read_csv(path).set_index("model")
    ordered = [name for name in config.MODEL_NAMES if name in summary.index]
    summary = summary.loc[ordered]
    summary.attrs["best_by_f1"] = summary["f1_mean"].idxmax()
    return summary


def _save(figure: plt.Figure, stem: Path) -> None:
    figure.savefig(stem.with_suffix(".png"))
    figure.savefig(stem.with_suffix(".pdf"))
    plt.close(figure)


def _bar_panel(
    axes: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    label: str,
    font_size: float = 7.5,
    upper_limit: float | None = None,
) -> None:
    """Bars with standard-deviation whiskers and values placed above them."""
    means = summary[f"{metric}_mean"].values * 100
    deviations = summary[f"{metric}_std"].values * 100
    positions = np.arange(len(summary))

    axes.bar(
        positions, means, width=0.68, color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=0.6, zorder=3
    )
    axes.errorbar(
        positions,
        means,
        yerr=deviations,
        fmt="none",
        ecolor="0.15",
        elinewidth=0.7,
        capsize=1.8,
        capthick=0.7,
        zorder=4,
    )

    top = upper_limit or min(100, (means + deviations).max() * 1.28)
    for position, mean, deviation in zip(positions, means, deviations):
        axes.text(
            position,
            mean + deviation + top * 0.025,
            f"{mean:.1f}",
            ha="center",
            va="bottom",
            fontsize=font_size - 1.0,
            zorder=5,
        )

    axes.set_xticks(positions)
    axes.set_xticklabels(
        [config.MODEL_ABBREVIATIONS[name] for name in summary.index], fontsize=font_size
    )
    axes.set_ylim(0, top)
    axes.set_ylabel(f"{label} (%)", fontsize=font_size)
    axes.tick_params(axis="y", labelsize=font_size - 0.5)
    axes.yaxis.grid(True, color="0.88", linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)


def _metric_pair(
    summary: pd.DataFrame,
    metrics: list[tuple[str, str]],
    stem: Path,
    upper_limit: float | None = None,
) -> None:
    figure, panels = plt.subplots(1, 2, figsize=(4.8, 1.95))
    for panel, (metric, label), tag in zip(panels, metrics, ["(a)", "(b)"]):
        _bar_panel(panel, summary, metric, label, upper_limit=upper_limit)
        panel.set_xlabel(tag, fontsize=7.5, labelpad=1.5)
    figure.tight_layout(w_pad=1.6)
    _save(figure, stem)


def _single_metric(summary: pd.DataFrame, metric: str, label: str, stem: Path) -> None:
    figure, axes = plt.subplots(figsize=(3.6, 2.3))
    _bar_panel(axes, summary, metric, label, font_size=8)
    figure.tight_layout()
    _save(figure, stem)


def _feature_importance(importance: pd.DataFrame, stem: Path) -> None:
    frame = importance.sort_values("importance")
    if "description" in frame.columns:
        labels = frame["description"] + " (" + frame["source_variable"] + ")"
    else:
        labels = frame["source_variable"]

    figure, axes = plt.subplots(figsize=(4.7, 3.1))
    positions = np.arange(len(frame))
    axes.barh(
        positions,
        frame["importance"].values,
        height=0.68,
        color=BAR_COLOR,
        edgecolor=BAR_EDGE,
        linewidth=0.6,
        zorder=3,
    )
    for position, value in zip(positions, frame["importance"].values):
        axes.text(value + 0.0016, position, f"{value:.4f}", va="center", ha="left", fontsize=6.2)

    axes.set_yticks(positions)
    axes.set_yticklabels(labels, fontsize=6.8)
    axes.set_xlabel("Aggregated feature importance", fontsize=7.5)
    axes.tick_params(axis="x", labelsize=7)
    axes.set_xlim(0, frame["importance"].max() * 1.20)
    axes.xaxis.grid(True, color="0.88", linewidth=0.6, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    _save(figure, stem)


def _confusion_matrix(matrix: pd.DataFrame, stem: Path) -> None:
    values = matrix.values
    labels = [config.CLASS_NAMES[0].capitalize(), config.CLASS_NAMES[1].capitalize()]

    figure, axes = plt.subplots(figsize=(2.9, 2.4))
    image = axes.imshow(values, cmap="Blues", vmin=0, vmax=1)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            axes.text(
                column,
                row,
                f"{value:.4f}\n({value * 100:.2f}%)",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.55 else "0.1",
            )

    axes.set_xticks(range(len(labels)), labels=labels, fontsize=8)
    axes.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    axes.set_xlabel("Predicted class", fontsize=8.5)
    axes.set_ylabel("True class", fontsize=8.5)
    axes.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
    axes.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    axes.grid(which="minor", color="white", linewidth=1.4)
    axes.tick_params(which="minor", length=0)

    colorbar = figure.colorbar(image, ax=axes, fraction=0.046, pad=0.03)
    colorbar.ax.tick_params(labelsize=7)
    colorbar.outline.set_linewidth(0.6)
    figure.tight_layout()
    _save(figure, stem)
