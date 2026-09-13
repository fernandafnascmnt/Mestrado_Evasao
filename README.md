# Predicting dropout risk in a Science and Technology undergraduate programme

Replication package for the article *Evaluating Machine Learning Algorithms for
Predicting Dropout Risk in a Science and Technology Course*. It contains the
survey dataset, the full experimental pipeline and the code that produces every
table and figure reported in the paper.

Seven classifier families — Perceptron, K-Nearest Neighbours, Naive Bayes,
Decision Tree, Random Forest, Multilayer Perceptron and Support Vector Machine —
are compared on their ability to identify, from personal and early academic
data, students at risk of leaving the programme.

## Experimental protocol

The design follows CRISP-DM and is fixed in
[`dropout_prediction/config.py`](dropout_prediction/config.py).

| | |
|---|---|
| Sample | 229 former students (184 completers, 45 dropouts) |
| Target | `var_0`, with dropout as the positive class |
| Excluded | `var_3` (semester of completion or dropout) is observable only after the outcome |
| Hold-out | 80/20 stratified, fixed seed: 183 training and 46 test records |
| Feature selection | Random Forest importance on the training partition; 15 variables retained |
| Preprocessing | Most-frequent imputation, one-hot encoding, Min-Max scaling for the scale-sensitive estimators — all inside the pipeline |
| Tuning | Grid search under stratified 10-fold cross-validation, repeated 50 times |
| Refit criterion | F1 of the dropout class |
| Reported metrics | Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix |
| Comparison | Friedman test followed by the Nemenyi post-hoc test, α = 0.05 |

Three decisions shape the numbers and are worth stating explicitly.

**Nothing is fitted on the test set.** The 20% partition is separated before
feature selection and never participates in encoding, imputation or tuning.
Imputation and encoding are pipeline steps, so they are refitted on the training
folds of every cross-validation split rather than once on the whole sample.

**One refit criterion for all metrics.** A single `GridSearchCV` per algorithm
and repetition selects one configuration by F1. Every metric reported for that
algorithm in that repetition therefore comes from the same fitted model, and the
confusion matrix is consistent with the Precision and Recall beside it.

**The minority class drives the evaluation.** Dropout is the positive class and
F1 is the tuning criterion. Class weighting is part of the search space wherever
the estimator supports it, so weighted and unweighted fitting compete on equal
terms instead of one being imposed. Accuracy is reported but should not be read
in isolation: a classifier that predicts the majority class almost everywhere
reaches roughly 78% accuracy on this sample while detecting almost no dropouts.

Interquartile-range filtering is deliberately not applied. Every predictor is a
categorised questionnaire answer whose stored value is a category code, so its
quartiles carry no meaning; each run records this in `outlier_handling.md`.

## Installation

Python 3.10 or later.

```bash
git clone https://github.com/fernandafnascmnt/Mestrado_Evasao.git
cd Mestrado_Evasao/dropout-prediction
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

On Windows, activate the environment with `.venv\Scripts\activate` in
PowerShell and use `py` in place of `python` if the launcher is installed.
`make` is not required: every target in the `Makefile` is a single command that
can also be typed directly.

## Running the experiment

Verify the installation with a single reduced repetition, which takes a few
seconds:

```bash
python -m dropout_prediction --output results_check --reduced
```

Then run the full protocol:

```bash
python -m dropout_prediction --output results
```

Fifty repetitions of a ten-fold grid search over seven algorithms take roughly
forty minutes on a recent desktop with every core in use, and proportionally
longer on fewer cores. `per_repetition_results.csv` is rewritten after each
model, so the run can be inspected while it is still in progress.

Useful options: `--repetitions`, `--cv-folds`, `--jobs` (`-1` uses every core),
`--no-figures`, `--quiet`. On Unix-like systems, `make check`, `make experiment`
and `make figures` wrap the same commands.

Results obtained with `--reduced` collapse each grid to one configuration and
must not be reported.

## Repository layout

```
dropout_prediction/
    config.py       protocol constants: target, split, seeds, metrics
    data.py         loading, class mapping, missing-value normalisation
    features.py     preprocessing steps and training-only importance ranking
    models.py       the seven pipelines and their search spaces
    evaluation.py   scoring on the hold-out set and aggregation over repetitions
    statistics.py   Friedman test, average ranks, Nemenyi post-hoc
    experiment.py   orchestration and artefact writing
    figures.py      the manuscript figures
data/
    survey_responses.csv   anonymised questionnaire responses, semicolon-separated
    codebook.csv           variable names, descriptions and roles
```

## Output artefacts

| File | Content |
|---|---|
| `run_configuration.json` | Protocol, sample sizes, selected predictors, library versions |
| `selected_features.csv` | The 15 retained variables with their importances |
| `feature_importance_by_variable.csv` | Full ranking aggregated per questionnaire item |
| `feature_importance_by_indicator.csv` | Importance of each one-hot indicator |
| `per_repetition_results.csv` | One row per repetition and algorithm, with all metrics and the chosen configuration |
| `metric_summary.csv` | Mean, standard deviation and range per algorithm |
| `best_params_frequency.csv` | How often each configuration was selected |
| `confusion_counts_*.csv`, `confusion_normalized_*.csv` | Confusion matrices aggregated over the repetitions |
| `friedman_*.txt`, `average_ranks_*.csv`, `nemenyi_*.csv` | Omnibus test, mean ranks and pairwise comparisons |
| `rank_differences_*.csv`, `significance_matrix_*.csv` | The square matrices reproduced as Tables II and III |
| `outlier_handling.md`, `runtime.txt` | Methodological note and wall-clock time |
| `figures/` | Figures 2 and 4–7 in PNG and PDF |

F1 is the primary criterion of the statistical comparison; the same tests are
also computed on Accuracy and written with the `_accuracy` suffix.

## Reproducibility

The outer split, the importance estimator and every classifier use fixed seeds,
and the cross-validation seed is derived from the repetition index, so a rerun on
the same library versions reproduces the reported numbers exactly. The versions
of the original run are recorded in `run_configuration.json`.

## Data

`data/survey_responses.csv` holds anonymised responses from former students who
completed or left the programme, collected through a voluntary electronic
questionnaire covering personal, academic, demographic and socioeconomic items.
No field identifies a respondent. Labels and category values are in Portuguese,
as collected; `data/codebook.csv` gives English and Portuguese descriptions for
each variable.

## Citation

See [`CITATION.cff`](CITATION.cff). Please cite the article rather than the
repository alone.

## License

Code released under the MIT License (see [`LICENSE`](LICENSE)).
