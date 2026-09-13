# Data

## `survey_responses.csv`

Anonymised responses from 229 former students of the Science and Technology
undergraduate programme: 184 who completed it and 45 who left. Data were
collected through a voluntary electronic questionnaire covering personal,
academic, demographic and socioeconomic items. No field identifies a
respondent.

Format: semicolon-separated, UTF-8 with BOM, CRLF line endings. Columns
`var_0`..`var_31` are the questionnaire items; any trailing column is
administrative and is discarded on load.

`var_0` is the class label, taking the values `Concludente/Egresso`
(completer) and `Evadido/Desistente` (dropout). Unanswered items are marked
`NULO` and converted to missing values at load time; imputation happens inside
the modelling pipeline.

Category values are recorded in Portuguese, as collected.

## `codebook.csv`

One row per variable, with the English and Portuguese descriptions and the role
of the variable in the experiment:

- `target` — the class label.
- `excluded (post-outcome information)` — `var_3`, the semester of completion or
  dropout, which is only observable after the outcome and is dropped before any
  modelling step.
- `predictor` — available as input to the feature-selection stage.
