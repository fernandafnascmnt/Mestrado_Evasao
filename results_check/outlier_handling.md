# Outlier handling

Interquartile-range filtering is not applied in this experiment.

All predictors are categorised questionnaire answers. Their stored values are category codes, not measurements on an interval scale, so the quartiles of such a column carry no meaning and filtering by them would discard valid responses whose only peculiarity is belonging to an infrequent category.

Should a continuous predictor be added to the dataset, outlier treatment should be reconsidered for that variable alone.
