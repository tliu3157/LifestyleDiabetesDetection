# Predicting Diabetes from NHANES Data

This project was an AP Research project that tests whether machine-learning and deep learning models could predict diabetes from lifestyle, diet, demographic, and medical-history survey data in the U.S. National Health and Nutrition Examination Survey (NHANES).

Each notebook trains a different model or a different version of the pipeline: a TabTransformer, a multilayer perceptron (MLP), a random forest, and XGBoost. A random-forest surrogate model is used to explain which features the neural networks rely on.

## Repository layout

```
.
├── NHANES DATA/                      # Raw NHANES .xpt files (see "Data" below)
├── TabTransformer.ipynb              # First TabTransformer, 2021–2023 data only
├── TabTransformerNoSMOTE.ipynb       # Same, without SMOTE oversampling (baseline)
├── TabTransfomerPart2.ipynb          # TabTransformer on all four NHANES cycles
├── TabTransformerNewFeatures.ipynb   # Adds medical, insurance, blood pressure, and waist features
├── MLP.ipynb                         # Feed-forward neural network on the full feature set
├── RandomForest.ipynb                # Random forest and XGBoost on the full feature set
└── README.md
```

## Data

All data comes from [NHANES](https://wwwn.cdc.gov/nchs/nhanes/), run by the CDC's National Center for Health Statistics. The notebooks use four survey cycles: 2013–2014, 2015–2016, 2017–March 2020 (pre-pandemic), and August 2021–August 2023.

The files were downloaded as SAS transport (`.xpt`) files and renamed to `<component>_<years>.xpt`, for example `glucose_2015-2016.xpt`. Put them all in a folder called `NHANES DATA` next to the notebooks.

| File prefix | NHANES component | Variables used |
|---|---|---|
| `glucose` | Plasma Fasting Glucose (GLU) | `LBDGLUSI` (mmol/L), used for the diabetes label |
| `demographic` | Demographics (DEMO) | Gender, age, ethnicity, income-to-poverty ratio |
| `diet1`, `diet2` | Dietary Interview, Total Nutrient Intakes, days 1 and 2 (DR1TOT, DR2TOT) | Calories, protein, carbs, sugar, fiber, fat, cholesterol |
| `smoking` | Smoking (SMQ) | Ever smoked 100 cigarettes |
| `alcohol` | Alcohol Use (ALQ) | Ever had alcohol |
| `physical` | Physical Activity (PAQ) | Minutes of sedentary activity |
| `weight` | Weight History (WHQ) | Self-reported height and weight, used for BMI |
| `sleep` | Sleep Disorders (SLQ) | Hours of sleep (2015 onward) |
| `body` | Body Measures (BMX) | Waist circumference |
| `medical` | Medical Conditions (MCQ) | Family history of diabetes, doctor advice, heart disease |
| `insurance` | Health Insurance (HIQ) | Covered by health insurance |
| `bloodpressure` | Blood Pressure & Cholesterol (BPQ) | High blood pressure, high cholesterol, medications |

Where NHANES renamed a variable between cycles (for example `ALQ110` became `ALQ111`, and `MCQ365*` became `MCQ366*`), the notebooks rename the old column so the cycles line up.

## Pipeline

All the notebooks follow roughly the same steps:

1. **Load and merge.** Each NHANES component is read with `pd.read_sas`, the cycles are stacked, and the components are joined on the participant ID (`SEQN`).
2. **Label.** A participant is labeled diabetic if fasting plasma glucose is above 7.0 mmol/L, the WHO fasting threshold. `TabTransfomerPart2` uses 11.1 mmol/L instead.
3. **Feature engineering.** BMI is calculated from height and weight, and nutrient intake is averaged across the two diet-recall days.
4. **Cleaning.** Rows with too many missing values are dropped. Remaining gaps are filled with the mean for numerical features and the most frequent value for categorical ones.
5. **Preprocessing.** Numerical features are standardized and categorical features are label-encoded.
6. **Split and rebalance.** The data is split 80/20 into stratified training and test sets. Only about 12–13% of participants are diabetic, so SMOTE-NC oversamples the minority class in the training set only.
7. **Train and evaluate.** Each model reports accuracy, precision, recall/sensitivity, specificity, F1, a confusion matrix, and, where applicable, a ROC curve and AUC.
8. **Interpret.** A random forest is trained to imitate the neural network's predictions, and its feature importances show which features drive those predictions.

## Models

| Notebook | Data | Model | Notes |
|---|---|---|---|
| `TabTransformer` | 2021–2023 | Transformer over a linear embedding of all features | First attempt |
| `TabTransformerNoSMOTE` | 2021–2023 | TabTransformer (per-feature categorical embeddings and a transformer encoder, followed by an MLP head) | No oversampling, as a baseline |
| `TabTransfomerPart2` | 2013–2023 | TabTransformer | SMOTE-NC at a 0.75 ratio, plus a surrogate random forest |
| `TabTransformerNewFeatures` | 2013–2023 | TabTransformer, mini-batches, LR warm-up, early stopping | Full feature set, decision threshold 0.67 |
| `MLP` | 2013–2023 | Four-layer MLP (128 → 64 → 32 → 1) | Full feature set |
| `RandomForest` | 2013–2023 | Random forest and XGBoost | Full feature set |

## Results

These numbers are from the outputs saved in each notebook when it was last run, on the held-out test set.

| Model | Accuracy | Sensitivity (diabetic recall) | Specificity | Diabetic-class F1 | AUC |
|---|---|---|---|---|---|
| TabTransformer (new features) | 0.84 | 0.43 | 0.89 | 0.38 | – |
| XGBoost | 0.84 | 0.36 | 0.91 | 0.37 | – |
| MLP | 0.78 | 0.41 | 0.84 | 0.33 | 0.74 |
| TabTransformer (first attempt) | 0.57 | 0.77 | – | 0.30 | – |

Because diabetic participants are a small minority, accuracy on its own is misleading. Sensitivity and diabetic-class F1 are the more useful measures.

## Running the notebooks

1. Install Python 3.11 and the dependencies:

   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn torch xgboost matplotlib seaborn jupyter
   ```

2. Make sure the `NHANES DATA` folder is in the same folder as the notebooks. The notebooks use relative paths, so nothing else needs to change.
3. Open a notebook in Jupyter or VS Code and run the cells in order.

If the data folder can't be found, the second cell of each notebook stops with an error that shows where it looked.

## Known issues

- **Alcohol feature.** In `MLP`, `RandomForest`, `TabTransfomerPart2`, and `TabTransformerNewFeatures`, the line `alcohol_df = pd.concat(sleep_dataframes, ...)` should use `alcohol_dataframes`. As written, the "ever had alcohol" column contains sleep data, so the results above include that error.
- **Feature names in importance plots.** In `MLP` and `RandomForest`, the feature names attached to the importances are not in the same order as the columns the model was trained on (`cat_columns + num_columns`), so the labels on those charts may be mismatched.
- **Label comment.** The code comment "2 hour fasting period" does not match the variable used. `LBDGLUSI` is fasting plasma glucose, for which 7.0 mmol/L is the standard diabetes threshold. The 11.1 mmol/L threshold is for a 2-hour glucose tolerance test.
