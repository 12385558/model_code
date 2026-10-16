# Dietary and Questionnaire Model (DQM): 10-fold cross-validation and temporal validation

This repository contains the **R modeling and validation code** for the 30-predictor Dietary and Questionnaire Model (DQM) in NHANES 2007–2018. Selected predictors originate from upstream stable LASSO screening.

## Analysis design

- Development: 2007–2014; temporal validation: 2015–2018.
- Three previously multiply imputed datasets per period (`m=3`). The same person appears in the same row of each imputed dataset.
- **Shared stratified 10-fold cross-validation** partitions are generated on the development outcome and reused across all three imputations.
- The five tuned algorithms are backpropagation neural network (BP, `nnet`), linear support vector machine (SVM, `LiblineaR`), random forest (RF, `ranger`), CatBoost and deep neural network (DNN, `h2o`). Logistic regression (LR) is included as a baseline without hyperparameter tuning.
- Training-fold-only SMOTENC and other recipe steps are fitted inside each fold; predictions on held-out folds and the 2015–2018 temporal validation cohort use untouched subject labels and distributions.
- Selected hyperparameters maximize the mean fold AUROC across the three imputations. Final per-imputation validation probabilities are averaged per subject.
- SVM Platt scaling uses development-only 10-fold out-of-fold (OOF) decision scores. Youden cutoffs are derived from development OOF predictions and applied unchanged to temporal validation.
- Time-validation metrics include AUROC (DeLong CI), PR-AUC and threshold-based metrics with bootstrap CIs, paired exploratory DeLong comparisons with Holm adjustment, decision curve analysis and six separate calibration plots with quantitative calibration measures.
- Optional RF permutation importance (`RUN_RF_PERMUTATION <- TRUE`) uses the final RF models on the temporal validation cohort (50 permutations per predictor, coordinated across the three imputed datasets). This step is disabled by default because it is computationally intensive.

**Outcome definition:** `MCQ160C` in the input files is a *study-defined composite CHD outcome*, combining self-reported physician diagnoses of coronary heart disease, angina and myocardial infarction. It is **not** simply the original individual NHANES variable `MCQ160C`.

## Input files

Place the following in your input directory:

```
Development_MI_1.csv
Development_MI_2.csv
Development_MI_3.csv
Validation_MI_1.csv
Validation_MI_2.csv
Validation_MI_3.csv
DQM_final_variables.csv
```

`DQM_final_variables.csv` must have a `Variable` column listing exactly **30** final predictor codes. Every imputed dataset must contain `SEQN`, the 30 selected predictors, and the binary study outcome `MCQ160C` (`0` = no CHD; `1` = CHD). Rows and outcome labels must be aligned across imputations for each period, and modeled data must contain no missing values.

The input datasets and LASSO results are produced by *upstream* cohort selection, imputation and feature screening, which are not reproduced in this script. Study assumptions such as predictor coding must match those preprocessing files.

## Run

Install the following R packages before running: `caret`, `recipes`, `themis`, `pROC`, `PRROC`, `dplyr`, `ggplot2`, `nnet`, `LiblineaR`, `ranger`, `catboost`, `h2o`, `dcurves`. `grid` is in base R. H2O requires a compatible Java environment; `h2o.init(max_mem_size="8G")` is used by default.

Run from a project directory containing the CSV input files:

```r
source("DQM_Model_10fold.R")
```

Alternatively, provide input/output locations via environment variables before sourcing:

```r
Sys.setenv(NHANES_INPUT_DIR = "path/to/input", NHANES_OUTPUT_DIR = "path/to/outputs")
source("DQM_Model_10fold.R")
```

The script saves all intermediate and final files under `outputs/` by default. This includes CV parameter tables, final model objects and validation predictions, 10-fold OOF predictions, Youden thresholds, performance tables, DeLong tests, ROC, DCA and calibration TIFFs, and `sessionInfo.txt`.

To run the optional RF permutation-importance step, set `RUN_RF_PERMUTATION <- TRUE` near the top of the script and clear its checkpoint folder if you have changed the underlying models or data.

## Methods and reproducibility notes

- Class balancing during model fitting can affect raw probability calibration. Temporal validation calibration should be reported using the unbalanced validation cohort.
- The SVM Platt model is fitted to OOF scores, but Platt-predicted development probabilities are not independently cross-fitted for the calibration layer. For completely nested/unbiased selection, an outer cross-validation layer is needed.
- The paired DeLong tests comparing the **numerically best temporal-validation** algorithm with the other five are exploratory: ranking and testing on the same temporal-validation set creates a selection effect. The tests do not establish that the selected model was pre-specified.
- The reproducible random seed is `2026`; running the revised 10-fold workflow may yield different optimal hyperparameters, thresholds, AUROCs, and classification metrics than the original two-fold results.
- The code has undergone static structural checks, but has **not been executed** with the original input CSVs in R in this environment.
- Neither raw NHANES data nor intermediate input CSVs are distributed in this repository; obtain them through appropriate sources or run the upstream preprocessing pipeline.
