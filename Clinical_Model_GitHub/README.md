# Clinical Characteristics Model
This repository provides the **R modeling pipeline** for identifying prevalent coronary heart disease (CHD) from NHANES 2007–2018 data, using a 32-predictor Clinical Characteristics Model selected upstream with LASSO.

## Workflow

1. Read three aligned, previously imputed development and temporal validation datasets, and the 32 selected predictors.
2. Define common **stratified 10-fold cross-validation** partitions on development data (2007–2014). The same participant indices are used in each imputation.
3. For each hyperparameter configuration, train on each development training fold using SMOTENC, normalization, and/or dummy encoding as appropriate. Validation folds are never oversampled. Choose hyperparameters using mean fold-wise AUROC across all 10 folds and all 3 imputations.
4. Fit BP, linear SVM, random forest, CatBoost, logistic regression baseline, and DNN on the complete development data for each imputation. Average three predictions per person for independent temporal validation (2015–2018).
5. Use 10-fold out-of-fold predictions on development data for Youden thresholds. For SVM, fit a logistic Platt calibrator to development OOF decision scores.
6. Evaluate independent temporal validation AUROC with DeLong confidence intervals, PR-AUC and classification metrics with bootstrap confidence intervals, calibration, and decision curve analysis (DCA). Optional SVM permutation importance is available.

**Outcome encoding:** The `MCQ160C` column in these processed files denotes the study-defined composite CHD outcome (reported physician diagnosis of CHD, angina, or myocardial infarction). It is not simply the original NHANES `MCQ160C` questionnaire item.

## Inputs (place all in your input folder)

```
Development_MI_1.csv
Development_MI_2.csv
Development_MI_3.csv
Validation_MI_1.csv
Validation_MI_2.csv
Validation_MI_3.csv
Clinical_final_variables.csv
```

`Clinical_final_variables.csv` must contain a `Variable` column listing **exactly 32** predictors. Each MI file must contain `SEQN`, these 32 predictors, and `MCQ160C` encoded as `0` (non-CHD) and `1` (CHD). The three imputations must have identical row order within each period, without missing values in modeled columns. The script checks the selected names and participant ID alignment. The file inputs are processed outputs from upstream cohort construction, multiple imputation and LASSO screening; upstream scripts are not included here.

## Requirements

R packages: `caret`, `recipes`, `themis`, `pROC`, `PRROC`, `dplyr`, `ggplot2`, `nnet`, `LiblineaR`, `ranger`, `catboost`, `h2o`, `dcurves`, `grid` (base R). H2O uses up to 8 GB of Java heap in the script, so ensure suitable Java/H2O installation. CatBoost may need installation following its official R installation documentation.

## How to run

Place the CSVs alongside the script and run:

```r
source("Clinical_Model_10fold.R")
```

By default the script reads from the current working directory and writes to `outputs/`. You may set paths without changing source:

```r
Sys.setenv(NHANES_INPUT_DIR = "F:/revise", NHANES_OUTPUT_DIR = "F:/revise/Clinical_results_10fold")
source("Clinical_Model_10fold.R")
```

Key controls at the start of the script: `CV_FOLDS = 10L`, `SEED = 2026L`, `N_BOOT = 2000L`, and `RUN_SVM_PERMUTATION = FALSE`. Set the latter to `TRUE` to execute SVM permutation importance (50 shuffles per variable); this step is computationally intensive.

## Outputs

Results are saved to `outputs/` (or the specified output directory). Key files:

- `Clinical_*_CV_results.csv` and `Clinical_*_bestTune.csv`: cross-validated hyperparameter selection.
- `Clinical_Development_10fold_OOF_probabilities.rds` and `Clinical_Development_10fold_Thresholds.csv`: development OOF probabilities and thresholds.
- `Clinical_TemporalValidation_AUROC.csv`, `Clinical_AllModels_Performance.csv`, `Clinical_AllModels_Confusion_Matrix.csv`: independent temporal validation metrics.
- `Clinical_BestModel_vs_OtherModels_DeLong_Holm.csv`: exploratory paired DeLong comparisons with Holm adjustment.
- `Clinical_Model_ROC.tiff`, `Clinical_AllModels_DCA.tiff`, `Clinical_AllModels_Confusion_Matrix.tiff`: main figures.
- `Clinical_AllModels_Calibration_Statistics.csv`, `Clinical_AllModels_Calibration_Curve_Data.csv`, `Clinical_Calibration_Plots/`: calibration statistics and one plot per algorithm.
- `Clinical_*_results.RData`, DNN model folder, and `Clinical_MODEL_SELECTION_RESULTS.RData`: saved model/prediction objects for reproducibility.

## Methodological cautions

- **Do not reuse two-fold CV outputs.** Ten-fold model tuning, OOF probabilities, Youden thresholds, and temporal validation results must be rerun; values in older manuscript drafts do not update automatically.
- **Upstream processing:** Pre-selected variables and multiple-imputation values were generated before this script. If those steps involved using the full development dataset, cross-validation estimates for hyperparameter comparison can be optimistic; the independent temporal-validation results should remain the main performance evidence. The script does not claim that feature selection or multiple imputation was re-fitted within each CV fold.
- **Platt calibration caveat:** SVM Platt calibration is trained on development OOF scores, then the same calibrated development probabilities are used for Youden cutoff estimation. The calibration stage is not independently cross-fitted.
- **Categorical coding:** Age, income-to-poverty ratio, BMI, and pulse regularity are treated as factors to preserve the supplied preprocessing specification. Confirm that the data and manuscript describe this exact encoding.
- **Best-model comparisons:** Selecting the numerically best algorithm and then testing differences on the same temporal-validation cohort makes the post-selection P-values exploratory, not confirmatory.
- **Permutation importance:** If enabled, shuffling predictors in the temporal-validation cohort is a *post hoc* interpretation analysis, not part of model training. Results are specific to the SVM regardless of which algorithm has highest AUROC.
- **Reproducibility:** `SEED = 2026` improves reproducibility, but package versions, machine configuration and multithreaded routines may affect results. The script saves `sessionInfo.txt`.

Only the code is supplied. Patient-level NHANES extracts, imputed datasets, trained model binaries, and generated results are excluded from this repository. Check NHANES data sharing and privacy conditions before distributing any data.
