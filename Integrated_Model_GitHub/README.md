# Integrated Model — CHD Identification (NHANES 2007–2018)

This repository provides the R modeling and evaluation code for the **Integrated Model** in a study of coronary heart disease (CHD) identification using multidimensional NHANES health data.

**Study design:** temporal split into a development cohort (2007–2014) and an independent temporal validation cohort (2015–2018). The script fits six algorithms: backpropagation neural network (BP), linear support vector machine (SVM), random forest (RF), CatBoost, logistic regression (LR), and deep neural network (DNN).

## Cross-validation and validation design

- **Ten-fold stratified cross-validation** is used within the development cohort. Folds are assigned using the first imputed development dataset and reused across all three multiple-imputation datasets. Participants remain in the same fold across imputations.
- BP, SVM, RF, CatBoost, and DNN hyperparameters are chosen by **mean cross-validation AUROC**, calculated over the 10 folds and 3 imputed datasets. LR is an untuned comparator.
- In each training fold, preprocessing is fitted on training rows only. **SMOTENC is applied only to training rows**, never the held-out fold or temporal validation data. Normalization and categorical encoding are likewise fitted on development/training data.
- With hyperparameters selected, one final model is trained on the full development cohort **for each imputed dataset**. The three predicted probabilities are **averaged** to obtain temporal-validation predictions. This is prediction averaging, **not** Rubin's-rule pooling of model parameters.
- Development out-of-fold (OOF) predictions are used for **Youden threshold selection**. The selected thresholds are then applied without retuning to the temporal validation cohort.
- SVM uses **Platt calibration fitted to ten-fold OOF development decision scores**. Its development Platt probabilities use the fitted calibrator on those same OOF scores; the calibrator itself is **not nested or cross-fitted**.
- The script evaluates temporal-validation AUROC, PR-AUC, confusion-matrix statistics, bootstrap confidence intervals, calibration, decision-curve analysis, and RF permutation importance.

**Important:** The `MCQ160C` outcome column in these study-specific input files denotes the **composite self-reported CHD outcome** (coronary heart disease, angina, or myocardial infarction). It must not be interpreted as the single original NHANES item named `MCQ160C`.

## Files required to run

This repository includes **modeling code only**, not participant-level data. Prepare these seven CSV files with the preceding cohort selection, missing-data handling, multiple imputation, and development-only variable selection pipeline:

```text
inputs/
├── Development_MI_1.csv
├── Development_MI_2.csv
├── Development_MI_3.csv
├── Validation_MI_1.csv
├── Validation_MI_2.csv
├── Validation_MI_3.csv
└── Integrated_final_variables.csv
```

Each dataset must include `SEQN`, the study-defined outcome `MCQ160C` (coded `0`/`1`), and the required predictor columns. **Row order and participant identifiers must agree across all three imputations** for each temporal period. The final-variable CSV must list the **58 selected predictors** in its `Variable` column (or first column), consistent with the selected-variable list in the script. Do not include outcome information among the predictors.

The R script **does not rerun** NHANES extraction, cohort inclusion/exclusion, multiple imputation, or stable-LASSO predictor selection; those steps need separate documentation/code for end-to-end reproducibility.

## R packages

The script uses: `caret`, `recipes`, `themis`, `pROC`, `PRROC`, `dplyr`, `tidyr`, `ggplot2`, `nnet`, `LiblineaR`, `ranger`, `catboost`, `h2o`, `dcurves`, and `grid`.

Install compatible package versions before running. Some packages, especially `catboost` and `h2o`, may need platform-specific setup. The `sessionInfo.txt` output records the versions used in a successful run.

## How to run

Set the input and output directories before running the R script. The paths below are examples for a **local** Windows R session:

```r
Sys.setenv(
  NHANES_INPUT_DIR = "F:/revise",
  NHANES_OUTPUT_DIR = "F:/revise/outputs_10fold"
)
source("Integrated_Model_10fold.R")
```

You can alternatively launch `Rscript Integrated_Model_10fold.R` from a directory containing the required CSVs. Default paths are the current directory for input and `outputs_10fold/` for output.

A full run can be computationally expensive: each hyperparameter setting is evaluated on **30 training/held-out pairs** (10 folds × 3 imputations); additional OOF and permutation analyses follow. The script uses a reproducible random seed (`2026`) but numerical results may vary across package versions and compute environments.

## Main output files

Results are written to `outputs_10fold/` by default, including:

- `Integrated_*_CV_results.csv` and `Integrated_*_bestTune.csv`: algorithm-specific tuning results.
- `Integrated_Development_10fold_OOF_probabilities.rds` and `Integrated_Development_10fold_Thresholds.csv`: development OOF predictions and cutoffs.
- `Integrated_AllModels_Performance.csv`: temporal-validation metrics and confidence intervals.
- `Integrated_BestModel_vs_OtherModels_DeLong_Holm.csv`: exploratory paired DeLong comparisons and Holm-adjusted *P* values.
- `Integrated_AllModels_Calibration_Statistics.csv` and the calibration-plot files: assessment of probability calibration.
- `Integrated_AllModels_DCA.tiff`: decision-curve analysis.
- `Integrated_RF_PermutationImportance_*`: RF-specific AUROC-drop variable importance estimates.
- `sessionInfo.txt`: R environment metadata.

The RF permutation-importance step saves incremental results in a checkpoint folder. **Delete that checkpoint folder** if the underlying data, fitted model, or permutation settings change; otherwise its existing values are reused. This step measures RF feature importance **on the temporal-validation cohort** and does not alter model tuning.

## Interpretation notes

1. Ten-fold CV **changes the analysis protocol compared with the earlier two-fold script**. All tuned models, OOF thresholds, plots, and manuscript tables should be recomputed; old CV output files must not be substituted.
2. Feature selection and imputation are **upstream** steps. Because the script uses a fixed development-selected predictor set, tuning-fold AUROC should not be represented as a fully nested feature-selection CV estimate.
3. Platt-calibrated SVM predictions and the other fitted probabilities are evaluated without additional temporal-validation recalibration. Probability calibration may be affected by class balancing.
4. Holm-adjusted best-vs-other DeLong *P* values are **exploratory** because the best model is selected using the same temporal-validation observations.
5. The input CSVs and participant-specific outputs are excluded from version control by `.gitignore`. Review all files for sensitive or unreleasable information before publishing.

## Scope and status

This repository contains the revised modeling code. **No numerical result is asserted here:** the script requires the original input CSVs, and its complete execution has not been verified within this repository package.
