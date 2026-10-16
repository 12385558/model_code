# ============================================================
# Integrated Model: CHD identification using NHANES 2007-2018
# Six algorithms: BP, linear SVM, random forest, CatBoost,
# logistic regression, and deep neural network (DNN).
# ============================================================
# DATA: Three pre-imputed datasets per period (MI 1-3).
# Development: 2007-2014; independent temporal validation: 2015-2018.
# MCQ160C in these datasets is the STUDY-DEFINED composite CHD outcome
# (self-reported CHD, angina, or myocardial infarction); it is not the
# single original NHANES questionnaire item of the same name.
#
# METHODS: Stratified 10-fold cross-validation on the development set,
# with identical patient-level folds across all imputed datasets.
# SMOTENC, normalization, and dummy encoding are trained inside each
# training fold; held-out folds and temporal validation are untouched.
# Hyperparameters are selected by mean fold-wise AUROC across 3 MI sets.
# Final probabilities are arithmetic means across the 3 fitted MI models.
# The temporal validation set is NEVER used for tuning or thresholds.
#
# REQUIREMENTS: Preprocessing, imputation, LASSO, and cohort selection
# were completed upstream; this script does not perform those stages.
# ============================================================

# Configure paths via environment variables (no user-specific paths).
INPUT_DIR  <- Sys.getenv("NHANES_INPUT_DIR", unset = ".")
OUTPUT_DIR <- Sys.getenv("NHANES_OUTPUT_DIR", unset = "outputs_10fold")
SEED <- 2026L
N_MI <- 3L
CV_FOLDS <- 10L

if (!dir.exists(INPUT_DIR)) stop("Input folder not found: ", INPUT_DIR)
dir.create(OUTPUT_DIR, showWarnings = FALSE, recursive = TRUE)
INPUT_DIR <- normalizePath(INPUT_DIR, mustWork = TRUE)
OUTPUT_DIR <- normalizePath(OUTPUT_DIR, mustWork = TRUE)
in_file  <- function(name) file.path(INPUT_DIR, name)
out_file <- function(name) file.path(OUTPUT_DIR, name)

# R packages used by the modeling, evaluation and plotting pipeline.
library(caret)
library(recipes)
library(themis)
library(pROC)
library(PRROC)
library(dplyr)
library(tidyr)
library(ggplot2)
library(nnet)
library(LiblineaR)
library(ranger)
library(catboost)
library(h2o)
library(dcurves)
library(grid)

set.seed(SEED)

# ============================================================
# 1. Read 3 MI datasets
# ============================================================

dev_list <- lapply(seq_len(N_MI),function(i)
  read.csv(in_file(paste0("Development_MI_",i,".csv")),
           check.names=FALSE,stringsAsFactors=FALSE))

val_list <- lapply(seq_len(N_MI),function(i)
  read.csv(in_file(paste0("Validation_MI_",i,".csv")),
           check.names=FALSE,stringsAsFactors=FALSE))


# ============================================================
# 2. Read Integrated stable-LASSO variables
# ============================================================

tmp_vars <- read.csv(
  in_file("Integrated_final_variables.csv"),
  check.names=FALSE,
  stringsAsFactors=FALSE
)

integrated_vars <- if("Variable" %in% names(tmp_vars))
  tmp_vars$Variable else tmp_vars[[1]]

integrated_vars <- unique(trimws(as.character(integrated_vars)))
integrated_vars <- integrated_vars[
  !is.na(integrated_vars) & integrated_vars!=""
]

expected_integrated_vars <- c(
  "RIAGENDR","RIDAGEYR","RIDRETH1","DMDEDUC2","INDFMPIR",
  "BPXPULS","BPXPLS","BPXDI",
  "LBDSBUSI","LBDSCHSI","LBDSIRSI","LBXSLDSI","LBDSTBSI",
  "LBDSUASI","LBXSKSI","LBXWBCSI","LBXLYPCT","LBXMOPCT",
  "LBXEOPCT","LBXHGB","LBXRDW","LBXPLTSI","LBXGH",
  "BPQ020","BPQ080","BPQ070","DIQ010","DIQ170","DIQ050",
  "HIQ011","HSD010","HSQ500","HSQ571","KIQ022","KIQ042",
  "KIQ480","MCQ010","MCQ080","MCQ092","MCQ160G","MCQ160K",
  "MCQ160L","MCQ300A","OCD150","DBQ229","DRQSDIET",
  "DSDANCNT","SMQ020","SLQ050",
  "DR1TPROT","DR1TFIBE","DR1TLZ","DR1TVK","DR1TPHOS",
  "DR1TZINC","DR1TS080","DR1TS120","DR1BWATZ"
)

cat("Integrated variables =",length(integrated_vars),"\n")
print(integrated_vars)

stopifnot(
  length(integrated_vars)==58,
  setequal(integrated_vars,expected_integrated_vars)
)

for(i in seq_len(N_MI)){
  stopifnot(
    all(c("SEQN",integrated_vars,"MCQ160C") %in% names(dev_list[[i]])),
    all(c("SEQN",integrated_vars,"MCQ160C") %in% names(val_list[[i]]))
  )
}


# ============================================================
# 3. Variable types
# ============================================================

# Only final selected predictors are listed here. Preserve the
# study's upstream variable encoding, including categorical age/income.
categorical_all <- c(
  "RIAGENDR", "RIDAGEYR", "RIDRETH1", "DMDEDUC2", "INDFMPIR",
  "BPXPULS", "BPQ020", "BPQ080", "BPQ070", "DIQ010", "DIQ170",
  "DIQ050", "HIQ011", "HSD010", "HSQ500", "HSQ571", "KIQ022",
  "KIQ042", "MCQ010", "MCQ080", "MCQ092", "MCQ160G", "MCQ160K",
  "MCQ160L", "MCQ300A", "OCD150", "DBQ229", "DRQSDIET",
  "SMQ020", "SLQ050"
)

categorical_vars <- intersect(categorical_all,integrated_vars)
numeric_vars <- setdiff(integrated_vars,categorical_vars)

cat("\nCategorical =",length(categorical_vars),"\n")
print(categorical_vars)
cat("\nNumeric =",length(numeric_vars),"\n")

factor_levels <- lapply(categorical_vars,function(v){
  x <- unlist(lapply(dev_list,function(d) as.character(d[[v]])))
  sort(unique(x[!is.na(x)]))
})
names(factor_levels) <- categorical_vars


# ============================================================
# 4. Format datasets
# ============================================================

for(i in seq_len(N_MI)){
  
  dev_list[[i]] <- dev_list[[i]][
    ,c("SEQN",integrated_vars,"MCQ160C"),drop=FALSE]
  
  val_list[[i]] <- val_list[[i]][
    ,c("SEQN",integrated_vars,"MCQ160C"),drop=FALSE]
  
  for(v in categorical_vars){
    dev_list[[i]][[v]] <- factor(
      as.character(dev_list[[i]][[v]]),
      levels=factor_levels[[v]]
    )
    val_list[[i]][[v]] <- factor(
      as.character(val_list[[i]][[v]]),
      levels=factor_levels[[v]]
    )
  }
  
  dev_list[[i]][numeric_vars] <-
    lapply(dev_list[[i]][numeric_vars],as.numeric)
  
  val_list[[i]][numeric_vars] <-
    lapply(val_list[[i]][numeric_vars],as.numeric)
  
  dev_list[[i]]$MCQ160C <- factor(
    dev_list[[i]]$MCQ160C,
    levels=c(0,1),labels=c("No","Yes")
  )
  
  val_list[[i]]$MCQ160C <- factor(
    val_list[[i]]$MCQ160C,
    levels=c(0,1),labels=c("No","Yes")
  )
}

stopifnot(
  identical(dev_list[[1]]$SEQN,dev_list[[2]]$SEQN),
  identical(dev_list[[1]]$SEQN,dev_list[[3]]$SEQN),
  identical(val_list[[1]]$SEQN,val_list[[2]]$SEQN),
  identical(val_list[[1]]$SEQN,val_list[[3]]$SEQN)
)

for(i in seq_len(N_MI)){
  stopifnot(
    !anyNA(dev_list[[i]][,c(integrated_vars,"MCQ160C")]),
    !anyNA(val_list[[i]][,c(integrated_vars,"MCQ160C")])
  )
}

# Integrity checks: a person must not appear twice or cross time periods.
stopifnot(!anyDuplicated(dev_list[[1]]$SEQN))
stopifnot(!anyDuplicated(val_list[[1]]$SEQN))
stopifnot(!any(dev_list[[1]]$SEQN %in% val_list[[1]]$SEQN))
for (i in seq_len(N_MI)) {
  stopifnot(identical(dev_list[[1]]$MCQ160C, dev_list[[i]]$MCQ160C))
  stopifnot(identical(val_list[[1]]$MCQ160C, val_list[[i]]$MCQ160C))
}

cat("\nDevelopment N =",nrow(dev_list[[1]]),"\n")
cat("Temporal Validation N =",nrow(val_list[[1]]),"\n")
print(table(dev_list[[1]]$MCQ160C))
print(table(val_list[[1]]$MCQ160C))


# ============================================================
# 5. Patient-level stratified 10-fold CV (shared across MI datasets)
# ============================================================

# SEQN ordering is checked above; folds can be shared by row index.
set.seed(SEED)
folds <- createFolds(
  dev_list[[1]]$MCQ160C,
  k = CV_FOLDS,
  returnTrain = FALSE
)
stopifnot(length(folds) == CV_FOLDS)
stopifnot(identical(sort(unlist(folds, use.names = FALSE)), seq_len(nrow(dev_list[[1]]))))


# ============================================================
# 6. Fold-safe preprocessing (SMOTENC applied ONLY to training rows)
# ============================================================

prepare_pair <- function(train_dat,valid_dat,vars,
                         normalize=FALSE,dummy=FALSE,
                         do_smotenc=TRUE,seed=2026){
  
  train0 <- train_dat[,c(vars,"MCQ160C"),drop=FALSE]
  valid0 <- valid_dat[,c(vars,"MCQ160C"),drop=FALSE]
  
  rec <- recipe(MCQ160C~.,data=train0)
  
  if(do_smotenc)
    rec <- rec %>%
    step_smotenc(
      MCQ160C,
      over_ratio=1,
      neighbors=5,
      skip=TRUE,
      seed=seed
    )
  
  if(normalize)
    rec <- rec %>%
    step_normalize(all_numeric_predictors())
  
  if(dummy)
    rec <- rec %>%
    step_dummy(all_nominal_predictors(),one_hot=FALSE)
  
  rec_prep <- prep(rec,training=train0,retain=TRUE)
  
  list(
    train=as.data.frame(juice(rec_prep)),
    valid=as.data.frame(bake(rec_prep,new_data=valid0)),
    recipe=rec_prep
  )
}


# ============================================================
# 7. Helpers
# ============================================================

get_auc <- function(y,p){
  if(is.factor(y)) y <- ifelse(y=="Yes",1,0)
  as.numeric(auc(
    roc(y,p,levels=c(0,1),direction="<",quiet=TRUE)
  ))
}

get_decision_score <- function(fit,newx){
  z <- predict(fit,newx,decisionValues=TRUE)$decisionValues
  if(is.matrix(z)||is.data.frame(z)) z <- z[,1]
  as.numeric(z)
}

orient_svm_score <- function(score_train,y_train,score_new){
  if(mean(score_train[y_train==1]) <
     mean(score_train[y_train==0]))
    score_new <- -score_new
  score_new
}

make_roc <- function(actual,prob){
  roc(
    actual,prob,
    levels=c("No","Yes"),
    direction="<",
    quiet=TRUE
  )
}


# ============================================================
# 8. H2O helpers
# ============================================================

h2o.init(nthreads=-1,max_mem_size="8G")
h2o.no_progress()

get_h2o_prob <- function(model,frame){
  p <- as.data.frame(h2o.predict(model,frame))
  if(!"Yes" %in% names(p))
    stop("H2O prediction does not contain Yes probability.")
  p$Yes
}

fit_h2o_dnn <- function(train_df,hidden,activation,seed=2026){
  
  htr <- as.h2o(train_df)
  htr$MCQ160C <- as.factor(htr$MCQ160C)
  
  pars <- list(
    x=setdiff(names(train_df),"MCQ160C"),
    y="MCQ160C",
    training_frame=htr,
    hidden=hidden,
    activation=activation,
    epochs=30,
    l2=1e-4,
    input_dropout_ratio=.1,
    standardize=FALSE,
    balance_classes=FALSE,
    reproducible=TRUE,
    seed=seed
  )
  
  if(activation=="RectifierWithDropout")
    pars$hidden_dropout_ratios <- rep(.1,length(hidden))
  
  list(
    model=do.call(h2o.deeplearning,pars),
    frame=htr
  )
}


# ============================================================
# 9. BP tuning: 10 folds x 3 imputations per hyperparameter
# ============================================================

bp_grid <- expand.grid(
  size=c(5,10,20),
  decay=c(.01,.10)
)

bp_result <- transform(bp_grid,Mean_ROC=NA_real_)

for(g in 1:nrow(bp_grid)){
  
  a <- c()
  cat("\nBP:",bp_grid$size[g],bp_grid$decay[g],"\n")
  
  for(i in seq_len(N_MI)) for(k in seq_along(folds)){
    
    id <- folds[[k]]
    
    pp <- prepare_pair(
      dev_list[[i]][-id,,drop=FALSE],
      dev_list[[i]][id,,drop=FALSE],
      integrated_vars,TRUE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train
    va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    set.seed(2026+k)
    
    fit <- nnet(
      x=as.matrix(tr[,x,drop=FALSE]),
      y=ifelse(tr$MCQ160C=="Yes",1,0),
      size=bp_grid$size[g],
      decay=bp_grid$decay[g],
      maxit=500,
      trace=FALSE,
      MaxNWts=50000
    )
    
    pr <- as.numeric(
      predict(
        fit,
        as.matrix(va[,x,drop=FALSE]),
        type="raw"
      )
    )
    
    a <- c(a,get_auc(va$MCQ160C,pr))
  }
  
  bp_result$Mean_ROC[g] <- mean(a)
}

bp_result <- bp_result[order(-bp_result$Mean_ROC),]
best_size <- bp_result$size[1]
best_decay <- bp_result$decay[1]

print(bp_result)

write.csv(
  bp_result,
  out_file("Integrated_BP_CV_results.csv"),
  row.names=FALSE
)

write.csv(
  bp_result[1,,drop=FALSE],
  out_file("Integrated_BP_bestTune.csv"),
  row.names=FALSE
)


# ============================================================
# 10. SVM tuning: ROC AUC of orientation-corrected decision scores
# ============================================================

svm_grid <- data.frame(C=c(.1,1,10))
svm_result <- transform(svm_grid,Mean_ROC=NA_real_)
svm_type <- 2

for(g in 1:nrow(svm_grid)){
  
  a <- c()
  cat("\nSVM C =",svm_grid$C[g],"\n")
  
  for(i in seq_len(N_MI)) for(k in seq_along(folds)){
    
    id <- folds[[k]]
    
    pp <- prepare_pair(
      dev_list[[i]][-id,,drop=FALSE],
      dev_list[[i]][id,,drop=FALSE],
      integrated_vars,TRUE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train
    va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    xtr <- as.matrix(tr[,x,drop=FALSE])
    xva <- as.matrix(va[,x,drop=FALSE])
    ytr <- ifelse(tr$MCQ160C=="Yes",1,0)
    
    fit <- LiblineaR(
      data=xtr,target=ytr,
      type=svm_type,
      cost=svm_grid$C[g],
      bias=1,
      verbose=FALSE
    )
    
    s_tr <- get_decision_score(fit,xtr)
    
    s_va <- orient_svm_score(
      s_tr,ytr,
      get_decision_score(fit,xva)
    )
    
    a <- c(a,get_auc(va$MCQ160C,s_va))
  }
  
  svm_result$Mean_ROC[g] <- mean(a)
}

svm_result <- svm_result[order(-svm_result$Mean_ROC),]
best_C <- svm_result$C[1]

print(svm_result)

write.csv(
  svm_result,
  out_file("Integrated_SVM_CV_results.csv"),
  row.names=FALSE
)

write.csv(
  data.frame(
    C=best_C,
    Type=svm_type,
    Mean_ROC=svm_result$Mean_ROC[1]
  ),
  out_file("Integrated_SVM_bestTune.csv"),
  row.names=FALSE
)


# ============================================================
# 11. RF tuning
# ============================================================

# Derive the dummy-encoded feature count from development data only.
pp0 <- prepare_pair(
  dev_list[[1]],dev_list[[1]],
  integrated_vars,
  FALSE,TRUE,FALSE
)

p_rf <- ncol(pp0$train)-1

mtry_grid <- sort(unique(
  pmax(
    1,
    pmin(
      p_rf,
      round(c(sqrt(p_rf),p_rf/3,p_rf/2))
    )
  )
))

rf_result <- data.frame(
  mtry=mtry_grid,
  Mean_ROC=NA_real_
)

RF_TREES <- 500
RF_NODE_SIZE <- 5

for(g in 1:nrow(rf_result)){
  
  a <- c()
  cat("\nRF mtry =",rf_result$mtry[g],"\n")
  
  for(i in seq_len(N_MI)) for(k in seq_along(folds)){
    
    id <- folds[[k]]
    
    pp <- prepare_pair(
      dev_list[[i]][-id,,drop=FALSE],
      dev_list[[i]][id,,drop=FALSE],
      integrated_vars,FALSE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train
    va <- pp$valid
    tr$MCQ160C <- factor(
      tr$MCQ160C,levels=c("No","Yes")
    )
    
    fit <- ranger(
      MCQ160C~.,
      data=tr,
      probability=TRUE,
      num.trees=RF_TREES,
      mtry=rf_result$mtry[g],
      min.node.size=RF_NODE_SIZE,
      seed=2026+k
    )
    
    pr <- predict(
      fit,
      data=va[,setdiff(names(va),"MCQ160C"),drop=FALSE]
    )$predictions[,"Yes"]
    
    a <- c(a,get_auc(va$MCQ160C,pr))
  }
  
  rf_result$Mean_ROC[g] <- mean(a)
}

rf_result <- rf_result[order(-rf_result$Mean_ROC),]
best_mtry <- rf_result$mtry[1]

print(rf_result)

write.csv(
  rf_result,
  out_file("Integrated_RF_CV_results.csv"),
  row.names=FALSE
)

write.csv(
  data.frame(
    mtry=best_mtry,
    num.trees=500,
    min.node.size=5,
    Mean_ROC=rf_result$Mean_ROC[1]
  ),
  out_file("Integrated_RF_bestTune.csv"),
  row.names=FALSE
)


# ============================================================
# 12. CatBoost tuning
# ============================================================

cb_grid <- expand.grid(
  depth=c(4,6),
  learning_rate=c(.03,.10),
  l2_leaf_reg=c(3,5)
)

cb_result <- transform(cb_grid,Mean_ROC=NA_real_)
CB_ITER <- 300

for(g in 1:nrow(cb_grid)){
  
  a <- c()
  
  cat(
    "\nCatBoost:",
    g,"/",nrow(cb_grid),
    "depth",cb_grid$depth[g],
    "lr",cb_grid$learning_rate[g],
    "l2",cb_grid$l2_leaf_reg[g],"\n"
  )
  
  for(i in seq_len(N_MI)) for(k in seq_along(folds)){
    
    id <- folds[[k]]
    
    pp <- prepare_pair(
      dev_list[[i]][-id,,drop=FALSE],
      dev_list[[i]][id,,drop=FALSE],
      integrated_vars,FALSE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train
    va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    ptr <- catboost.load_pool(
      as.matrix(tr[,x,drop=FALSE]),
      label=ifelse(tr$MCQ160C=="Yes",1,0)
    )
    
    pva <- catboost.load_pool(
      as.matrix(va[,x,drop=FALSE])
    )
    
    fit <- catboost.train(
      ptr,
      params=list(
        iterations=CB_ITER,
        depth=as.integer(cb_grid$depth[g]),
        learning_rate=cb_grid$learning_rate[g],
        l2_leaf_reg=cb_grid$l2_leaf_reg[g],
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=2026+k,
        allow_writing_files=FALSE,
        logging_level="Silent"
      )
    )
    
    pr <- catboost.predict(
      fit,pva,
      prediction_type="Probability"
    )
    
    a <- c(a,get_auc(va$MCQ160C,pr))
  }
  
  cb_result$Mean_ROC[g] <- mean(a)
}

cb_result <- cb_result[order(-cb_result$Mean_ROC),]

best_depth <- cb_result$depth[1]
best_lr <- cb_result$learning_rate[1]
best_l2 <- cb_result$l2_leaf_reg[1]

print(cb_result)

write.csv(
  cb_result,
  out_file("Integrated_CatBoost_CV_results.csv"),
  row.names=FALSE
)

write.csv(
  data.frame(
    depth=best_depth,
    learning_rate=best_lr,
    l2_leaf_reg=best_l2,
    iterations=300,
    Mean_ROC=cb_result$Mean_ROC[1]
  ),
  out_file("Integrated_CatBoost_bestTune.csv"),
  row.names=FALSE
)


# ============================================================
# 13. DNN tuning
# ============================================================

dnn_grid <- expand.grid(
  architecture=c(
    "30-30-30",
    "64-32-16",
    "64-64-32"
  ),
  activation=c(
    "Rectifier",
    "RectifierWithDropout"
  ),
  stringsAsFactors=FALSE
)

get_hidden <- function(x)
  as.numeric(strsplit(as.character(x),"-")[[1]])

dnn_result <- transform(
  dnn_grid,
  Mean_ROC=NA_real_
)

for(g in 1:nrow(dnn_grid)){
  
  a <- c()
  hidden <- get_hidden(dnn_grid$architecture[g])
  activation <- dnn_grid$activation[g]
  
  cat(
    "\nDNN:",
    dnn_grid$architecture[g],
    activation,"\n"
  )
  
  for(i in seq_len(N_MI)) for(k in seq_along(folds)){
    
    id <- folds[[k]]
    
    pp <- prepare_pair(
      dev_list[[i]][-id,,drop=FALSE],
      dev_list[[i]][id,,drop=FALSE],
      integrated_vars,TRUE,TRUE,TRUE,2026+k
    )
    
    obj <- fit_h2o_dnn(
      pp$train,
      hidden,
      activation,
      2026+k
    )
    
    hva <- as.h2o(pp$valid)
    hva$MCQ160C <- as.factor(hva$MCQ160C)
    
    pr <- get_h2o_prob(obj$model,hva)
    
    a <- c(a,get_auc(pp$valid$MCQ160C,pr))
    
    h2o.rm(obj$model)
    h2o.rm(obj$frame)
    h2o.rm(hva)
  }
  
  dnn_result$Mean_ROC[g] <- mean(a)
}

dnn_result <- dnn_result[order(-dnn_result$Mean_ROC),]

best_hidden <- get_hidden(
  dnn_result$architecture[1]
)

best_activation <- as.character(
  dnn_result$activation[1]
)

print(dnn_result)

write.csv(
  dnn_result,
  out_file("Integrated_DNN_CV_results.csv"),
  row.names=FALSE
)

write.csv(
  data.frame(
    architecture=dnn_result$architecture[1],
    activation=best_activation,
    epochs=30,
    l2=1e-4,
    Mean_ROC=dnn_result$Mean_ROC[1]
  ),
  out_file("Integrated_DNN_bestTune.csv"),
  row.names=FALSE
)


# ============================================================
# 14. Best hyperparameters
# ============================================================

cat("\n===== BEST PARAMETERS =====\n")
cat("BP:",best_size,best_decay,"\n")
cat("SVM C:",best_C,"\n")
cat("RF mtry:",best_mtry,"\n")
cat("CatBoost:",best_depth,best_lr,best_l2,"\n")
cat(
  "DNN:",
  paste(best_hidden,collapse="-"),
  best_activation,"\n"
)


# ============================================================
# 15. Final BP
# ============================================================

bp_final_models <- vector("list", N_MI)
bp_final_recipes <- vector("list", N_MI)
bp_prob_mi <- vector("list", N_MI)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    integrated_vars,TRUE,TRUE,TRUE,2026
  )
  
  tr <- pp$train
  va <- pp$valid
  x <- setdiff(names(tr),"MCQ160C")
  
  set.seed(SEED)
  
  fit <- nnet(
    x=as.matrix(tr[,x,drop=FALSE]),
    y=ifelse(tr$MCQ160C=="Yes",1,0),
    size=best_size,
    decay=best_decay,
    maxit=500,
    trace=FALSE,
    MaxNWts=50000
  )
  
  bp_prob_mi[[i]] <- as.numeric(
    predict(
      fit,
      as.matrix(va[,x,drop=FALSE]),
      type="raw"
    )
  )
  
  bp_final_models[[i]] <- fit
  bp_final_recipes[[i]] <- pp$recipe
}

bp_final_prob <- rowMeans(
  do.call(cbind,bp_prob_mi)
)

bp_actual <- val_list[[1]]$MCQ160C

r_bp <- make_roc(
  bp_actual,bp_final_prob
)

cat(
  "\nBP AUROC =",
  round(as.numeric(auc(r_bp)),4),"\n"
)

save(
  bp_final_models,bp_final_recipes,
  bp_prob_mi,bp_final_prob,
  bp_actual,r_bp,
  file=out_file("Integrated_BP_results.RData")
)

saveRDS(
  r_bp,
  out_file("Integrated_BP_ROC.rds")
)


# ============================================================
# 16. Final SVM + 10-fold OOF-score Platt probability calibration
# Platt regression is trained using development OOF decision scores.
# ============================================================

svm_final_models <- vector("list", N_MI)
svm_final_recipes <- vector("list", N_MI)
svm_platt_models <- vector("list", N_MI)
svm_prob_mi <- vector("list", N_MI)
svm_score_mi <- vector("list", N_MI)
svm_oof_prob_mi <- vector("list", N_MI)

for(i in seq_len(N_MI)){
  
  dat <- dev_list[[i]]
  oof_score <- rep(NA_real_,nrow(dat))
  
  for(k in seq_along(folds)){
    
    id <- folds[[k]]
    
    pp <- prepare_pair(
      dat[-id,,drop=FALSE],
      dat[id,,drop=FALSE],
      integrated_vars,TRUE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train
    va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    xtr <- as.matrix(tr[,x,drop=FALSE])
    xva <- as.matrix(va[,x,drop=FALSE])
    ytr <- ifelse(tr$MCQ160C=="Yes",1,0)
    
    fit <- LiblineaR(
      data=xtr,target=ytr,
      type=2,cost=best_C,
      bias=1,verbose=FALSE
    )
    
    oof_score[id] <- orient_svm_score(
      get_decision_score(fit,xtr),
      ytr,
      get_decision_score(fit,xva)
    )
  }
  
  y_dev_i <- ifelse(dat$MCQ160C=="Yes",1,0)
  
  platt_i <- glm(
    y_dev_i~oof_score,
    family=binomial()
  )
  
  svm_oof_prob_mi[[i]] <- as.numeric(
    predict(
      platt_i,
      newdata=data.frame(oof_score=oof_score),
      type="response"
    )
  )
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    integrated_vars,TRUE,TRUE,TRUE,2026
  )
  
  tr <- pp$train
  va <- pp$valid
  x <- setdiff(names(tr),"MCQ160C")
  
  xtr <- as.matrix(tr[,x,drop=FALSE])
  xva <- as.matrix(va[,x,drop=FALSE])
  ytr <- ifelse(tr$MCQ160C=="Yes",1,0)
  
  fit <- LiblineaR(
    data=xtr,target=ytr,
    type=2,cost=best_C,
    bias=1,verbose=FALSE
  )
  
  s_va <- orient_svm_score(
    get_decision_score(fit,xtr),
    ytr,
    get_decision_score(fit,xva)
  )
  
  pr <- as.numeric(
    predict(
      platt_i,
      newdata=data.frame(oof_score=s_va),
      type="response"
    )
  )
  
  svm_final_models[[i]] <- fit
  svm_final_recipes[[i]] <- pp$recipe
  svm_platt_models[[i]] <- platt_i
  svm_score_mi[[i]] <- s_va
  svm_prob_mi[[i]] <- pr
}

svm_final_prob <- rowMeans(
  do.call(cbind,svm_prob_mi)
)

svm_final_score <- rowMeans(
  do.call(cbind,svm_score_mi)
)

svm_oof_prob <- rowMeans(
  do.call(cbind,svm_oof_prob_mi)
)

svm_actual <- val_list[[1]]$MCQ160C

r_svm <- make_roc(
  svm_actual,svm_final_prob
)

cat(
  "\nSVM AUROC =",
  round(as.numeric(auc(r_svm)),4),"\n"
)

save(
  svm_final_models,svm_final_recipes,
  svm_platt_models,svm_score_mi,
  svm_prob_mi,svm_final_score,
  svm_final_prob,svm_oof_prob,
  svm_actual,r_svm,
  file=out_file("Integrated_SVM_results.RData")
)

saveRDS(
  r_svm,
  out_file("Integrated_SVM_ROC.rds")
)


# ============================================================
# 17. Final RF
# ============================================================

rf_final_models <- vector("list", N_MI)
rf_final_recipes <- vector("list", N_MI)
rf_prob_mi <- vector("list", N_MI)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    integrated_vars,FALSE,TRUE,TRUE,2026
  )
  
  tr <- pp$train
  va <- pp$valid
  
  tr$MCQ160C <- factor(
    tr$MCQ160C,
    levels=c("No","Yes")
  )
  
  fit <- ranger(
    MCQ160C~.,
    data=tr,
    probability=TRUE,
    num.trees=500,
    mtry=best_mtry,
    min.node.size=5,
    importance="impurity",
    seed=2026
  )
  
  rf_prob_mi[[i]] <- predict(
    fit,
    data=va[
      ,setdiff(names(va),"MCQ160C"),
      drop=FALSE
    ]
  )$predictions[,"Yes"]
  
  rf_final_models[[i]] <- fit
  rf_final_recipes[[i]] <- pp$recipe
}

rf_final_prob <- rowMeans(
  do.call(cbind,rf_prob_mi)
)

rf_actual <- val_list[[1]]$MCQ160C

r_rf <- make_roc(
  rf_actual,rf_final_prob
)

cat(
  "\nRF AUROC =",
  round(as.numeric(auc(r_rf)),4),"\n"
)

save(
  rf_final_models,rf_final_recipes,
  rf_prob_mi,rf_final_prob,
  rf_actual,r_rf,
  file=out_file("Integrated_RF_results.RData")
)

saveRDS(
  r_rf,
  out_file("Integrated_RF_ROC.rds")
)


# ============================================================
# 18. Final CatBoost
# ============================================================

cb_final_models <- vector("list", N_MI)
cb_final_recipes <- vector("list", N_MI)
cb_prob_mi <- vector("list", N_MI)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    integrated_vars,FALSE,TRUE,TRUE,2026
  )
  
  tr <- pp$train
  va <- pp$valid
  x <- setdiff(names(tr),"MCQ160C")
  
  ptr <- catboost.load_pool(
    as.matrix(tr[,x,drop=FALSE]),
    label=ifelse(tr$MCQ160C=="Yes",1,0)
  )
  
  pva <- catboost.load_pool(
    as.matrix(va[,x,drop=FALSE])
  )
  
  fit <- catboost.train(
    ptr,
    params=list(
      iterations=300,
      depth=as.integer(best_depth),
      learning_rate=best_lr,
      l2_leaf_reg=best_l2,
      loss_function="Logloss",
      eval_metric="AUC",
      random_seed=2026,
      allow_writing_files=FALSE,
      logging_level="Silent"
    )
  )
  
  cb_prob_mi[[i]] <- as.numeric(
    catboost.predict(
      fit,pva,
      prediction_type="Probability"
    )
  )
  
  cb_final_models[[i]] <- fit
  cb_final_recipes[[i]] <- pp$recipe
  
  catboost.save_model(
    fit,
    out_file(paste0(
      "Integrated_CatBoost_MI", i, ".cbm"
    ))
  )
}

cb_final_prob <- rowMeans(
  do.call(cbind,cb_prob_mi)
)

cb_actual <- val_list[[1]]$MCQ160C

r_cb <- make_roc(
  cb_actual,cb_final_prob
)

cat(
  "\nCatBoost AUROC =",
  round(as.numeric(auc(r_cb)),4),"\n"
)

save(
  cb_final_models,cb_final_recipes,
  cb_prob_mi,cb_final_prob,
  cb_actual,r_cb,
  file=out_file("Integrated_CatBoost_results.RData")
)

saveRDS(
  r_cb,
  out_file("Integrated_CatBoost_ROC.rds")
)


# ============================================================
# 19. Logistic regression baseline + SMOTENC
# No hyperparameter search is carried out for this baseline.
# ============================================================

lr_final_models <- vector("list", N_MI)
lr_final_recipes <- vector("list", N_MI)
lr_prob_mi <- vector("list", N_MI)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    integrated_vars,FALSE,FALSE,TRUE,2026
  )
  
  tr <- pp$train
  va <- pp$valid
  
  tr$Y <- ifelse(
    tr$MCQ160C=="Yes",1,0
  )
  
  fit <- glm(
    Y~.-MCQ160C,
    data=tr,
    family=binomial()
  )
  
  lr_prob_mi[[i]] <- as.numeric(
    predict(
      fit,
      newdata=va,
      type="response"
    )
  )
  
  lr_final_models[[i]] <- fit
  lr_final_recipes[[i]] <- pp$recipe
}

lr_final_prob_raw <- rowMeans(
  do.call(cbind,lr_prob_mi)
)

lr_actual <- val_list[[1]]$MCQ160C

r_lr <- make_roc(
  lr_actual,lr_final_prob_raw
)

cat(
  "\nLogistic AUROC =",
  round(as.numeric(auc(r_lr)),4),"\n"
)

save(
  lr_final_models,lr_final_recipes,
  lr_prob_mi,lr_final_prob_raw,
  lr_actual,r_lr,
  file=out_file("Integrated_Logistic_SMOTENC_results.RData")
)

saveRDS(
  r_lr,
  out_file("Integrated_Logistic_SMOTENC_ROC.rds")
)


# ============================================================
# 20. Final DNN
# ============================================================

dnn_prob_mi <- vector("list", N_MI)
dnn_final_recipes <- vector("list", N_MI)
dnn_model_paths <- character(N_MI)

dir.create(
  out_file("Integrated_DNN_models"),
  showWarnings=FALSE,
  recursive=TRUE
)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    integrated_vars,TRUE,TRUE,TRUE,2026
  )
  
  obj <- fit_h2o_dnn(
    pp$train,
    best_hidden,
    best_activation,
    2026
  )
  
  hva <- as.h2o(pp$valid)
  hva$MCQ160C <- as.factor(hva$MCQ160C)
  
  dnn_prob_mi[[i]] <- as.numeric(
    get_h2o_prob(obj$model,hva)
  )
  
  dnn_model_paths[i] <- h2o.saveModel(
    obj$model,
    path=file.path(
      OUTPUT_DIR,
      "Integrated_DNN_models"
    ),
    force=TRUE
  )
  
  dnn_final_recipes[[i]] <- pp$recipe
  
  h2o.rm(obj$model)
  h2o.rm(obj$frame)
  h2o.rm(hva)
}

dnn_final_prob <- rowMeans(
  do.call(cbind,dnn_prob_mi)
)

dnn_actual <- val_list[[1]]$MCQ160C

r_dnn <- make_roc(
  dnn_actual,dnn_final_prob
)

cat(
  "\nDNN AUROC =",
  round(as.numeric(auc(r_dnn)),4),"\n"
)

save(
  dnn_prob_mi,dnn_final_prob,
  dnn_final_recipes,dnn_model_paths,
  dnn_actual,r_dnn,
  file=out_file("Integrated_DNN_results.RData")
)

saveRDS(
  r_dnn,
  out_file("Integrated_DNN_ROC.rds")
)


# ============================================================
# 21. Independent temporal validation: AUROC and DeLong CI
# ============================================================

roc_list <- list(
  BP=r_bp,
  SVM=r_svm,
  RF=r_rf,
  CatBoost=r_cb,
  Logistic=r_lr,
  DNN=r_dnn
)

auc_table <- bind_rows(
  lapply(names(roc_list),function(m){
    
    r <- roc_list[[m]]
    ci <- ci.auc(r,method="delong")
    
    data.frame(
      Model=m,
      AUROC=as.numeric(auc(r)),
      Lower95CI=as.numeric(ci[1]),
      Upper95CI=as.numeric(ci[3])
    )
  })
) %>%
  arrange(desc(AUROC))

print(auc_table)

write.csv(
  auc_table,
  out_file("Integrated_TemporalValidation_AUROC.csv"),
  row.names=FALSE
)


# ============================================================
# 22. Combined ROC plot
# Preserved manuscript figure style
# ============================================================

roc_plot_list <- roc_list[
  auc_table$Model
]

roc_auc <- sapply(
  roc_plot_list,
  function(x) as.numeric(auc(x))
)

cols <- c(
  "#D55E00",
  "#0072B2",
  "#009E73",
  "#CC79A7",
  "#E69F00",
  "#56B4E9"
)

legend_text <- paste0(
  names(roc_plot_list),
  ", AUC = ",
  sprintf("%.3f",roc_auc)
)

tiff(
  out_file("Integrated_Model_ROC.tiff"),
  width=6.5,
  height=6.5,
  units="in",
  res=600,
  compression="lzw"
)

plot(
  roc_plot_list[[1]],
  col=cols[1],
  lwd=2,
  legacy.axes=TRUE,
  xlim=c(1,0),
  ylim=c(0,1),
  xaxp=c(1,0,5),
  yaxp=c(0,1,5),
  xaxs="i",
  yaxs="i",
  xlab="1 - Specificity",
  ylab="Sensitivity"
)

for(i in 2:length(roc_plot_list)){
  plot(
    roc_plot_list[[i]],
    col=cols[i],
    lwd=2,
    add=TRUE,
    legacy.axes=TRUE
  )
}

legend(
  "bottomright",
  legend=legend_text,
  col=cols,
  lwd=2,
  bty="n",
  cex=.80
)

dev.off()


# ============================================================
# 23. Development OOF helper for cutoff estimation
# Uses training-fold-only preprocessing and selected hyperparameters.
# ============================================================

fit_oof_model <- function(model,tr_raw,va_raw,seed){
  
  if(model=="BP"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,integrated_vars,
      TRUE,TRUE,TRUE,seed
    )
    
    tr <- pp$train
    va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    set.seed(seed)
    
    fit <- nnet(
      x=as.matrix(tr[,x,drop=FALSE]),
      y=ifelse(tr$MCQ160C=="Yes",1,0),
      size=best_size,
      decay=best_decay,
      maxit=500,
      trace=FALSE,
      MaxNWts=50000
    )
    
    return(
      as.numeric(
        predict(
          fit,
          as.matrix(va[,x,drop=FALSE]),
          type="raw"
        )
      )
    )
  }
  
  
  if(model=="RF"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,integrated_vars,
      FALSE,TRUE,TRUE,seed
    )
    
    tr <- pp$train
    va <- pp$valid
    
    tr$MCQ160C <- factor(
      tr$MCQ160C,
      levels=c("No","Yes")
    )
    
    fit <- ranger(
      MCQ160C~.,
      data=tr,
      probability=TRUE,
      num.trees=500,
      mtry=best_mtry,
      min.node.size=5,
      seed=seed
    )
    
    return(
      predict(
        fit,
        data=va[
          ,setdiff(names(va),"MCQ160C"),
          drop=FALSE
        ]
      )$predictions[,"Yes"]
    )
  }
  
  
  if(model=="CB"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,integrated_vars,
      FALSE,TRUE,TRUE,seed
    )
    
    tr <- pp$train
    va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    ptr <- catboost.load_pool(
      as.matrix(tr[,x,drop=FALSE]),
      label=ifelse(tr$MCQ160C=="Yes",1,0)
    )
    
    pva <- catboost.load_pool(
      as.matrix(va[,x,drop=FALSE])
    )
    
    fit <- catboost.train(
      ptr,
      params=list(
        iterations=300,
        depth=as.integer(best_depth),
        learning_rate=best_lr,
        l2_leaf_reg=best_l2,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=seed,
        allow_writing_files=FALSE,
        logging_level="Silent"
      )
    )
    
    return(
      as.numeric(
        catboost.predict(
          fit,pva,
          prediction_type="Probability"
        )
      )
    )
  }
  
  
  if(model=="LR"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,integrated_vars,
      FALSE,FALSE,TRUE,seed
    )
    
    tr <- pp$train
    va <- pp$valid
    
    tr$Y <- ifelse(
      tr$MCQ160C=="Yes",1,0
    )
    
    fit <- glm(
      Y~.-MCQ160C,
      data=tr,
      family=binomial()
    )
    
    return(
      as.numeric(
        predict(
          fit,
          newdata=va,
          type="response"
        )
      )
    )
  }
  
  
  if(model=="DNN"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,integrated_vars,
      TRUE,TRUE,TRUE,seed
    )
    
    obj <- fit_h2o_dnn(
      pp$train,
      best_hidden,
      best_activation,
      seed
    )
    
    hva <- as.h2o(pp$valid)
    hva$MCQ160C <- as.factor(hva$MCQ160C)
    
    pr <- get_h2o_prob(
      obj$model,hva
    )
    
    h2o.rm(obj$model)
    h2o.rm(obj$frame)
    h2o.rm(hva)
    
    return(as.numeric(pr))
  }
  
  stop("Unknown model.")
}


# ============================================================
# 24. Development OOF predictions (stratified 10-fold CV)
# ============================================================
# Refit the chosen hyperparameters on each training fold, then predict
# the held-out fold. ALL samples receive exactly one OOF prediction per MI.
# Do not reuse previous cross-validation caches or outdated model fits.
models <- c("RF", "SVM", "LR", "CB", "DNN", "BP")
oof <- setNames(vector("list", length(models)), models)

for (m in setdiff(models, "SVM")) {
  cat("\nComputing development OOF:", m, "\n")
  mi_pred <- vector("list", N_MI)

  for (i in seq_len(N_MI)) {
    dat <- dev_list[[i]]
    p <- rep(NA_real_, nrow(dat))

    for (k in seq_along(folds)) {
      id <- folds[[k]]
      cat("  Model:", m, "| MI:", i, "| fold:", k,
          "/", CV_FOLDS, "\n")
      p[id] <- fit_oof_model(
        m, dat[-id, , drop = FALSE], dat[id, , drop = FALSE],
        seed = SEED + k
      )
    }
    stopifnot(!anyNA(p))
    mi_pred[[i]] <- p
  }
  oof[[m]] <- rowMeans(do.call(cbind, mi_pred))
}

# Reuse the SVM OOF decision scores calibrated in section 16.
# Important: Platt's calibrator was fitted on these development OOF scores;
# the resulting development probabilities are not cross-fitted for Platt.
oof$SVM <- svm_oof_prob
saveRDS(oof, out_file("Integrated_Development_10fold_OOF_probabilities.rds"))

# ============================================================
# 25. Development OOF prediction checks
# ============================================================

oof_check <- sapply(
  oof,
  function(x)
    c(
      N=length(x),
      Missing=sum(is.na(x)),
      Min=min(x,na.rm=TRUE),
      Max=max(x,na.rm=TRUE)
    )
)

print(oof_check)

stopifnot(
  all(oof_check["Missing",]==0)
)


# ============================================================
# 26. Select Youden thresholds using DEVELOPMENT OOF predictions only
# ============================================================

y_dev <- ifelse(
  dev_list[[1]]$MCQ160C=="Yes",
  1,0
)

threshold_info <- bind_rows(
  lapply(models,function(m){
    
    r <- roc(
      y_dev,oof[[m]],
      levels=c(0,1),
      direction="<",
      quiet=TRUE
    )
    
    th <- coords(
      r,
      x="best",
      best.method="youden",
      ret=c(
        "threshold",
        "sensitivity",
        "specificity"
      ),
      transpose=FALSE
    )
    
    data.frame(
      Model=m,
      Threshold=as.numeric(th[["threshold"]]),
      Development_Sensitivity=
        as.numeric(th[["sensitivity"]]),
      Development_Specificity=
        as.numeric(th[["specificity"]])
    )
  })
)

print(threshold_info)

write.csv(
  threshold_info,
  out_file("Integrated_Development_10fold_Thresholds.csv"),
  row.names=FALSE
)


# ============================================================
# 27. Temporal Validation probabilities
# ============================================================

y <- ifelse(
  val_list[[1]]$MCQ160C=="Yes",
  1,0
)

val_pred <- list(
  RF=rf_final_prob,
  SVM=svm_final_prob,
  LR=lr_final_prob_raw,
  CB=cb_final_prob,
  DNN=dnn_final_prob,
  BP=bp_final_prob
)

stopifnot(
  all(
    sapply(val_pred,length)==length(y)
  )
)


# ============================================================
# 28. Performance functions
# ============================================================

safe_div <- function(a,b)
  if(b==0) NA_real_ else a/b

calc_metric <- function(y,prob,thr){
  
  pred <- ifelse(prob>=thr,1,0)
  
  TP <- sum(pred==1 & y==1)
  TN <- sum(pred==0 & y==0)
  FP <- sum(pred==1 & y==0)
  FN <- sum(pred==0 & y==1)
  
  Se <- safe_div(TP,TP+FN)
  Sp <- safe_div(TN,TN+FP)
  PPV <- safe_div(TP,TP+FP)
  NPV <- safe_div(TN,TN+FN)
  Acc <- safe_div(TP+TN,length(y))
  F1 <- safe_div(2*TP,2*TP+FP+FN)
  BA <- (Se+Sp)/2
  
  r <- roc(
    y,prob,
    levels=c(0,1),
    direction="<",
    quiet=TRUE
  )
  
  PRAUC <- PRROC::pr.curve(
    scores.class0=prob[y==1],
    scores.class1=prob[y==0],
    curve=FALSE
  )$auc.integral
  
  c(
    AUROC=as.numeric(auc(r)),
    PRAUC=PRAUC,
    Sensitivity=Se,
    Specificity=Sp,
    PPV=PPV,
    NPV=NPV,
    Accuracy=Acc,
    F1=F1,
    BalancedAccuracy=BA,
    TN=TN,FP=FP,FN=FN,TP=TP
  )
}

boot_ci <- function(y,prob,thr,B=2000){
  
  set.seed(2026)
  
  nm <- c(
    "PRAUC","Sensitivity","Specificity",
    "PPV","NPV","Accuracy","F1",
    "BalancedAccuracy"
  )
  
  z <- matrix(
    NA_real_,
    B,length(nm),
    dimnames=list(NULL,nm)
  )
  
  for(b in 1:B){
    
    id <- sample(
      seq_along(y),
      replace=TRUE
    )
    
    yy <- y[id]
    pp <- prob[id]
    
    if(length(unique(yy))<2)
      next
    
    z[b,] <- calc_metric(
      yy,pp,thr
    )[nm]
  }
  
  apply(
    z,2,quantile,
    c(.025,.975),
    na.rm=TRUE
  )
}


# ============================================================
# 29. Temporal validation performance and bootstrap confidence intervals
# ============================================================

results <- list()
confusion <- list()

for(m in models){
  
  cat("Processing",m,"\n")
  
  thr <- threshold_info$Threshold[
    threshold_info$Model==m
  ]
  
  prob <- val_pred[[m]]
  
  est <- calc_metric(
    y,prob,thr
  )
  
  ci <- boot_ci(
    y,prob,thr,B=2000
  )
  
  r <- roc(
    y,prob,
    levels=c(0,1),
    direction="<",
    quiet=TRUE
  )
  
  aci <- ci.auc(
    r,
    method="delong"
  )
  
  results[[m]] <- data.frame(
    Model=m,
    Threshold=thr,
    
    AUROC=est["AUROC"],
    AUROC_L=as.numeric(aci[1]),
    AUROC_U=as.numeric(aci[3]),
    
    PRAUC=est["PRAUC"],
    PRAUC_L=ci[1,"PRAUC"],
    PRAUC_U=ci[2,"PRAUC"],
    
    Sensitivity=est["Sensitivity"],
    Sens_L=ci[1,"Sensitivity"],
    Sens_U=ci[2,"Sensitivity"],
    
    Specificity=est["Specificity"],
    Spec_L=ci[1,"Specificity"],
    Spec_U=ci[2,"Specificity"],
    
    PPV=est["PPV"],
    PPV_L=ci[1,"PPV"],
    PPV_U=ci[2,"PPV"],
    
    NPV=est["NPV"],
    NPV_L=ci[1,"NPV"],
    NPV_U=ci[2,"NPV"],
    
    Accuracy=est["Accuracy"],
    Accuracy_L=ci[1,"Accuracy"],
    Accuracy_U=ci[2,"Accuracy"],
    
    F1=est["F1"],
    F1_L=ci[1,"F1"],
    F1_U=ci[2,"F1"],
    
    Balanced_Accuracy=
      est["BalancedAccuracy"],
    BA_L=ci[1,"BalancedAccuracy"],
    BA_U=ci[2,"BalancedAccuracy"],
    
    TN=est["TN"],
    FP=est["FP"],
    FN=est["FN"],
    TP=est["TP"]
  )
  
  confusion[[m]] <- data.frame(
    Model=m,
    Actual=c(
      "Non-CHD","Non-CHD",
      "CHD","CHD"
    ),
    Predicted=c(
      "Non-CHD","CHD",
      "Non-CHD","CHD"
    ),
    Count=c(
      est["TN"],est["FP"],
      est["FN"],est["TP"]
    )
  )
}

performance_table <- bind_rows(
  results
) %>%
  mutate(
    across(-Model,as.numeric),
    across(
      where(is.numeric),
      ~round(.x,4)
    )
  ) %>%
  arrange(desc(AUROC))

confusion_table <- bind_rows(
  confusion
) %>%
  mutate(
    Count=as.numeric(Count)
  )

print(performance_table)

write.csv(
  performance_table,
  out_file("Integrated_AllModels_Performance.csv"),
  row.names=FALSE
)

write.csv(
  confusion_table,
  out_file("Integrated_AllModels_Confusion_Matrix.csv"),
  row.names=FALSE
)


# ============================================================
# 30. Identify numerically best Integrated algorithm
# ============================================================

best_model <- performance_table$Model[1]

cat(
  "\nNumerically best Integrated model =",
  best_model,"\n"
)


# ============================================================
# 31. Paired DeLong test on the SAME temporal-validation subjects:
# best algorithm vs all remaining algorithms
# ============================================================

roc_list2 <- list(
  RF=r_rf,
  SVM=r_svm,
  LR=r_lr,
  CB=r_cb,
  DNN=r_dnn,
  BP=r_bp
)

full_name <- c(
  RF="Random Forest",
  SVM="SVM",
  LR="Logistic Regression",
  CB="CatBoost",
  DNN="DNN",
  BP="BP"
)

other_models <- setdiff(
  models,
  best_model
)

delong_best <- bind_rows(
  lapply(other_models,function(m){
    
    r1 <- roc_list2[[best_model]]
    r2 <- roc_list2[[m]]
    
    tt <- roc.test(
      r1,r2,
      paired=TRUE,
      method="delong",
      conf.int=TRUE
    )
    
    ci1 <- ci.auc(
      r1,
      method="delong"
    )
    
    ci2 <- ci.auc(
      r2,
      method="delong"
    )
    
    diff_ci <- if(!is.null(tt$conf.int))
      as.numeric(tt$conf.int)
    else
      c(NA_real_,NA_real_)
    
    data.frame(
      Comparison=paste(
        unname(full_name[best_model]),
        "vs",
        unname(full_name[m])
      ),
      
      Best_AUROC=
        as.numeric(auc(r1)),
      
      Best_Lower95CI=
        as.numeric(ci1[1]),
      
      Best_Upper95CI=
        as.numeric(ci1[3]),
      
      Comparator_AUROC=
        as.numeric(auc(r2)),
      
      Comparator_Lower95CI=
        as.numeric(ci2[1]),
      
      Comparator_Upper95CI=
        as.numeric(ci2[3]),
      
      Delta_AUROC=
        as.numeric(
          auc(r1)-auc(r2)
        ),
      
      Delta_Lower95CI=
        diff_ci[1],
      
      Delta_Upper95CI=
        diff_ci[2],
      
      P_value=
        tt$p.value
    )
  })
)

# Exploratory multiplicity adjustment across best-vs-others tests.
# Because "best" was selected from the same validation set, adjusted
# p-values should NOT be interpreted as confirmatory post-selection tests.
delong_best$P_Holm <- p.adjust(delong_best$P_value, method = "holm")
delong_best$Significant_Holm <- ifelse(
  delong_best$P_Holm < 0.05, "Yes", "No"
)
print(delong_best)

write.csv(
  delong_best,
  out_file("Integrated_BestModel_vs_OtherModels_DeLong_Holm.csv"),
  row.names=FALSE
)


# ============================================================
# 32. Six-model confusion matrix
# Preserved manuscript figure style
# ============================================================

cm_dat <- confusion_table

cm_dat$Model <- factor(
  cm_dat$Model,
  levels=c(
    "RF","SVM","LR",
    "CB","DNN","BP"
  ),
  labels=c(
    "Random Forest",
    "SVM",
    "Logistic Regression",
    "CatBoost",
    "DNN",
    "BP"
  )
)

cm_dat$Actual <- factor(
  cm_dat$Actual,
  levels=c(
    "CHD",
    "Non-CHD"
  )
)

cm_dat$Predicted <- factor(
  cm_dat$Predicted,
  levels=c(
    "Non-CHD",
    "CHD"
  )
)

p_cm <- ggplot(
  cm_dat,
  aes(
    x=Predicted,
    y=Actual,
    fill=Count
  )
) +
  geom_tile(
    colour="white",
    linewidth=1
  ) +
  geom_text(
    aes(label=Count),
    size=4.5
  ) +
  facet_wrap(
    ~Model,
    ncol=3
  ) +
  scale_fill_gradient(
    low="white",
    high="grey60",
    name="Count"
  ) +
  labs(
    x="Predicted class",
    y="Actual class"
  ) +
  theme_classic(
    base_size=12
  ) +
  theme(
    axis.line=element_blank(),
    panel.border=element_rect(
      colour="black",
      fill=NA,
      linewidth=.7
    ),
    strip.background=element_blank(),
    strip.text=element_text(size=11),
    legend.position="right"
  )

ggsave(
  out_file("Integrated_AllModels_Confusion_Matrix.tiff"),
  p_cm,
  width=10,
  height=7,
  units="in",
  dpi=600,
  compression="lzw"
)


# ============================================================
# 33. Save first-stage Integrated results
# ============================================================

save(
  integrated_vars,
  
  bp_result,
  svm_result,
  rf_result,
  cb_result,
  dnn_result,
  
  best_size,
  best_decay,
  best_C,
  best_mtry,
  best_depth,
  best_lr,
  best_l2,
  best_hidden,
  best_activation,
  
  bp_final_models,
  bp_final_recipes,
  bp_final_prob,
  
  svm_final_models,
  svm_final_recipes,
  svm_platt_models,
  svm_final_prob,
  
  rf_final_models,
  rf_final_recipes,
  rf_final_prob,
  
  cb_final_models,
  cb_final_recipes,
  cb_final_prob,
  
  lr_final_models,
  lr_final_recipes,
  lr_final_prob_raw,
  
  dnn_final_recipes,
  dnn_model_paths,
  dnn_final_prob,
  
  auc_table,
  oof,
  oof_check,
  threshold_info,
  performance_table,
  confusion_table,
  best_model,
  delong_best,
  
  file=
    out_file("Integrated_MODEL_SELECTION_RESULTS.RData")
)


cat(
  "\n============================================\n",
  "Integrated Model completed.\n",
  "Best algorithm =",best_model,"\n",
  "============================================\n"
)


# ============================================================
# RF-specific follow-up: the feature-importance analysis below
# is always about RF, regardless of which model ranks first on AUROC.
# ============================================================

# ============================================================
# 34. Decision curve analysis: all six algorithms
# Preserved manuscript figure style
# ============================================================

dca_dat <- data.frame(
  CHD=y,
  RandomForest=rf_final_prob,
  SVM=svm_final_prob,
  Logistic=lr_final_prob_raw,
  CatBoost=cb_final_prob,
  DNN=dnn_final_prob,
  BP=bp_final_prob
)

dca_all <- dca(
  CHD~RandomForest+SVM+Logistic+CatBoost+DNN+BP,
  data=dca_dat,
  thresholds=seq(0.01,0.50,0.01)
)

p_dca <- plot(
  dca_all,
  smooth=TRUE
) +
  scale_x_continuous(
    limits=c(0,0.50),
    breaks=seq(0,0.50,0.10),
    expand=c(0,0)
  ) +
  labs(
    x="Threshold probability",
    y="Net benefit",
    color=NULL
  ) +
  theme_classic(base_size=12) +
  theme(
    panel.border=element_rect(
      colour="black",
      fill=NA,
      linewidth=0.7
    ),
    axis.ticks=element_line(
      linewidth=0.7
    ),
    legend.position=c(0.82,0.78),
    legend.background=element_blank(),
    legend.text=element_text(size=10),
    legend.key.width=unit(1.1,"cm")
  )

ggsave(
  out_file("Integrated_AllModels_DCA.tiff"),
  plot=p_dca,
  width=7,
  height=6,
  units="in",
  dpi=600,
  compression="lzw"
)


# ============================================================
# 35. Reconstruct the RF predictions across three imputations
# ============================================================

predict_rf_3mi <- function(new_val_list){
  
  prob_mi <- vector("list", N_MI)
  
  for(i in seq_len(N_MI)){
    
    new_dat <- new_val_list[[i]][
      ,c(integrated_vars,"MCQ160C"),
      drop=FALSE
    ]
    
    baked <- as.data.frame(
      bake(
        rf_final_recipes[[i]],
        new_data=new_dat
      )
    )
    
    prob_mi[[i]] <- predict(
      rf_final_models[[i]],
      data=baked[
        ,setdiff(names(baked),"MCQ160C"),
        drop=FALSE
      ]
    )$predictions[,"Yes"]
  }
  
  rowMeans(do.call(cbind,prob_mi))
}


# ============================================================
# 36. Validate reconstruction of RF temporal predictions
# ============================================================

rf_prob_check <- predict_rf_3mi(val_list)

max_difference <- max(
  abs(rf_prob_check-rf_final_prob)
)

cat(
  "\nMaximum prediction difference =",
  max_difference,"\n"
)

stopifnot(
  max_difference < 1e-6
)


# ============================================================
# 37. Baseline RF AUROC for permutation importance
# ============================================================

y_val <- ifelse(
  val_list[[1]]$MCQ160C=="Yes",
  1,0
)

baseline_roc <- roc(
  y_val,
  rf_final_prob,
  levels=c(0,1),
  direction="<",
  quiet=TRUE
)

baseline_auc <- as.numeric(
  auc(baseline_roc)
)

cat(
  "\nBaseline RF AUROC =",
  baseline_auc,"\n"
)


# ============================================================
# 38. RF permutation importance measured on temporal validation
# 58 variables × 50 repetitions
# Checkpoint every 5 repetitions; use the same row permutation in each MI
# ============================================================

B_perm <- 50

# Checkpoints allow resuming long permutation analyses; delete this
# directory if the input data, trained RF, or number of repetitions changes.
checkpoint_dir <- out_file("Integrated_RF_LASSO58_10fold_perm50")

dir.create(
  checkpoint_dir,
  showWarnings=FALSE,
  recursive=TRUE
)

for(j in seq_along(integrated_vars)){
  
  v <- integrated_vars[j]
  
  rep_file <- file.path(
    checkpoint_dir,
    paste0(v,"_drops.rds")
  )
  
  drops <- if(file.exists(rep_file))
    readRDS(rep_file)
  else
    numeric(0)
  
  if(length(drops)>=B_perm){
    cat(
      "Already completed:",
      j,"/",length(integrated_vars),
      v,"\n"
    )
    next
  }
  
  cat(
    "\nPermutation importance:",
    j,"/",length(integrated_vars),
    v,
    "| completed",length(drops),"/",B_perm,"\n"
  )
  
  for(b in seq.int(length(drops)+1L,B_perm)){
    
    set.seed(2026+j*1000+b)
    
    idx <- sample(
      seq_len(nrow(val_list[[1]]))
    )
    
    val_perm <- lapply(
      val_list,
      function(dat){
        tmp <- dat
        tmp[[v]] <- dat[[v]][idx]
        tmp
      }
    )
    
    perm_prob <- predict_rf_3mi(
      val_perm
    )
    
    perm_auc <- as.numeric(
      auc(
        roc(
          y_val,
          perm_prob,
          levels=c(0,1),
          direction="<",
          quiet=TRUE
        )
      )
    )
    
    drops <- c(
      drops,
      baseline_auc-perm_auc
    )
    
    if(b%%5==0 || b==B_perm)
      saveRDS(
        drops,
        rep_file
      )
    
    if(b%%10==0)
      cat(
        " repetition",
        b,"/",B_perm,"\n"
      )
  }
  
  gc(verbose=FALSE)
}


# ============================================================
# 39. Summarize permutation AUROC decreases
# ============================================================

perm_files <- file.path(
  checkpoint_dir,
  paste0(integrated_vars,"_drops.rds")
)

complete_check <- sapply(
  perm_files,
  function(f)
    file.exists(f) &&
    length(readRDS(f))>=B_perm
)

cat(
  "\nCompleted variables =",
  sum(complete_check),
  "/",
  length(integrated_vars),
  "\n"
)

if(!all(complete_check)){
  print(
    integrated_vars[
      !complete_check
    ]
  )
  stop("Permutation importance has not finished.")
}

rf_perm_importance <- bind_rows(
  lapply(
    seq_along(integrated_vars),
    function(j){
      
      d <- readRDS(
        perm_files[j]
      )[1:B_perm]
      
      data.frame(
        Variable=integrated_vars[j],
        MeanDecreaseAUROC=mean(d),
        SD=sd(d),
        Lower95=as.numeric(
          quantile(d,0.025)
        ),
        Upper95=as.numeric(
          quantile(d,0.975)
        )
      )
    }
  )
) %>%
  arrange(desc(MeanDecreaseAUROC)) %>%
  mutate(Rank=row_number()) %>%
  select(Rank,everything())

print(rf_perm_importance)

write.csv(
  rf_perm_importance,
  out_file("Integrated_RF_PermutationImportance_AllVariables_50.csv"),
  row.names=FALSE
)


# ============================================================
# 40. Select 15 variables with the largest mean AUROC decrease
# ============================================================

rf_top15 <- rf_perm_importance %>%
  slice_head(n=15)


# ============================================================
# 41. Human-readable labels for the selected variables
# ============================================================

var_labels <- c(
  
  "RIAGENDR"="Sex",
  "RIDAGEYR"="Age group",
  "RIDRETH1"="Race",
  "DMDEDUC2"="Education level",
  "INDFMPIR"="Family income-to-poverty ratio category",
  
  "BPXPULS"="Pulse regularity",
  "BPXPLS"="Pulse rate",
  "BPXDI"="Diastolic blood pressure",
  
  "LBDSBUSI"="Blood urea nitrogen",
  "LBDSCHSI"="Total cholesterol",
  "LBDSIRSI"="Serum iron",
  "LBXSLDSI"="Lactate dehydrogenase",
  "LBDSTBSI"="Total bilirubin",
  "LBDSUASI"="Uric acid",
  "LBXSKSI"="Serum potassium",
  "LBXWBCSI"="White blood cell count",
  "LBXLYPCT"="Lymphocyte percentage",
  "LBXMOPCT"="Monocyte percentage",
  "LBXEOPCT"="Eosinophil percentage",
  "LBXHGB"="Hemoglobin",
  "LBXRDW"="Red cell distribution width",
  "LBXPLTSI"="Platelet count",
  "LBXGH"="Glycated hemoglobin",
  
  "BPQ020"="History of hypertension",
  "BPQ080"="History of high cholesterol",
  "BPQ070"="Time since last cholesterol check",
  
  "DIQ010"="History of diabetes",
  "DIQ170"="Diabetes risk",
  "DIQ050"="Current insulin use",
  
  "HIQ011"="Health insurance coverage",
  "HSD010"="Self-rated general health",
  "HSQ500"="History of hepatitis B",
  "HSQ571"="History of hepatitis C",
  
  "KIQ022"="History of kidney disease",
  "KIQ042"="Urinary leakage during physical activity",
  "KIQ480"="Nocturia frequency",
  
  "MCQ010"="History of asthma",
  "MCQ080"="History of overweight",
  "MCQ092"="History of blood transfusion",
  "MCQ160G"="History of emphysema",
  "MCQ160K"="History of chronic bronchitis",
  "MCQ160L"="History of liver disease",
  "MCQ300A"="Family history of heart attack",
  
  "OCD150"="Employment status",
  "DBQ229"="Regular milk consumption pattern",
  "DRQSDIET"="Special diet",
  "DSDANCNT"="Number of antacid products taken",
  "SMQ020"="Smoked at least 100 cigarettes in life",
  "SLQ050"="Trouble sleeping",
  
  "DR1TPROT"="Protein intake",
  "DR1TFIBE"="Dietary fiber intake",
  "DR1TLZ"="Lutein and zeaxanthin intake",
  "DR1TVK"="Vitamin K intake",
  "DR1TPHOS"="Phosphorus intake",
  "DR1TZINC"="Zinc intake",
  "DR1TS080"="Octanoic acid intake",
  "DR1TS120"="Dodecanoic acid intake",
  "DR1BWATZ"="Bottled water intake"
)

rf_top15$Variable_name <- unname(
  var_labels[
    rf_top15$Variable
  ]
)

rf_top15$Variable_name[
  is.na(rf_top15$Variable_name)
] <- rf_top15$Variable[
  is.na(rf_top15$Variable_name)
]


# ============================================================
# 42. Relative importance (mean AUROC decrease / maximum x 100)
# Same definition / figure style as before
# ============================================================

rf_top15 <- rf_top15 %>%
  mutate(
    RelativeImportance=
      MeanDecreaseAUROC /
      max(MeanDecreaseAUROC) *
      100
  )

print(
  rf_top15[
    ,c(
      "Rank",
      "Variable",
      "Variable_name",
      "MeanDecreaseAUROC",
      "RelativeImportance",
      "Lower95",
      "Upper95"
    )
  ]
)

write.csv(
  rf_top15,
  out_file("Integrated_RF_PermutationImportance_Top15_50.csv"),
  row.names=FALSE
)


# ============================================================
# 43. Plot 15 highest-ranking RF predictors
# ORIGINAL Integrated style
# ============================================================

rf_top15$Variable_name <- factor(
  rf_top15$Variable_name,
  levels=rev(
    rf_top15$Variable_name
  )
)

p_rf_imp <- ggplot(
  rf_top15,
  aes(
    x=RelativeImportance,
    y=Variable_name
  )
) +
  geom_col(
    width=0.70,
    fill="#4E79A7"
  ) +
  labs(
    x="Relative importance",
    y=NULL
  ) +
  scale_x_continuous(
    expand=expansion(
      mult=c(0,0.05)
    )
  ) +
  theme_classic(
    base_size=12
  ) +
  theme(
    axis.line=element_blank(),
    panel.border=element_rect(
      colour="black",
      fill=NA,
      linewidth=0.7
    ),
    axis.ticks=element_line(
      linewidth=0.7
    ),
    axis.text.y=element_text(
      size=10
    )
  )

ggsave(
  out_file("Integrated_RF_PermutationImportance_Top15_50.tiff"),
  plot=p_rf_imp,
  width=8,
  height=6.5,
  units="in",
  dpi=600,
  compression="lzw"
)


# ============================================================
# 44. Calibration of all six algorithms in temporal validation
# Decile calibration curves, Brier score, calibration intercept and slope.
# ============================================================
# INTEGRATED MODEL
# Calibration evaluation for ALL SIX algorithms
# No additional recalibration
############################################################


# ============================================================
# 1. Outcome + six model probabilities
# ============================================================

y_cal <- ifelse(
  val_list[[1]]$MCQ160C=="Yes",
  1,0
)

prob_list <- list(
  "Random Forest"=rf_final_prob,
  "CatBoost"=cb_final_prob,
  "SVM"=svm_final_prob,
  "Logistic Regression"=lr_final_prob_raw,
  "BP"=bp_final_prob,
  "DNN"=dnn_final_prob
)

stopifnot(
  all(
    sapply(prob_list,length)==length(y_cal)
  ),
  !anyNA(y_cal),
  !any(
    sapply(prob_list,anyNA)
  )
)


# ============================================================
# 2. Calibration statistics
# Brier + intercept + slope + AUROC
# ============================================================

eps <- 1e-6

calibration_statistics <- bind_rows(
  lapply(names(prob_list),function(m){
    
    prob_raw <- as.numeric(
      prob_list[[m]]
    )
    
    prob_clip <- pmin(
      pmax(prob_raw,eps),
      1-eps
    )
    
    logit_prob <- qlogis(
      prob_clip
    )
    
    # Calibration-in-the-large:
    # ideal intercept = 0
    fit_intercept <- glm(
      y_cal~1+offset(logit_prob),
      family=binomial()
    )
    
    # Calibration slope:
    # ideal slope = 1
    fit_slope <- glm(
      y_cal~logit_prob,
      family=binomial()
    )
    
    r <- roc(
      y_cal,
      prob_raw,
      levels=c(0,1),
      direction="<",
      quiet=TRUE
    )
    
    ci_r <- ci.auc(
      r,
      method="delong"
    )
    
    data.frame(
      Model=m,
      AUROC=as.numeric(auc(r)),
      AUROC_Lower=as.numeric(ci_r[1]),
      AUROC_Upper=as.numeric(ci_r[3]),
      Brier=mean(
        (y_cal-prob_raw)^2
      ),
      Calibration_Intercept=
        unname(coef(fit_intercept)[1]),
      Calibration_Slope=
        unname(coef(fit_slope)["logit_prob"])
    )
  })
) %>%
  arrange(desc(AUROC))

print(calibration_statistics)

write.csv(
  calibration_statistics,
  out_file("Integrated_AllModels_Calibration_Statistics.csv"),
  row.names=FALSE
)


# ============================================================
# 3. Decile calibration data for all six algorithms
# ============================================================

calibration_curve_data <- bind_rows(
  lapply(names(prob_list),function(m){
    
    data.frame(
      Model=m,
      outcome=y_cal,
      prob=as.numeric(prob_list[[m]])
    ) %>%
      mutate(
        group=ntile(prob,10)
      ) %>%
      group_by(
        Model,
        group
      ) %>%
      summarise(
        Predicted=mean(prob),
        Observed=mean(outcome),
        N=n(),
        .groups="drop"
      )
  })
)

calibration_curve_data$Model <- factor(
  calibration_curve_data$Model,
  levels=c(
    "Random Forest",
    "CatBoost",
    "SVM",
    "Logistic Regression",
    "BP",
    "DNN"
  )
)

print(calibration_curve_data)

write.csv(
  calibration_curve_data,
  out_file("Integrated_AllModels_Calibration_Curve_Data.csv"),
  row.names=FALSE
)


# ============================================================
# 4. Six-model calibration figure
# Same visual style, 6 panels for direct comparison
# ============================================================
# ------------------------------------------------------------
# Mini legend repeated in every panel
# ------------------------------------------------------------

legend_panel <- expand.grid(
  Model=levels(calibration_curve_data$Model),
  Type=c(
    "Observed calibration",
    "Ideal calibration"
  ),
  stringsAsFactors=FALSE
)

legend_panel$Model <- factor(
  legend_panel$Model,
  levels=levels(calibration_curve_data$Model)
)

legend_panel$y <- ifelse(
  legend_panel$Type=="Observed calibration",
  0.18,
  0.10
)

legend_panel$x1 <- 0.48
legend_panel$x2 <- 0.60
legend_panel$x_text <- 0.63

legend_panel$LineType <- ifelse(
  legend_panel$Type=="Observed calibration",
  "solid",
  "dashed"
)


# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------

p_cal_all <- ggplot(
  calibration_curve_data,
  aes(
    x=Predicted,
    y=Observed
  )
) +
  
  # Observed calibration
  geom_line(
    linewidth=0.9
  ) +
  
  geom_point(
    size=2.2
  ) +
  
  # Ideal calibration
  geom_abline(
    intercept=0,
    slope=1,
    linetype="dashed",
    linewidth=0.7
  ) +
  
  # ----------------------------------------------------------
# Mini legend line in each panel
# ----------------------------------------------------------
geom_segment(
  data=legend_panel,
  aes(
    x=x1,
    xend=x2,
    y=y,
    yend=y,
    linetype=LineType
  ),
  inherit.aes=FALSE,
  linewidth=0.7,
  show.legend=FALSE
) +
  
  # Mini legend text in each panel
  geom_text(
    data=legend_panel,
    aes(
      x=x_text,
      y=y,
      label=Type
    ),
    inherit.aes=FALSE,
    hjust=0,
    vjust=0.5,
    size=3.0
  ) +
  
  scale_linetype_identity() +
  
  facet_wrap(
    ~Model,
    ncol=3
  ) +
  
  scale_x_continuous(
    limits=c(0,1),
    breaks=seq(0,1,0.2),
    expand=c(0,0)
  ) +
  
  scale_y_continuous(
    limits=c(0,1),
    breaks=seq(0,1,0.2),
    expand=c(0,0)
  ) +
  
  labs(
    x="Predicted probability",
    y="Observed probability"
  ) +
  
  theme_classic(
    base_size=12
  ) +
  
  theme(
    axis.line=element_blank(),
    
    panel.border=element_rect(
      colour="black",
      fill=NA,
      linewidth=0.7
    ),
    
    axis.ticks=element_line(
      linewidth=0.7
    ),
    
    strip.background=element_blank(),
    
    strip.text=element_text(
      size=11
    )
  )


# ============================================================
# Save
# ============================================================

ggsave(
  out_file("Integrated_AllModels_Calibration.tiff"),
  plot=p_cal_all,
  width=10,
  height=7,
  units="in",
  dpi=600,
  compression="lzw"
)

# ============================================================
# 45. Individual calibration plots (one per algorithm)
# Same style as previous RF calibration figure
# ============================================================

dir.create(
  out_file("Integrated_Calibration_Plots"),
  showWarnings=FALSE,
  recursive=TRUE
)

legend_tbl <- tibble(
  lab=c(
    "Observed calibration",
    "Ideal calibration"
  ),
  x=list(
    c(0,0.2),
    c(0,0.2)
  ),
  y=list(
    c(-10,-10),
    c(-10,-10)
  )
) %>%
  unnest(c(x,y))

for(m in levels(calibration_curve_data$Model)){
  
  dat_m <- calibration_curve_data %>%
    filter(Model==m)
  
  p <- ggplot() +
    
    geom_line(
      data=dat_m,
      aes(
        x=Predicted,
        y=Observed,
        linetype="Observed calibration"
      ),
      linewidth=0.9
    ) +
    
    geom_point(
      data=dat_m,
      aes(
        x=Predicted,
        y=Observed
      ),
      size=2.2
    ) +
    
    geom_abline(
      intercept=0,
      slope=1,
      linetype="dashed",
      linewidth=0.7,
      show.legend=FALSE
    ) +
    
    geom_path(
      data=legend_tbl,
      aes(
        x=x,
        y=y,
        linetype=lab
      ),
      linewidth=0.7
    ) +
    
    scale_linetype_manual(
      name=NULL,
      values=c(
        "Observed calibration"="solid",
        "Ideal calibration"="dashed"
      ),
      breaks=c(
        "Observed calibration",
        "Ideal calibration"
      )
    ) +
    
    scale_x_continuous(
      limits=c(0,1),
      breaks=seq(0,1,0.2),
      expand=c(0,0)
    ) +
    
    scale_y_continuous(
      limits=c(0,1),
      breaks=seq(0,1,0.2),
      expand=c(0,0)
    ) +
    
    labs(
      x="Predicted probability",
      y="Observed probability"
    ) +
    
    theme_classic(
      base_size=12
    ) +
    
    theme(
      axis.line=element_blank(),
      
      panel.border=element_rect(
        colour="black",
        fill=NA,
        linewidth=0.7
      ),
      
      axis.ticks=element_line(
        linewidth=0.7
      ),
      
      legend.position=c(
        0.72,
        0.24
      ),
      
      legend.background=
        element_blank(),
      
      legend.text=
        element_text(
          size=12
        ),
      
      legend.key.width=
        unit(
          1.1,
          "cm"
        )
    )
  
  print(p)
  
  file_name <- gsub(
    "[^A-Za-z0-9]+",
    "_",
    m
  )
  
  ggsave(
    file.path(
      out_file("Integrated_Calibration_Plots"),
      paste0(
        "Integrated_",
        file_name,
        "_Calibration.tiff"
      )
    ),
    plot=p,
    width=6,
    height=6,
    units="in",
    dpi=600,
    compression="lzw"
  )
}


# ============================================================
# 46. Compact calibration comparison table
# ============================================================

calibration_comparison <- calibration_statistics %>%
  transmute(
    Model,
    AUROC=round(AUROC,4),
    `95% CI`=paste0(
      round(AUROC_Lower,4),
      "–",
      round(AUROC_Upper,4)
    ),
    Brier=round(Brier,4),
    Calibration_Intercept=
      round(Calibration_Intercept,4),
    Calibration_Slope=
      round(Calibration_Slope,4)
  )

print(
  calibration_comparison,
  row.names=FALSE
)

write.csv(
  calibration_comparison,
  out_file("Integrated_AllModels_Calibration_Comparison.csv"),
  row.names=FALSE
)


# ============================================================
# 47. Save calibration outputs
# ============================================================

save(
  calibration_statistics,
  calibration_comparison,
  calibration_curve_data,
  p_cal_all,
  file=
    out_file("Integrated_AllModels_Calibration_Results.RData")
)

cat(
  "\nSix-model calibration evaluation completed.\n"
)


# ============================================================
# Reproducibility metadata
# ============================================================
writeLines(capture.output(sessionInfo()), out_file("sessionInfo.txt"))
cat("\nCompleted. All generated results are in: ",
    normalizePath(OUTPUT_DIR), "\n", sep = "")
