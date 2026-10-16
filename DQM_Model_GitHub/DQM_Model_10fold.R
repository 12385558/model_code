# ============================================================
# Dietary and Questionnaire Model | NHANES 2007-2018
# Reproducible model development and independent temporal validation
# ============================================================
# Outcome: MCQ160C is the STUDY-DEFINED composite outcome combining
# self-reported coronary heart disease, angina and myocardial infarction;
# it is not the original single NHANES questionnaire item MCQ160C.
#
# Development period: 2007-2014. Temporal validation: 2015-2018.
# Inputs: 3 matched multiply imputed development/validation datasets,
# and the 30 LASSO-selected Dietary and Questionnaire Model predictors.
#
# STRATIFIED 10-FOLD CROSS-VALIDATION:
# - Subjects use identical fold memberships in each of the 3 MI sets.
# - AUROC is averaged over all 10 folds and all 3 MI datasets.
# - SMOTENC, scaling, and dummy encoding are fitted on training rows only.
# - SVM Platt scaling uses development out-of-fold decision scores.
# - Independent temporal validation is never used for tuning or cutoffs.
# - Youden thresholds are selected from development OOF predictions.
#
# IMPORTANT: Cohort selection, multiple imputation and LASSO screening
# have already been performed upstream and are not repeated here.
# ============================================================

INPUT_DIR <- Sys.getenv("NHANES_INPUT_DIR", unset = ".")
OUTPUT_DIR <- Sys.getenv("NHANES_OUTPUT_DIR", unset = "outputs")
SEED <- 2026L
N_MI <- 3L
CV_FOLDS <- 10L
N_BOOT <- 2000L                   # Bootstrap replications for non-AUROC CIs
RUN_RF_PERMUTATION <- FALSE      # Optional: slow (30 predictors x 50 permutations)
B_PERM <- 50L

if (!dir.exists(INPUT_DIR)) stop("Input directory not found: ", INPUT_DIR)
dir.create(OUTPUT_DIR, recursive = TRUE, showWarnings = FALSE)
in_file <- function(name) file.path(INPUT_DIR, name)
out_file <- function(name) file.path(OUTPUT_DIR, name)

library(caret)
library(recipes)
library(themis)
library(pROC)
library(PRROC)
library(dplyr)
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
# 2. Read 30 stable LASSO-selected DQM variables
# ============================================================

dqm_vars <- read.csv(
  in_file("DQM_final_variables.csv"),
  stringsAsFactors=FALSE,
  check.names=FALSE
)$Variable

dqm_vars <- unique(trimws(as.character(dqm_vars)))
dqm_vars <- dqm_vars[!is.na(dqm_vars) & dqm_vars!=""]

cat("DQM variables =", length(dqm_vars), "\n")
print(dqm_vars)
stopifnot(length(dqm_vars) == 30L)
# All 30 variable identities are read from the upstream LASSO export.
# Do not assume a fixed variable set beyond the provided selection file.


for(i in seq_len(N_MI)){
  stopifnot(
    all(c("SEQN",dqm_vars,"MCQ160C") %in% names(dev_list[[i]])),
    all(c("SEQN",dqm_vars,"MCQ160C") %in% names(val_list[[i]]))
  )
}


# ============================================================
# 3. Variable types
# ============================================================

# Keep the original DQM script's variable classifications.
# The LASSO export is the source of predictor names; this list supplies types.
# Check any discretized continuous NHANES variables against upstream coding.
categorical_all <- c(
  "RIDRETH1", "DMDEDUC2", "BPQ070", "DBQ197", "DBQ229", "DR1_300",
  "DRQSPREP", "HSD010", "KIQ005", "OCD150", "BMXBMI", "WHQ030",
  "WHQ040", "RIAGENDR", "RIDAGEYR", "DMDMARTL", "INDFMPIR", "BPQ020",
  "BPQ080", "BPXPULS", "DIQ010", "DIQ170", "DIQ180", "DIQ050",
  "DRQSDIET", "DRD340", "DRD360", "HIQ011", "HIQ270", "HSQ500",
  "HSQ510", "HSQ520", "HSQ571", "HSQ590", "KIQ022", "KIQ042",
  "KIQ044", "KIQ046", "MCQ010", "MCQ053", "MCQ080", "MCQ092",
  "MCQ160G", "MCQ160M", "MCQ160K", "MCQ160L", "MCQ220", "MCQ300A",
  "MCQ300B", "MCQ300C", "SMQ020", "WHQ070", "SLQ050"
)

categorical_vars <- intersect(categorical_all,dqm_vars)
numeric_vars <- setdiff(dqm_vars,categorical_vars)

cat("\nCategorical =",length(categorical_vars),"\n")
print(categorical_vars)
cat("\nNumeric =",length(numeric_vars),"\n")


# Development-derived factor levels only
factor_levels <- lapply(categorical_vars,function(v){
  x <- unlist(lapply(dev_list,function(d) as.character(d[[v]])))
  sort(unique(x[!is.na(x)]))
})
names(factor_levels) <- categorical_vars


# ============================================================
# 4. Format datasets
# ============================================================

for(i in seq_len(N_MI)){
  
  dev_list[[i]] <- dev_list[[i]][,c("SEQN",dqm_vars,"MCQ160C"),drop=FALSE]
  val_list[[i]] <- val_list[[i]][,c("SEQN",dqm_vars,"MCQ160C"),drop=FALSE]
  
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
  
  dev_list[[i]][numeric_vars] <- lapply(dev_list[[i]][numeric_vars],as.numeric)
  val_list[[i]][numeric_vars] <- lapply(val_list[[i]][numeric_vars],as.numeric)
  
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
    !anyNA(dev_list[[i]][,c(dqm_vars,"MCQ160C")]),
    !anyNA(val_list[[i]][,c(dqm_vars,"MCQ160C")])
  )
}

cat("\nDevelopment N =",nrow(dev_list[[1]]),"\n")
cat("Temporal Validation N =",nrow(val_list[[1]]),"\n")
print(table(dev_list[[1]]$MCQ160C))
print(table(val_list[[1]]$MCQ160C))
# Identical SEQN and disease outcomes across MI datasets are mandatory
# before applying shared row-based fold indexes.
stopifnot(
  !anyDuplicated(dev_list[[1]]$SEQN),
  !anyDuplicated(val_list[[1]]$SEQN),
  all(vapply(dev_list, function(z)
    identical(z$MCQ160C, dev_list[[1]]$MCQ160C), logical(1))),
  all(vapply(val_list, function(z)
    identical(z$MCQ160C, val_list[[1]]$MCQ160C), logical(1)))
)


# ============================================================
# 5. Shared subject-level stratified 10-fold CV
# ============================================================

# Input SEQN row alignment is verified above; one fold split is reused.
set.seed(SEED)
folds <- createFolds(
  dev_list[[1]]$MCQ160C,
  k = CV_FOLDS,
  returnTrain = FALSE
)
stopifnot(length(folds) == CV_FOLDS)
stopifnot(identical(sort(as.integer(unlist(folds, use.names = FALSE))),
                    seq_len(nrow(dev_list[[1]]))))
stopifnot(all(sapply(folds, function(id) length(unique(dev_list[[1]]$MCQ160C[id])) == 2L)))


# ============================================================
# 6. Fold-safe preprocessing; SMOTENC trains only on training rows
# ============================================================

prepare_pair <- function(train_dat,valid_dat,vars,
                         normalize=FALSE,dummy=FALSE,
                         do_smotenc=TRUE,seed=2026){
  
  train0 <- train_dat[,c(vars,"MCQ160C"),drop=FALSE]
  valid0 <- valid_dat[,c(vars,"MCQ160C"),drop=FALSE]
  
  rec <- recipe(MCQ160C~.,data=train0)
  
  if(do_smotenc)
    rec <- rec %>%
    step_smotenc(MCQ160C,over_ratio=1,neighbors=5,
                 skip=TRUE,seed=seed)
  
  if(normalize)
    rec <- rec %>% step_normalize(all_numeric_predictors())
  
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
  as.numeric(auc(roc(
    y,p,levels=c("No","Yes"),
    direction="<",quiet=TRUE
  )))
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


# ============================================================
# 8. H2O / DNN helpers
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
    input_dropout_ratio=0.1,
    standardize=FALSE,
    balance_classes=FALSE,
    reproducible=TRUE,
    seed=seed
  )
  
  if(activation=="RectifierWithDropout")
    pars$hidden_dropout_ratios <- rep(0.1,length(hidden))
  
  list(
    model=do.call(h2o.deeplearning,pars),
    frame=htr
  )
}

# ============================================================
# 9. BP tuning: development 10-fold CV x 3 MI datasets
# ============================================================

bp_grid <- expand.grid(
  size=c(5,10,20),
  decay=c(0.01,0.10)
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
      dqm_vars,TRUE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train; va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    set.seed(2026+k)
    
    fit <- nnet(
      x=as.matrix(tr[,x,drop=FALSE]),
      y=ifelse(tr$MCQ160C=="Yes",1,0),
      size=bp_grid$size[g],
      decay=bp_grid$decay[g],
      maxit=500,trace=FALSE,MaxNWts=50000
    )
    
    pr <- as.numeric(predict(
      fit,as.matrix(va[,x,drop=FALSE]),type="raw"
    ))
    
    a <- c(a,get_auc(va$MCQ160C,pr))
  }
  
  bp_result$Mean_ROC[g] <- mean(a)
}

bp_result <- bp_result[order(-bp_result$Mean_ROC),]
best_size <- bp_result$size[1]
best_decay <- bp_result$decay[1]

print(bp_result)
write.csv(bp_result,out_file("DQM_BP_CV_results.csv"),row.names=FALSE)
write.csv(bp_result[1,],out_file("DQM_BP_bestTune.csv"),row.names=FALSE)


# ============================================================
# 10. SVM tuning
# ============================================================

svm_grid <- data.frame(C=c(0.1,1,10))
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
      dqm_vars,TRUE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train; va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    xtr <- as.matrix(tr[,x,drop=FALSE])
    xva <- as.matrix(va[,x,drop=FALSE])
    ytr <- ifelse(tr$MCQ160C=="Yes",1,0)
    
    fit <- LiblineaR(
      xtr,ytr,type=svm_type,
      cost=svm_grid$C[g],
      bias=1,verbose=FALSE
    )
    
    s_tr <- get_decision_score(fit,xtr)
    s_va <- orient_svm_score(
      s_tr,ytr,get_decision_score(fit,xva)
    )
    
    a <- c(a,get_auc(va$MCQ160C,s_va))
  }
  
  svm_result$Mean_ROC[g] <- mean(a)
}

svm_result <- svm_result[order(-svm_result$Mean_ROC),]
best_C <- svm_result$C[1]

print(svm_result)
write.csv(svm_result,out_file("DQM_SVM_CV_results.csv"),row.names=FALSE)
write.csv(
  data.frame(C=best_C,Type=2,Mean_ROC=svm_result$Mean_ROC[1]),
  out_file("DQM_SVM_bestTune.csv"),row.names=FALSE
)


# ============================================================
# 11. RF tuning (number of dummy predictors from development data)
# ============================================================

pp0 <- prepare_pair(
  dev_list[[1]],dev_list[[1]],dqm_vars,
  FALSE,TRUE,FALSE
)

p_rf <- ncol(pp0$train)-1

mtry_grid <- sort(unique(
  pmax(1,pmin(
    p_rf,
    round(c(sqrt(p_rf),p_rf/3,p_rf/2))
  ))
))

rf_result <- data.frame(mtry=mtry_grid,Mean_ROC=NA_real_)
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
      dqm_vars,FALSE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train; va <- pp$valid
    tr$MCQ160C <- factor(tr$MCQ160C,levels=c("No","Yes"))
    
    fit <- ranger(
      MCQ160C~.,data=tr,probability=TRUE,
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
write.csv(rf_result,out_file("DQM_RF_CV_results.csv"),row.names=FALSE)
write.csv(
  data.frame(
    mtry=best_mtry,
    num.trees=500,
    min.node.size=5,
    Mean_ROC=rf_result$Mean_ROC[1]
  ),
  out_file("DQM_RF_bestTune.csv"),row.names=FALSE
)


# ============================================================
# 12. CatBoost tuning
# ============================================================

cb_grid <- expand.grid(
  depth=c(4,6),
  learning_rate=c(0.03,0.10),
  l2_leaf_reg=c(3,5)
)
cb_result <- transform(cb_grid,Mean_ROC=NA_real_)
CB_ITER <- 300

for(g in 1:nrow(cb_grid)){
  
  a <- c()
  cat("\nCatBoost:",g,"/",nrow(cb_grid),"\n")
  
  for(i in seq_len(N_MI)) for(k in seq_along(folds)){
    
    id <- folds[[k]]
    
    pp <- prepare_pair(
      dev_list[[i]][-id,,drop=FALSE],
      dev_list[[i]][id,,drop=FALSE],
      dqm_vars,FALSE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train; va <- pp$valid
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
      fit,pva,prediction_type="Probability"
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
write.csv(cb_result,out_file("DQM_CatBoost_CV_results.csv"),row.names=FALSE)
write.csv(
  data.frame(
    depth=best_depth,
    learning_rate=best_lr,
    l2_leaf_reg=best_l2,
    iterations=300,
    Mean_ROC=cb_result$Mean_ROC[1]
  ),
  out_file("DQM_CatBoost_bestTune.csv"),row.names=FALSE
)


# ============================================================
# 13. DNN tuning
# ============================================================

dnn_grid <- expand.grid(
  architecture=c("30-30-30","64-32-16","64-64-32"),
  activation=c("Rectifier","RectifierWithDropout"),
  stringsAsFactors=FALSE
)

get_hidden <- function(x)
  as.numeric(strsplit(as.character(x),"-")[[1]])

dnn_result <- transform(dnn_grid,Mean_ROC=NA_real_)

for(g in 1:nrow(dnn_grid)){
  
  a <- c()
  hidden <- get_hidden(dnn_grid$architecture[g])
  activation <- dnn_grid$activation[g]
  
  cat("\nDNN:",dnn_grid$architecture[g],activation,"\n")
  
  for(i in seq_len(N_MI)) for(k in seq_along(folds)){
    
    id <- folds[[k]]
    
    pp <- prepare_pair(
      dev_list[[i]][-id,,drop=FALSE],
      dev_list[[i]][id,,drop=FALSE],
      dqm_vars,TRUE,TRUE,TRUE,2026+k
    )
    
    obj <- fit_h2o_dnn(
      pp$train,hidden,activation,2026+k
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

best_hidden <- get_hidden(dnn_result$architecture[1])
best_activation <- as.character(dnn_result$activation[1])

print(dnn_result)

write.csv(dnn_result,out_file("DQM_DNN_CV_results.csv"),row.names=FALSE)

write.csv(
  data.frame(
    architecture=dnn_result$architecture[1],
    activation=best_activation,
    epochs=30,l2=1e-4,
    Mean_ROC=dnn_result$Mean_ROC[1]
  ),
  out_file("DQM_DNN_bestTune.csv"),row.names=FALSE
)


# ============================================================
# 14. Best hyperparameters
# ============================================================

cat("\n===== BEST PARAMETERS =====\n")
cat("BP:",best_size,best_decay,"\n")
cat("SVM C:",best_C,"\n")
cat("RF mtry:",best_mtry,"\n")
cat("CatBoost:",best_depth,best_lr,best_l2,"\n")
cat("DNN:",paste(best_hidden,collapse="-"),best_activation,"\n")


# ============================================================
# 15. Final BP
# ============================================================

bp_final_models <- bp_final_recipes <- bp_prob_mi <- vector("list",N_MI)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    dqm_vars,TRUE,TRUE,TRUE,2026
  )
  
  tr <- pp$train; va <- pp$valid
  x <- setdiff(names(tr),"MCQ160C")
  
  set.seed(SEED)
  
  fit <- nnet(
    x=as.matrix(tr[,x,drop=FALSE]),
    y=ifelse(tr$MCQ160C=="Yes",1,0),
    size=best_size,decay=best_decay,
    maxit=500,trace=FALSE,MaxNWts=50000
  )
  
  bp_prob_mi[[i]] <- as.numeric(
    predict(fit,as.matrix(va[,x,drop=FALSE]),type="raw")
  )
  
  bp_final_models[[i]] <- fit
  bp_final_recipes[[i]] <- pp$recipe
}

bp_final_prob <- rowMeans(do.call(cbind,bp_prob_mi))
bp_actual <- val_list[[1]]$MCQ160C
r_bp <- roc(bp_actual,bp_final_prob,
            levels=c("No","Yes"),direction="<",quiet=TRUE)

cat("\nBP AUROC =",as.numeric(auc(r_bp)),"\n")

save(
  bp_final_models,bp_final_recipes,bp_prob_mi,
  bp_final_prob,bp_actual,r_bp,
  file=out_file("DQM_BP_results.RData")
)


# ============================================================
# 16. Final SVM + Platt scaling fitted to 10-fold OOF scores
# Platt calibration is fitted on development data only.
# ============================================================

svm_final_models <- svm_final_recipes <-
  svm_platt_models <- svm_prob_mi <-
  svm_score_mi <- svm_oof_prob_mi <- vector("list",N_MI)

for(i in seq_len(N_MI)){
  
  dat <- dev_list[[i]]
  oof_score <- rep(NA_real_,nrow(dat))
  
  for(k in seq_along(folds)){
    
    id <- folds[[k]]
    
    pp <- prepare_pair(
      dat[-id,,drop=FALSE],dat[id,,drop=FALSE],
      dqm_vars,TRUE,TRUE,TRUE,2026+k
    )
    
    tr <- pp$train; va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    xtr <- as.matrix(tr[,x,drop=FALSE])
    xva <- as.matrix(va[,x,drop=FALSE])
    ytr <- ifelse(tr$MCQ160C=="Yes",1,0)
    
    fit <- LiblineaR(
      xtr,ytr,type=2,cost=best_C,
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
  
  svm_oof_prob_mi[[i]] <- predict(
    platt_i,
    newdata=data.frame(oof_score=oof_score),
    type="response"
  )
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    dqm_vars,TRUE,TRUE,TRUE,2026
  )
  
  tr <- pp$train; va <- pp$valid
  x <- setdiff(names(tr),"MCQ160C")
  
  xtr <- as.matrix(tr[,x,drop=FALSE])
  xva <- as.matrix(va[,x,drop=FALSE])
  ytr <- ifelse(tr$MCQ160C=="Yes",1,0)
  
  fit <- LiblineaR(
    xtr,ytr,type=2,cost=best_C,
    bias=1,verbose=FALSE
  )
  
  s_va <- orient_svm_score(
    get_decision_score(fit,xtr),
    ytr,
    get_decision_score(fit,xva)
  )
  
  pr <- predict(
    platt_i,
    newdata=data.frame(oof_score=s_va),
    type="response"
  )
  
  svm_final_models[[i]] <- fit
  svm_final_recipes[[i]] <- pp$recipe
  svm_platt_models[[i]] <- platt_i
  svm_score_mi[[i]] <- s_va
  svm_prob_mi[[i]] <- as.numeric(pr)
}

svm_final_prob <- rowMeans(do.call(cbind,svm_prob_mi))
svm_final_score <- rowMeans(do.call(cbind,svm_score_mi))
svm_oof_prob <- rowMeans(do.call(cbind,svm_oof_prob_mi))

svm_actual <- val_list[[1]]$MCQ160C
r_svm <- roc(svm_actual,svm_final_prob,
             levels=c("No","Yes"),direction="<",quiet=TRUE)

cat("\nSVM AUROC =",as.numeric(auc(r_svm)),"\n")

save(
  svm_final_models,svm_final_recipes,svm_platt_models,
  svm_score_mi,svm_prob_mi,svm_final_score,
  svm_final_prob,svm_oof_prob,svm_actual,r_svm,
  file=out_file("DQM_SVM_results.RData")
)


# ============================================================
# 17. Final RF
# ============================================================

rf_final_models <- rf_final_recipes <- rf_prob_mi <- vector("list",N_MI)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    dqm_vars,FALSE,TRUE,TRUE,2026
  )
  
  tr <- pp$train; va <- pp$valid
  tr$MCQ160C <- factor(tr$MCQ160C,levels=c("No","Yes"))
  
  fit <- ranger(
    MCQ160C~.,data=tr,
    probability=TRUE,
    num.trees=500,
    mtry=best_mtry,
    min.node.size=5,
    importance="impurity",
    seed=2026
  )
  
  rf_prob_mi[[i]] <- predict(
    fit,
    data=va[,setdiff(names(va),"MCQ160C"),drop=FALSE]
  )$predictions[,"Yes"]
  
  rf_final_models[[i]] <- fit
  rf_final_recipes[[i]] <- pp$recipe
}

rf_final_prob <- rowMeans(do.call(cbind,rf_prob_mi))
rf_actual <- val_list[[1]]$MCQ160C
r_rf <- roc(rf_actual,rf_final_prob,
            levels=c("No","Yes"),direction="<",quiet=TRUE)

cat("\nRF AUROC =",as.numeric(auc(r_rf)),"\n")

save(
  rf_final_models,rf_final_recipes,rf_prob_mi,
  rf_final_prob,rf_actual,r_rf,
  file=out_file("DQM_RF_results.RData")
)


# ============================================================
# 18. Final CatBoost
# ============================================================

cb_final_models <- cb_final_recipes <- cb_prob_mi <- vector("list",N_MI)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    dqm_vars,FALSE,TRUE,TRUE,2026
  )
  
  tr <- pp$train; va <- pp$valid
  x <- setdiff(names(tr),"MCQ160C")
  
  ptr <- catboost.load_pool(
    as.matrix(tr[,x,drop=FALSE]),
    label=ifelse(tr$MCQ160C=="Yes",1,0)
  )
  pva <- catboost.load_pool(as.matrix(va[,x,drop=FALSE]))
  
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
    catboost.predict(fit,pva,prediction_type="Probability")
  )
  
  cb_final_models[[i]] <- fit
  cb_final_recipes[[i]] <- pp$recipe
}

cb_final_prob <- rowMeans(do.call(cbind,cb_prob_mi))
cb_actual <- val_list[[1]]$MCQ160C
r_cb <- roc(cb_actual,cb_final_prob,
            levels=c("No","Yes"),direction="<",quiet=TRUE)

cat("\nCatBoost AUROC =",as.numeric(auc(r_cb)),"\n")

save(
  cb_final_models,cb_final_recipes,cb_prob_mi,
  cb_final_prob,cb_actual,r_cb,
  file=out_file("DQM_CatBoost_results.RData")
)


# ============================================================
# 19. Logistic regression baseline + SMOTENC (no hyperparameter tuning)
# ============================================================

lr_final_models <- lr_final_recipes <- lr_prob_mi <- vector("list",N_MI)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    dqm_vars,FALSE,FALSE,TRUE,2026
  )
  
  tr <- pp$train; va <- pp$valid
  tr$Y <- ifelse(tr$MCQ160C=="Yes",1,0)
  
  fit <- glm(
    Y~.-MCQ160C,
    data=tr,
    family=binomial()
  )
  
  lr_prob_mi[[i]] <- as.numeric(
    predict(fit,newdata=va,type="response")
  )
  
  lr_final_models[[i]] <- fit
  lr_final_recipes[[i]] <- pp$recipe
}

lr_final_prob_raw <- rowMeans(do.call(cbind,lr_prob_mi))
lr_actual <- val_list[[1]]$MCQ160C
r_lr <- roc(lr_actual,lr_final_prob_raw,
            levels=c("No","Yes"),direction="<",quiet=TRUE)

cat("\nLogistic AUROC =",as.numeric(auc(r_lr)),"\n")

save(
  lr_final_models,lr_final_recipes,lr_prob_mi,
  lr_final_prob_raw,lr_actual,r_lr,
  file=out_file("DQM_Logistic_SMOTENC_results.RData")
)


# ============================================================
# 20. Final DNN
# ============================================================

dnn_prob_mi <- vector("list",N_MI)
dnn_final_recipes <- vector("list",N_MI)
dnn_model_paths <- character(N_MI)

dir.create(out_file("DQM_DNN_models"),showWarnings=FALSE)

for(i in seq_len(N_MI)){
  
  pp <- prepare_pair(
    dev_list[[i]],val_list[[i]],
    dqm_vars,TRUE,TRUE,TRUE,2026
  )
  
  obj <- fit_h2o_dnn(
    pp$train,best_hidden,best_activation,2026
  )
  
  hva <- as.h2o(pp$valid)
  hva$MCQ160C <- as.factor(hva$MCQ160C)
  
  dnn_prob_mi[[i]] <- as.numeric(
    get_h2o_prob(obj$model,hva)
  )
  
  dnn_model_paths[i] <- h2o.saveModel(
    obj$model,
    path=out_file("DQM_DNN_models"),
    force=TRUE
  )
  
  dnn_final_recipes[[i]] <- pp$recipe
  
  h2o.rm(obj$model)
  h2o.rm(obj$frame)
  h2o.rm(hva)
}

dnn_final_prob <- rowMeans(do.call(cbind,dnn_prob_mi))
dnn_actual <- val_list[[1]]$MCQ160C
r_dnn <- roc(dnn_actual,dnn_final_prob,
             levels=c("No","Yes"),direction="<",quiet=TRUE)

cat("\nDNN AUROC =",as.numeric(auc(r_dnn)),"\n")

save(
  dnn_prob_mi,dnn_final_prob,dnn_final_recipes,
  dnn_model_paths,dnn_actual,r_dnn,
  file=out_file("DQM_DNN_results.RData")
)


# ============================================================
# 21. Independent temporal validation AUROC
# ============================================================

roc_list <- list(
  BP=r_bp,
  SVM=r_svm,
  RF=r_rf,
  CatBoost=r_cb,
  Logistic=r_lr,
  DNN=r_dnn
)

auc_table <- bind_rows(lapply(names(roc_list),function(m){
  
  r <- roc_list[[m]]
  ci <- ci.auc(r,method="delong")
  
  data.frame(
    Model=m,
    AUROC=as.numeric(auc(r)),
    Lower95CI=as.numeric(ci[1]),
    Upper95CI=as.numeric(ci[3])
  )
})) %>% arrange(desc(AUROC))

print(auc_table)

write.csv(
  auc_table,
  out_file("DQM_TemporalValidation_AUROC.csv"),
  row.names=FALSE
)


# ============================================================
# 22. Combined ROC plot
# ============================================================

roc_plot_list <- roc_list[auc_table$Model]

cols <- c(
  "#D55E00","#0072B2","#009E73",
  "#CC79A7","#E69F00","#56B4E9"
)

legend_text <- paste0(
  auc_table$Model,
  ", AUC = ",
  sprintf("%.3f",auc_table$AUROC)
)

tiff(
  out_file("DQM_Model_ROC.tiff"),
  width=6.5,height=6.5,
  units="in",res=600,
  compression="lzw"
)

plot(
  roc_plot_list[[1]],
  col=cols[1],lwd=2,
  legacy.axes=TRUE,
  xlim=c(1,0),ylim=c(0,1),
  xaxs="i",yaxs="i",
  xlab="1 - Specificity",
  ylab="Sensitivity"
)

for(i in 2:length(roc_plot_list))
  plot(
    roc_plot_list[[i]],
    col=cols[i],lwd=2,
    add=TRUE,legacy.axes=TRUE
  )

legend(
  "bottomright",
  legend=legend_text,
  col=cols,lwd=2,
  bty="n",cex=0.8
)

dev.off()


# ============================================================
# 23. OOF helper: selected models are refit on each training fold
# ============================================================

fit_oof_model <- function(model,tr_raw,va_raw,seed){
  
  if(model=="BP"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,dqm_vars,
      TRUE,TRUE,TRUE,seed
    )
    tr <- pp$train; va <- pp$valid
    x <- setdiff(names(tr),"MCQ160C")
    
    set.seed(seed)
    
    fit <- nnet(
      x=as.matrix(tr[,x,drop=FALSE]),
      y=ifelse(tr$MCQ160C=="Yes",1,0),
      size=best_size,decay=best_decay,
      maxit=500,trace=FALSE,MaxNWts=50000
    )
    
    return(as.numeric(
      predict(fit,as.matrix(va[,x,drop=FALSE]),type="raw")
    ))
  }
  
  
  if(model=="RF"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,dqm_vars,
      FALSE,TRUE,TRUE,seed
    )
    
    tr <- pp$train; va <- pp$valid
    tr$MCQ160C <- factor(tr$MCQ160C,levels=c("No","Yes"))
    
    fit <- ranger(
      MCQ160C~.,data=tr,
      probability=TRUE,
      num.trees=500,
      mtry=best_mtry,
      min.node.size=5,
      seed=seed
    )
    
    return(
      predict(
        fit,
        data=va[,setdiff(names(va),"MCQ160C"),drop=FALSE]
      )$predictions[,"Yes"]
    )
  }
  
  
  if(model=="CB"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,dqm_vars,
      FALSE,TRUE,TRUE,seed
    )
    
    tr <- pp$train; va <- pp$valid
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
    
    return(as.numeric(
      catboost.predict(
        fit,pva,prediction_type="Probability"
      )
    ))
  }
  
  
  if(model=="LR"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,dqm_vars,
      FALSE,FALSE,TRUE,seed
    )
    
    tr <- pp$train; va <- pp$valid
    tr$Y <- ifelse(tr$MCQ160C=="Yes",1,0)
    
    fit <- glm(
      Y~.-MCQ160C,
      data=tr,
      family=binomial()
    )
    
    return(as.numeric(
      predict(fit,newdata=va,type="response")
    ))
  }
  
  
  if(model=="DNN"){
    
    pp <- prepare_pair(
      tr_raw,va_raw,dqm_vars,
      TRUE,TRUE,TRUE,seed
    )
    
    obj <- fit_h2o_dnn(
      pp$train,best_hidden,best_activation,seed
    )
    
    hva <- as.h2o(pp$valid)
    hva$MCQ160C <- as.factor(hva$MCQ160C)
    
    pr <- get_h2o_prob(obj$model,hva)
    
    h2o.rm(obj$model)
    h2o.rm(obj$frame)
    h2o.rm(hva)
    
    return(as.numeric(pr))
  }
  
  stop("Unknown model.")
}


# ============================================================
# 24. Development OOF probabilities (same 10 folds across 3 MI)
# ============================================================
# No pre-existing cached OOF files are loaded, since old two-fold
# predictions and predictions from other hyperparameters are invalid.
models <- c("RF", "SVM", "LR", "CB", "DNN", "BP")
oof <- setNames(vector("list", length(models)), models)

for (m in setdiff(models, "SVM")) {
  cat("\nDevelopment OOF model:", m, "\n")
  mi_pred <- vector("list", N_MI)
  for (i in seq_len(N_MI)) {
    dat <- dev_list[[i]]
    pred <- rep(NA_real_, nrow(dat))
    for (k in seq_along(folds)) {
      id <- folds[[k]]
      cat("  ", m, "MI", i, "fold", k, "/", CV_FOLDS, "\n")
      pred[id] <- fit_oof_model(
        m, dat[-id, , drop = FALSE], dat[id, , drop = FALSE],
        seed = SEED + k
      )
    }
    stopifnot(!anyNA(pred))
    mi_pred[[i]] <- pred
  }
  oof[[m]] <- rowMeans(do.call(cbind, mi_pred))
}

# Reuse SVM's 10-fold OOF decision scores converted by its Platt model.
# Note: the Platt fit itself is fitted on the same OOF scores; these
# probabilities are not cross-fitted for the calibration layer.
oof$SVM <- svm_oof_prob
saveRDS(oof, out_file("DQM_Development_10fold_OOF_probabilities.rds"))

# ============================================================
# 25. Validate development OOF probabilities
# ============================================================

oof_check <- sapply(oof,function(x)
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
# 26. Development OOF -> Youden thresholds (no temporal data used)
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
      ret=c("threshold","sensitivity","specificity"),
      transpose=FALSE
    )
    
    # pROC may return multiple tied Youden-optimal thresholds.
    # Use the first entry deterministically and extract by column.
    th <- as.data.frame(th)[1, , drop = FALSE]
    data.frame(
      Model = m,
      Threshold = as.numeric(th[["threshold"]]),
      Development_Sensitivity = as.numeric(th[["sensitivity"]]),
      Development_Specificity = as.numeric(th[["specificity"]])
    )
  })
)

print(threshold_info)

write.csv(
  threshold_info,
  out_file("DQM_Development_10fold_Thresholds.csv"),
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
    y,prob,levels=c(0,1),
    direction="<",quiet=TRUE
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
    nrow=B,ncol=length(nm),
    dimnames=list(NULL,nm)
  )
  
  for(b in 1:B){
    
    id <- sample(seq_along(y),replace=TRUE)
    yy <- y[id]
    pp <- prob[id]
    
    if(length(unique(yy))<2) next
    
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
# 29. Temporal validation metrics and confidence intervals
# ============================================================

results <- list()
confusion <- list()

for(m in models){
  
  cat("Processing",m,"\n")
  
  thr <- threshold_info$Threshold[
    threshold_info$Model==m
  ]
  
  prob <- val_pred[[m]]
  
  est <- calc_metric(y,prob,thr)
  ci <- boot_ci(y,prob,thr,B=N_BOOT)
  
  r <- roc(
    y,prob,levels=c(0,1),
    direction="<",quiet=TRUE
  )
  
  aci <- ci.auc(r,method="delong")
  
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
    
    Balanced_Accuracy=est["BalancedAccuracy"],
    BA_L=ci[1,"BalancedAccuracy"],
    BA_U=ci[2,"BalancedAccuracy"],
    
    TN=est["TN"],
    FP=est["FP"],
    FN=est["FN"],
    TP=est["TP"]
  )
  
  confusion[[m]] <- data.frame(
    Model=m,
    Actual=c("Non-CHD","Non-CHD","CHD","CHD"),
    Predicted=c("Non-CHD","CHD","Non-CHD","CHD"),
    Count=c(est["TN"],est["FP"],est["FN"],est["TP"])
  )
}

performance_table <- bind_rows(results) %>%
  mutate(
    across(-Model,as.numeric),
    across(where(is.numeric),~round(.x,4))
  ) %>%
  arrange(desc(AUROC))

confusion_table <- bind_rows(confusion) %>%
  mutate(Count=as.numeric(Count))

print(performance_table)

write.csv(
  performance_table,
  out_file("DQM_AllModels_Performance.csv"),
  row.names=FALSE
)

write.csv(
  confusion_table,
  out_file("DQM_AllModels_Confusion_Matrix.csv"),
  row.names=FALSE
)


# ============================================================
# 30. Identify numerically best algorithm
# ============================================================

best_model <- performance_table$Model[1]

cat(
  "\nNumerically best DQM model =",
  best_model,"\n"
)


# ============================================================
# 31. Exploratory paired DeLong tests: selected best vs others
# ============================================================

roc_list2 <- list(
  RF=r_rf,SVM=r_svm,LR=r_lr,
  CB=r_cb,DNN=r_dnn,BP=r_bp
)

full_name <- c(
  RF="Random Forest",
  SVM="SVM",
  LR="Logistic Regression",
  CB="CatBoost",
  DNN="DNN",
  BP="BP"
)

other_models <- setdiff(models,best_model)

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
    
    ci1 <- ci.auc(r1,method="delong")
    ci2 <- ci.auc(r2,method="delong")
    
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
      
      Best_AUROC=as.numeric(auc(r1)),
      Best_Lower95CI=as.numeric(ci1[1]),
      Best_Upper95CI=as.numeric(ci1[3]),
      
      Comparator_AUROC=as.numeric(auc(r2)),
      Comparator_Lower95CI=as.numeric(ci2[1]),
      Comparator_Upper95CI=as.numeric(ci2[3]),
      
      Delta_AUROC=as.numeric(auc(r1)-auc(r2)),
      Delta_Lower95CI=diff_ci[1],
      Delta_Upper95CI=diff_ci[2],
      
      P_value=tt$p.value
    )
  })
)

# These tests are exploratory because the best algorithm was selected
# using the same temporal-validation dataset.
delong_best$P_Holm <- p.adjust(delong_best$P_value, method = "holm")
delong_best$Significant_Holm <- ifelse(delong_best$P_Holm < 0.05, "Yes", "No")
print(delong_best)

write.csv(
  delong_best,
  out_file("DQM_BestModel_vs_OtherModels_DeLong_Holm.csv"),
  row.names=FALSE
)


# ============================================================
# 32. Confusion matrix figure
# ============================================================

cm_dat <- confusion_table

cm_dat$Model <- factor(
  cm_dat$Model,
  levels=c("RF","SVM","LR","CB","DNN","BP"),
  labels=c(
    "Random Forest","SVM","Logistic Regression",
    "CatBoost","DNN","BP"
  )
)

cm_dat$Actual <- factor(
  cm_dat$Actual,
  levels=c("CHD","Non-CHD")
)

cm_dat$Predicted <- factor(
  cm_dat$Predicted,
  levels=c("Non-CHD","CHD")
)

p_cm <- ggplot(
  cm_dat,
  aes(Predicted,Actual,fill=Count)
) +
  geom_tile(colour="white",linewidth=1) +
  geom_text(aes(label=Count),size=4.5) +
  facet_wrap(~Model,ncol=3) +
  scale_fill_gradient(
    low="white",high="grey60",
    name="Count"
  ) +
  labs(
    x="Predicted class",
    y="Actual class"
  ) +
  theme_classic(base_size=12) +
  theme(
    axis.line=element_blank(),
    panel.border=element_rect(
      colour="black",fill=NA,linewidth=.7
    ),
    strip.background=element_blank()
  )

print(p_cm)

ggsave(
  out_file("DQM_AllModels_Confusion_Matrix.tiff"),
  p_cm,
  width=10,height=7,
  units="in",dpi=600,
  compression="lzw"
)


# ============================================================
# 33. Save first-stage results
# ============================================================

save(
  dqm_vars,
  
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
  
  file=out_file("DQM_MODEL_SELECTION_RESULTS.RData")
)


cat(
  "\n============================================\n",
  "Dietary and Questionnaire Model completed.\n",
  "Best algorithm =",best_model,"\n",
  "============================================\n"
)

# 34. Decision curve analysis: all six temporal-validation models
# Keep original figure style
# ============================================================

dca_dat <- data.frame(
  CHD=ifelse(val_list[[1]]$MCQ160C=="Yes",1,0),
  SVM=svm_final_prob,
  Logistic=lr_final_prob_raw,
  RF=rf_final_prob,
  BP=bp_final_prob,
  CatBoost=cb_final_prob,
  DNN=dnn_final_prob
)

dca_all <- dca(
  CHD~SVM+Logistic+RF+BP+CatBoost+DNN,
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
    axis.ticks=element_line(linewidth=0.7),
    legend.position=c(0.82,0.78),
    legend.background=element_blank()
  )

ggsave(
  out_file("DQM_AllModels_DCA.tiff"),
  p_dca,
  width=7,height=6,
  units="in",
  dpi=600,
  compression="lzw"
)

# ============================================================
# 35. Calibration in the temporal validation cohort (six algorithms)
# ============================================================
# Brier score, calibration intercept (ideal=0), slope (ideal=1).
# One decile plot per model; SMOTENC is not applied to validation rows.
calibration_prob <- list(
  RF = rf_final_prob,
  CatBoost = cb_final_prob,
  SVM = svm_final_prob,
  Logistic = lr_final_prob_raw,
  BP = bp_final_prob,
  DNN = dnn_final_prob
)
y_cal <- ifelse(val_list[[1]]$MCQ160C == "Yes", 1, 0)
calibration_dir <- out_file("DQM_Calibration_Plots")
dir.create(calibration_dir, showWarnings = FALSE, recursive = TRUE)

calibration_stats <- list()
calibration_curve <- list()

for (m in names(calibration_prob)) {
  prob <- as.numeric(calibration_prob[[m]])
  stopifnot(length(prob) == length(y_cal), !anyNA(prob))
  p_clip <- pmin(pmax(prob, 1e-6), 1 - 1e-6)
  logit_prob <- qlogis(p_clip)

  intercept_fit <- glm(y_cal ~ 1 + offset(logit_prob), family = binomial())
  slope_fit <- glm(y_cal ~ logit_prob, family = binomial())
  r <- pROC::roc(y_cal, prob, levels = c(0, 1), direction = "<", quiet = TRUE)
  r_ci <- pROC::ci.auc(r, method = "delong")
  calibration_stats[[m]] <- data.frame(
    Model = m,
    AUROC = as.numeric(pROC::auc(r)),
    AUROC_Lower = as.numeric(r_ci[1]),
    AUROC_Upper = as.numeric(r_ci[3]),
    Brier = mean((y_cal - prob)^2),
    Calibration_Intercept = unname(coef(intercept_fit)[1]),
    Calibration_Slope = unname(coef(slope_fit)["logit_prob"])
  )

  dat <- data.frame(outcome = y_cal, prob = prob)
  cal_group <- dat %>%
    mutate(group = dplyr::ntile(prob, 10)) %>%
    group_by(group) %>%
    summarise(Predicted = mean(prob), Observed = mean(outcome),
              N = n(), .groups = "drop") %>%
    mutate(Model = m, Type = "Observed calibration")
  calibration_curve[[m]] <- cal_group

  ideal <- data.frame(
    Predicted = c(0, 1), Observed = c(0, 1),
    Type = "Ideal calibration"
  )
  p <- ggplot() +
    geom_line(data = cal_group,
              aes(x = Predicted, y = Observed, linetype = Type),
              linewidth = 0.9) +
    geom_point(data = cal_group, aes(x = Predicted, y = Observed),
               size = 2.2) +
    geom_line(data = ideal,
              aes(x = Predicted, y = Observed, linetype = Type),
              linewidth = 0.7) +
    scale_linetype_manual(
      name = NULL,
      values = c("Observed calibration" = "solid",
                 "Ideal calibration" = "dashed"),
      breaks = c("Observed calibration", "Ideal calibration")
    ) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                       expand = c(0, 0)) +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, 0.2),
                       expand = c(0, 0)) +
    labs(x = "Predicted probability", y = "Observed probability") +
    theme_classic(base_size = 12) +
    theme(
      axis.line = element_blank(),
      panel.border = element_rect(colour = "black", fill = NA,
                                  linewidth = 0.7),
      axis.ticks = element_line(linewidth = 0.7),
      legend.position = c(0.76, 0.08),
      legend.background = element_blank(),
      legend.text = element_text(size = 12),
      legend.key.width = grid::unit(1.1, "cm")
    )

  ggsave(filename = file.path(calibration_dir,
                               paste0("DQM_", m, "_Calibration.tiff")),
         plot = p, width = 6, height = 6, units = "in", dpi = 600,
         compression = "lzw")
}
calibration_statistics <- dplyr::bind_rows(calibration_stats) %>%
  arrange(desc(AUROC))
calibration_curve_data <- dplyr::bind_rows(calibration_curve)
write.csv(calibration_statistics,
          out_file("DQM_AllModels_Calibration_Statistics.csv"),
          row.names = FALSE)
write.csv(calibration_curve_data,
          out_file("DQM_AllModels_Calibration_Curve_Data.csv"),
          row.names = FALSE)

# ============================================================
# 36. OPTIONAL RF permutation importance (temporal validation)
# ============================================================
# Importance is the reduction in validation AUROC after permuting
# a single predictor. One common shuffled subject index is used for
# all three imputed datasets per repetition. Default = 50 permutations.
# The finalized full-development RF models remain unchanged.
# This is an explanatory analysis, not part of model tuning.
if (RUN_RF_PERMUTATION) {
  permutation_dir <- out_file("DQM_RF_perm_checkpoint")
  dir.create(permutation_dir, recursive = TRUE, showWarnings = FALSE)

  predict_rf_3mi <- function(new_val_list) {
    prob_mi <- vector("list", N_MI)
    for (i in seq_len(N_MI)) {
      new_dat <- new_val_list[[i]][, c(dqm_vars, "MCQ160C"), drop = FALSE]
      baked <- as.data.frame(bake(rf_final_recipes[[i]], new_data = new_dat))
      x_names <- setdiff(names(baked), "MCQ160C")
      prob_mi[[i]] <- as.numeric(predict(
        rf_final_models[[i]], data = baked[, x_names, drop = FALSE]
      )$predictions[, "Yes"])
    }
    rowMeans(do.call(cbind, prob_mi))
  }

  # Confirm model reconstruction before permuting any variable.
  reconstructed_prob <- predict_rf_3mi(val_list)
  stopifnot(max(abs(reconstructed_prob - rf_final_prob)) < 1e-6)
  y_perm <- ifelse(val_list[[1]]$MCQ160C == "Yes", 1, 0)
  baseline_auc <- get_auc(val_list[[1]]$MCQ160C, rf_final_prob)
  cat("\nBaseline RF AUROC =", baseline_auc, "\n")

  for (j in seq_along(dqm_vars)) {
    v <- dqm_vars[[j]]
    filename <- file.path(permutation_dir, paste0(v, "_perm", B_PERM, ".rds"))
    # A checkpoint may be reused only with the SAME code, data, fitted
    # model and B_PERM. Otherwise clear the checkpoint directory.
    if (file.exists(filename)) {
      cat("Checkpoint exists:", v, "\n")
      next
    }
    cat("RF permutation:", j, "/", length(dqm_vars), v, "\n")
    set.seed(SEED + j)
    auc_drop <- numeric(B_PERM)
    for (b in seq_len(B_PERM)) {
      perm_index <- sample(seq_len(nrow(val_list[[1]])))
      new_val <- lapply(val_list, function(dat) {
        tmp <- dat
        tmp[[v]] <- dat[[v]][perm_index]
        tmp
      })
      perm_prob <- predict_rf_3mi(new_val)
      auc_drop[b] <- baseline_auc - get_auc(val_list[[1]]$MCQ160C, perm_prob)
    }
    saveRDS(data.frame(
      Variable = v,
      MeanDecreaseAUROC = mean(auc_drop),
      SD = sd(auc_drop),
      Lower95 = unname(quantile(auc_drop, 0.025)),
      Upper95 = unname(quantile(auc_drop, 0.975))
    ), filename)
  }

  checkpoints <- file.path(permutation_dir,
                           paste0(dqm_vars, "_perm", B_PERM, ".rds"))
  stopifnot(all(file.exists(checkpoints)))
  rf_perm_importance <- dplyr::bind_rows(lapply(checkpoints, readRDS)) %>%
    arrange(desc(MeanDecreaseAUROC)) %>%
    mutate(Rank = row_number()) %>%
    select(Rank, everything())
  write.csv(rf_perm_importance,
            out_file("DQM_RF_PermutationImportance_AllVariables_50.csv"),
            row.names = FALSE)

  # Human-readable labels follow the original DQM manuscript figure.
  dqm_labels <- c(

    "BPQ020" =
      "History of hypertension",

    "BPQ080" =
      "History of high cholesterol",

    "BPQ070" =
      "Time since last cholesterol check",

    "DIQ010" =
      "History of diabetes",

    "DIQ050" =
      "Current insulin use",

    "HIQ011" =
      "Health insurance coverage",

    "HSD010" =
      "Self-rated general health",

    "HSQ500" =
      "Recent head or chest cold",

    "HSQ590" =
      "History of HIV testing",

    "KIQ022" =
      "History of kidney disease",

    "KIQ480" =
      "Nocturia frequency",

    "MCQ010" =
      "History of asthma",

    "MCQ092" =
      "History of blood transfusion",

    "MCQ160G" =
      "History of emphysema",

    "MCQ160M" =
      "History of thyroid disease",

    "MCQ160K" =
      "History of chronic bronchitis",

    "MCQ220" =
      "History of cancer or malignancy",

    "MCQ300A" =
      "Family history of heart attack",

    "OCD150" =
      "Employment status",

    "DRQSPREP" =
      "Salt use in food preparation",

    "DRQSDIET" =
      "Special diet",

    "DSDANCNT" =
      "Number of antacid products taken",

    "SMQ020" =
      "Smoked at least 100 cigarettes in life",

    "SLQ050" =
      "Trouble sleeping",

    "DR1TLZ" =
      "Lutein and zeaxanthin intake",

    "DR1TVK" =
      "Vitamin K intake",

    "DR1TZINC" =
      "Zinc intake",

    "DR1TS080" =
      "Octanoic acid intake",

    "DR1TS120" =
      "Dodecanoic acid intake",

    "DR1BWATZ" =
      "Bottled water intake"
  )

  rf_top15 <- rf_perm_importance %>% slice_head(n = 15)
  rf_top15$Variable_name <- unname(dqm_labels[rf_top15$Variable])
  missing_label <- is.na(rf_top15$Variable_name)
  rf_top15$Variable_name[missing_label] <- rf_top15$Variable[missing_label]
  write.csv(rf_top15,
            out_file("DQM_RF_PermutationImportance_Top15_50.csv"),
            row.names = FALSE)

  plot_dat <- rf_top15
  plot_dat$Variable_name <- factor(plot_dat$Variable_name,
                                    levels = rev(plot_dat$Variable_name))
  p_rf_imp <- ggplot(plot_dat,
                     aes(x = MeanDecreaseAUROC, y = Variable_name)) +
    geom_col(width = 0.70, fill = "#4E79A7") +
    labs(x = "Mean decrease in AUROC", y = NULL) +
    scale_x_continuous(expand = expansion(mult = c(0, 0.05))) +
    theme_classic(base_size = 12) +
    theme(
      axis.line = element_blank(),
      panel.border = element_rect(colour = "black", fill = NA, linewidth = 0.7),
      axis.ticks = element_line(linewidth = 0.7),
      axis.text.y = element_text(size = 10)
    )
  ggsave(out_file("DQM_RF_PermutationImportance_Top15_50.tiff"),
         plot = p_rf_imp, width = 8, height = 6.5,
         units = "in", dpi = 600, compression = "lzw")
  saveRDS(rf_perm_importance,
          out_file("DQM_RF_10fold_permutation_importance.rds"))
}

# ============================================================
# Session details for reproducibility
# ============================================================
writeLines(capture.output(sessionInfo()), out_file("sessionInfo.txt"))
cat("\nFinished. Outputs: ", normalizePath(OUTPUT_DIR), "\n", sep = "")
