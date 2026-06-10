#!/usr/bin/env Rscript
# analysis/main.R
library(here)
library(mlflow)
library(optparse)

# 1. Sources & Arguments
source(here::here("analysis", "config.R"))
source(here::here("analysis", "formula_builder.R"))
source(here::here("analysis", "mlflow_helpers.R"))
source(here::here("analysis", "lcmm_models.R"))

# 2. Parse Arguments
opt <- parse_args(OptionParser(option_list = get_option_list()))
# Correct arguments:
if (opt$transform != "ilr") {opt$basis <- "NA"}
if (opt$nclass == 1) {opt$mixture <- "NA"}
if (opt$nclass == 1) {opt$rep <- "NA"}


# 3. Load data
file_path <- here::here("data", opt$file)
if (!file.exists(file_path)) stop("Data file not found at: ", file_path)
message("Reading data from: ", file_path)
df <- readRDS(file_path)

# 4. Build Formulas
formulas <- build_lcmm_formulas(opt)
message("------------------------------------------------")
message("=== DATA ===")
message("N experiments: ", length(unique(df$experiment)))
message("N participants: ", length(unique(df$ID)))
message("\n")
message("=== MODEL ===")
message("FORMULA:")
message("  Fixed Effects:  ", format(formulas$fixed))
message("  Random Effects: ", format(formulas$random))
if (opt$nclass > 1) {
  message("  Mixture: ", format(formulas$mixture))
}
message("CONFIGURATION:")
message("  Number of classes: ", opt$nclass)
message("  Link: ", opt$link)
message("  maxiter (lcmm::multlcmm)): ", opt$lcmm_maxiter)
if (opt$nclass > 1) {
  message("  maxiter (lcmm::gridsearch): ", format(opt$grid_maxiter))
}
message("------------------------------------------------")

# 4. Configure MLflow
exp_name <- get_experiment_name(opt)
setup_mlflow_connection(exp_name)
mlflow::mlflow_start_run()
run_name <- get_run_name()
mlflow::mlflow_set_tag("mlflow.runName", run_name)
mlflow_set_lcmm_tags(opt)

tryCatch({
  # 5 & 6. Fetching Predecessors & Running Models
  if (opt$phase == "gridsearch") {
    if (opt$nclass == 1) {
      stop("Gridsearch phase is only applicable for K > 1.")
    }

    # Fetch K=1 Baseline to initialize gridsearch
    baseline_key <- get_baseline_key(opt)
    baseline_run_id <- mlflow_find_baseline_run_id(exp_name, baseline_key)
    if (is.na(baseline_run_id)) {
      stop("Baseline model not found for baseline_key: ", baseline_key)
    }
    
    message("Found K=1 baseline with key: ", baseline_key)
    message("Baseline run ID: ", baseline_run_id)
    m_baseline <- readRDS(
      mlflow::mlflow_download_artifacts(
        path = "model/model.rds",
        run_id = baseline_run_id
      )
    )
    
    m <- fit_lcmm_gridsearch(df, formulas, m_baseline, opt)

  } else if (opt$phase == "estimation") {
    if (opt$nclass == 1) {
      # Standard K=1 Baseline Estimation (no initialization needed)
      m <- fit_lcmm_estimation(df, formulas, opt)
    } else {
      grid_key <- get_gridsearch_key(opt)
      grid_run_id <- mlflow_find_grid_run_id(exp_name, grid_key)
      
      if (is.na(grid_run_id)) {
        stop("Completed gridsearch not found for gridsearch_key: ", grid_key)
      }
      
      message("Found gridsearch model (run ID: ", grid_run_id, ")")
      m_grid <- readRDS(
        mlflow::mlflow_download_artifacts("model/model.rds", run_id = grid_run_id)
      )
      
      m <- fit_lcmm_estimation(df, formulas, opt, m_init = m_grid)
    }
  } else {
    stop("Invalid phase provided: ", opt$phase)
  }
  
  # 7. Saving & Logging
  mlflow_log_lcmm_metrics(m)
  saved_path <- mlflow_save_and_log_model(m)
  mlflow::mlflow_set_tag("model_path", saved_path)

  mlflow::mlflow_end_run(status = "FINISHED")
  message("Successfully completed run: ", run_name)

}, error = function(e) {
  message("\nERROR ENCOUNTERED: ", e$message)

  # Safely attempt to log the error and close the active run as FAILED
  try({
    mlflow::mlflow_set_tag("error_message", e$message)
    mlflow::mlflow_end_run(status = "FAILED")
  }, silent = TRUE)

  stop(e)
})