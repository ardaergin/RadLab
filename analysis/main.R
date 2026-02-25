#!/usr/bin/env Rscript
# analysis/main.R
library(here)
library(mlflow)

# 0. Thread Safety for HPC
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

# 1. Sources & Arguments
source("analysis/config.R")
source("analysis/formula_builder.R")
source("analysis/mlflow_helpers.R")
source("analysis/lcmm_models.R")

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
if (opt$nclass > 1) message("  Mixture: ", format(formulas$mixture))
message("CONFIGURATION:")
message("  Number of classes: ", opt$nclass)
message("  Link: ", opt$link)
message("  Maxiter: ", opt$maxiter)
message("------------------------------------------------")

# 4. Configure MLflow
exp_name <- get_experiment_name(opt)
setup_mlflow_connection(exp_name)
mlflow::mlflow_start_run()
run_name <- get_run_name()
mlflow::mlflow_set_tag("mlflow.runName", run_name)
mlflow_set_lcmm_tags(opt)

tryCatch({
  # 5. Baseline Logic (Only if K > 1)
  if (opt$nclass > 1) {
    baseline_key <- get_baseline_key(opt)
    baseline_run_id <- mlflow_find_baseline_run_id(exp_name, baseline_key)
    if (is.na(baseline_run_id)) {
      stop("Baseline model not found for baseline_key: ", baseline_key)
    }
    message("Found baseline with key: ", baseline_key)
    message("Baseline run ID: ", baseline_run_id)
    m_baseline <- readRDS(
      mlflow::mlflow_download_artifacts(
        path = "model/model.rds",
        run_id = baseline_run_id
      )
    )
  }

  # 6. Run Model
  if (opt$nclass == 1) {
    m <- fit_lcmm_baseline(df, formulas, opt)
  } else {
    m <- fit_lcmm_gridsearch(df, formulas, m_baseline, opt)
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