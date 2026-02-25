# analysis/mlflow_helpers.R
library(here)
library(mlflow)

########## Label Utilities ##########
random_label <- function(opt) {
  val <- as.character(opt$random)
  if (identical(val, "1")) return("RI") # Random Intercept
  if (identical(val, "2")) return("RS") # Random Slope
  if (identical(val, "3")) return("RQ") # Random Quadratic
  stop("Invalid --random: ", opt$random)
}
mixture_label <- function(opt) {
  val <- as.character(opt$mixture)
  if (identical(val, "1")) return("MI") # Mixture Intercept
  if (identical(val, "2")) return("MS") # Mixture Slope
  if (identical(val, "3")) return("MQ") # Mixture Quadratic
  if (identical(val, "NA")) return("NA") # NA, for nclass=1
  stop("Invalid --mixture: ", opt$mixture)
}
########## End of Label Utilities ##########


########## Run and Experiment Name ##########
get_run_name <- function() {
  job_id <- Sys.getenv("SLURM_JOB_ID", unset = "")
  if (nzchar(job_id)) return(job_id)
  paste0("local_", format(Sys.time(), "%Y%m%d_%H%M%S"))
}
get_experiment_name <- function(opt) {
  if (identical(opt$transform, "hellinger")) return("lcmm_hellinger")
  if (identical(opt$transform, "ilr")) return(paste0("lcmm_ilr_", opt$basis))
  paste0("lcmm_", opt$transform)
}
########## End of Run and Experiment Name ##########


########## Baseline Key & Predecessor Lookup ##########
# `lcmm::multlcmm` fits K-class models by reusing the 1–class solution (ng=1)
# as an initial value. To support this, we construct a stable "chain key"
# that uniquely identifies a model configuration (outcomes, controls,
# time structure, random/mixed effects, link, etc.) so we can reliably
# find the corresponding K=1 run for any given K.
get_baseline_key <- function(opt) {
  paste(
    "outcomes=", opt$outcomes,
    ";controls=", opt$controls,
    ";quadratic=", isTRUE(opt$quadratic),
    ";random=", random_label(opt),
    ";link=", opt$link,
    sep = ""
  )
}
mlflow_find_baseline_run_id <- function(exp_name, baseline_key) {
  exp_id <- mlflow::mlflow_id(mlflow::mlflow_get_experiment(name = exp_name))
  filter <- paste0(
    'tags.baseline_key = "', baseline_key, '" AND ',
    'attributes.status = "FINISHED"'
  )
  runs <- mlflow::mlflow_search_runs(
    experiment_ids = exp_id,
    filter = filter,
    order_by = list("attributes.start_time DESC")
  )
  if (is.null(runs) || nrow(runs) == 0) return(NA_character_)
  runs$run_id[[1]]
}
########## End of Baseline Key & Predecessor Lookup ##########


########## MLflow Connection Setup ##########
setup_mlflow_connection <- function(exp_name) {
  # 1. Generate the R equivalent of $(pwd)/outputs/mlruns
  default_dir <- here::here("outputs", "mlruns")
  if (!dir.exists(default_dir)) dir.create(
    default_dir, recursive = TRUE, showWarnings = FALSE
  )
  default_uri <- paste0("file://", normalizePath(default_dir, mustWork = FALSE))

  # 2. Check the environment, otherwise default to the file path
  uri <- Sys.getenv(
    "MLFLOW_TRACKING_URI",
    unset = default_uri
  )

  options(mlflow.connect.wait = 60, mlflow.connect.sleep = 1)
  mlflow::mlflow_set_tracking_uri(uri)
  mlflow::mlflow_set_experiment(exp_name)
}

########## MLflow Tags ##########
mlflow_set_lcmm_tags <- function(opt) {
  if (opt$nclass == 1) {
    mlflow::mlflow_set_tag("baseline_key", get_baseline_key(opt))
  } else {
    mlflow::mlflow_set_tag("baseline_key", "NA")
  }
  mlflow::mlflow_set_tag("k", as.character(opt$nclass))
  mlflow::mlflow_set_tag("outcomes", opt$outcomes)
  mlflow::mlflow_set_tag("controls", opt$controls)
  mlflow::mlflow_set_tag("quadratic", as.character(isTRUE(opt$quadratic)))
  mlflow::mlflow_set_tag("random", random_label(opt))
  mlflow::mlflow_set_tag("link", opt$link)
  # Special cases
  mlflow::mlflow_set_tag("basis", opt$basis)
  mlflow::mlflow_set_tag("mixture", mixture_label(opt))

  # SLURM job ID
  slurm_job_id <- Sys.getenv("SLURM_JOB_ID", unset = "")
  if (nzchar(slurm_job_id)) {
    mlflow::mlflow_set_tag("slurm.job_id", slurm_job_id)
  } else {
    mlflow::mlflow_set_tag("slurm.job_id", "NA")
  }

  # Log params
  for (nm in names(opt)) {
    if (!is.null(opt[[nm]])) mlflow::mlflow_log_param(nm, as.character(opt[[nm]]))
  }
}
########## End of MLflow Tags ##########


########## MLflow Model Saving & Logging ##########
mlflow_log_lcmm_metrics <- function(m) {
  # Log standard metrics
  if (!is.null(m$loglik)) mlflow::mlflow_log_metric("loglik", as.numeric(m$loglik))
  if (!is.null(m$AIC)) mlflow::mlflow_log_metric("AIC", as.numeric(m$AIC))
  if (!is.null(m$BIC)) mlflow::mlflow_log_metric("BIC", as.numeric(m$BIC))
  if (!is.null(m$niter)) mlflow::mlflow_log_metric("niter", as.numeric(m$niter))

  # Log convergence as a human-readable tag and a param
  if (!is.null(m$conv)) {
    mlflow::mlflow_log_param("conv_code", as.character(m$conv))
    
    # lcmm convergence codes translation
    conv_status <- switch(
      as.character(m$conv),
      "1" = "Converged",
      "2" = "Max iterations reached",
      "3" = "Converged with negative variance",
      "4" = "Singular Hessian / Boundary",
      "5" = "Algorithm stopped",
      "Unknown"
    )
    mlflow::mlflow_set_tag("convergence_status", conv_status)
  }
}

mlflow_save_and_log_model <- function(model_obj) {
  # 1. Print summary to the R console (so it appears in SLURM .txt logs)
  message("\n=================== MODEL SUMMARY ===================")
  print(summary(model_obj))
  message("=====================================================\n")

  # 2. Setup a temporary directory for file generation
  temp_dir <- tempfile(pattern = "lcmm_artifacts_")
  dir.create(temp_dir)
  on.exit(unlink(temp_dir, recursive = TRUE), add = TRUE) # Auto-cleans up when done

  paths <- list(
    rds = file.path(temp_dir, "model.rds"),
    txt = file.path(temp_dir, "summary.txt"),
    sess = file.path(temp_dir, "sessionInfo.txt")
  )

  # 3. Save files to the temporary disk
  saveRDS(model_obj, paths$rds)
  writeLines(capture.output(summary(model_obj)), paths$txt)
  writeLines(capture.output(sessionInfo()), paths$sess)

  # 4. Upload files to the MLflow Artifacts tab
  for (p in paths) {
    mlflow::mlflow_log_artifact(p, artifact_path = "model")
  }
  
  # 5. Return the logical MLflow URI path instead of a hardcoded local path
  run <- mlflow::mlflow_get_run()
  run_id <- if ("run_id" %in% names(run)) run$run_id else run$run_uuid
  if (is.list(run_id)) run_id <- run_id[[1]]
  
  return(paste0("runs:/", run_id, "/model/model.rds"))
}
########## End of MLflow Model Saving & Logging ##########
