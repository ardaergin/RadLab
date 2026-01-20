#!/usr/bin/env Rscript

# 0. Disable renv sandbox to speed up parallel workers
# This must be done BEFORE loading packages
Sys.setenv(RENV_CONFIG_SANDBOX_ENABLED = "FALSE")

# Loading the shared setup (Libraries + Data + Preprocessing + Formulas)
source("analysis/clustering_setup.R")
library(parallel)

# Class Enumeration (Quadratic Mixture) --------------------------------
message("\nRunning Class Enumeration (Quadratic Mixture)...")


# --- Step 0: Detect and Report Cores ---

# 1. Check what the hardware actually has
n_physical <- parallel::detectCores(logical = FALSE)
message(paste0("Physical cores detected on node: ", n_physical))

# 2. Check what SLURM allocated to this job
slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK")

if (slurm_cpus != "") {
  n_cores <- as.numeric(slurm_cpus)
  message(paste0("SLURM allocation detected:     ", n_cores))
  message(paste0("--> Using ", n_cores, " cores for parallel gridsearch."))
} else {
  # Strict mode: Stop if not running under SLURM
  stop("Error: SLURM_CPUS_PER_TASK is empty. This script must be run via sbatch.")
}


# --- Step 1: Load Baseline Model (Seed) ---
# We load the 1-class quadratic model to use as the initializer (minit)
# This significantly improves stability and speed for multi-class models.
path_m1 <- here("outputs", "models", "m_a5_all_controls_quad.rds")

if (file.exists(path_m1)) {
  message("Loading 1-class baseline model for initialization...")
  m_init <- readRDS(path_m1)
} else {
  stop("Baseline model (m_a5_all_controls_quad.rds) not found! Run the base script first.")
}


# --- Step 2: The Loop (k = 2, 3, 4) ---
message("\nStarting Loop (2 to 4 classes)...")

for (k in 2:4) {

  message(paste0("\n===== Fitting Model with ", k, " Classes ====="))

  # CRITICAL FIX FOR PARALLEL: Inject the VALUE of k into the call.
  # We can use bquote() to turn 'ng = k' into 'ng = 2' (or 3, 4) explicitly.
  # This prevents the "object 'k' not found" error on the parallel workers.
  run_model_structure <- bquote(
    multlcmm(
      data = model_data, subject = "ID", link = "5-equi-splines",
      # Fixed structure: Full controls + Quadratic Time
      fixed = formula_A_all_controls_quad,
      # Mixture: Allow Intercept, Slope, AND Curve to differ by class
      mixture = ~ time + I(time^2),
      random = ~ time,
      ng = .(k) # <--- The .(k) injects the actual number (e.g., 2)
    )
  )

  # Gridsearch: Tries 32 random starting points
  m_k <- lcmm::gridsearch(
    rep = 32,
    maxiter = 500,
    minit = m_init,
    cl = n_cores,
    # We evaluate the structure locally first so gridsearch receives a
    # clear, hardcoded model object to distribute to workers.
    eval(run_model_structure)
  )

  # Save the result
  save_name <- paste0("m_quad_", k, "class.rds")
  saveRDS(m_k, here("outputs", "models", save_name))

  # Print the summary to the log immediately
  message(paste0("\n===== Summary for ", k, "-Class Model ====="))
  print(summary(m_k))
  message(paste0("===== End of Summary for ", k, "-Class Model =====\n"))
}

message("\nAnalysis Complete.")
