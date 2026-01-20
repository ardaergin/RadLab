#!/usr/bin/env Rscript

# 0. Disable renv sandbox
Sys.setenv(RENV_CONFIG_SANDBOX_ENABLED = "FALSE")

# Loading the shared setup
source("analysis/clustering_setup.R")
library(parallel)

message("\nRunning Class Enumeration (Quadratic Mixture)...")

# --- Step 0: Detect and Setup Cluster ---
slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK")
n_cores <- if (slurm_cpus != "") as.numeric(slurm_cpus) else 4

message(paste0("--> Initializing Cluster with ", n_cores, " cores..."))
cl <- makeCluster(n_cores)

# CRITICAL: Export everything the workers need to hear
message("--> Exporting environment to workers...")
clusterEvalQ(cl, {
  # Each worker needs the libraries and the local package
  library(devtools)
  library(here)
  devtools::load_all(here())
  library(lcmm)
})

# Export the actual data and formulas from your current global environment
clusterExport(cl, c("model_data", "formula_A_all_controls_quad"))

# --- Step 1: Load Baseline ---
path_m1 <- here("outputs", "models", "m_a5_all_controls_quad.rds")
if (file.exists(path_m1)) {
  m_init <- readRDS(path_m1)
} else {
  stop("Baseline model not found!")
}

# --- Step 2: The Loop ---
for (k in 2:4) {
  message(paste0("\n===== Fitting Model with ", k, " Classes ====="))

  # Inject value of k
  run_model_structure <- bquote(
    multlcmm(
      data = model_data, subject = "ID", link = "5-equi-splines",
      fixed = formula_A_all_controls_quad,
      mixture = ~ time + I(time^2),
      random = ~ time,
      ng = .(k)
    )
  )

  m_k <- lcmm::gridsearch(
    rep = 32,
    maxiter = 500,
    minit = m_init,
    cl = cl, # <--- Pass the cluster object we created and "fed"
    eval(run_model_structure)
  )

  saveRDS(m_k, here("outputs", "models", paste0("m_quad_", k, "class.rds")))

  message(paste0("\n===== Summary for ", k, "-Class Model ====="))
  print(summary(m_k))
}

stopCluster(cl)
message("\nAnalysis Complete.")
