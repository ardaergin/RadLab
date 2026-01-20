#!/usr/bin/env Rscript

# 0. Disable renv sandbox
Sys.setenv(RENV_CONFIG_SANDBOX_ENABLED = "FALSE")

# Strongly recommended on HPC to avoid oversubscription (32 workers × BLAS threads)
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)

source("analysis/clustering_setup.R")

library(parallel)
message("\nRunning Class Enumeration (Quadratic Mixture)...")

# --- Step 0: Detect and Setup Cluster ---
slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK")
n_cores <- if (slurm_cpus != "") as.numeric(slurm_cpus) else 4

message(paste0("--> Initializing Cluster with ", n_cores, " cores..."))
cl <- makeCluster(n_cores, type = "PSOCK", outfile = "")
on.exit(stopCluster(cl), add = TRUE)

message("--> Exporting environment to workers...")
clusterEvalQ(cl, {
  library(devtools)
  library(here)
  devtools::load_all(here())
  library(lcmm)
  NULL
})

clusterExport(cl, c("model_data", "formula_A_all_controls_quad"), envir = environment())

# --- Step 1: Load Baseline ---
path_m1 <- here::here("outputs", "models", "m_a5_all_controls_quad.rds")
if (file.exists(path_m1)) {
  m_init <- readRDS(path_m1)
} else {
  stop("Baseline model not found!")
}

# Optional: export baseline once (can reduce repeated serialization inside gridsearch)
clusterExport(cl, "m_init", envir = environment())

# --- Step 2: The Loop ---
for (k in 2:4) {
  message(paste0("\n===== Fitting Model with ", k, " Classes ====="))

  # Build the *actual* multlcmm call that gridsearch expects
  m_call <- substitute(
    lcmm::multlcmm(
      data    = model_data,
      subject = "ID",
      link    = "5-equi-splines",
      fixed   = formula_A_all_controls_quad,
      mixture = ~ time + I(time^2),
      random  = ~ time,
      ng      = K
    ),
    list(K = k)
  )

  m_k <- lcmm::gridsearch(
    m       = m_call,
    rep     = 32,
    maxiter = 500,
    minit   = m_init,   # or just minit = m_init (even if exported)
    cl      = cl
  )

  saveRDS(m_k, here::here("outputs", "models", paste0("m_quad_", k, "class.rds")))
  message(paste0("\n===== Summary for ", k, "-Class Model ====="))
  print(summary(m_k))
}

message("\nAnalysis Complete.")
