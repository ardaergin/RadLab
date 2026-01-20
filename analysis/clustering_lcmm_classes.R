#!/usr/bin/env Rscript

Sys.setenv(RENV_CONFIG_SANDBOX_ENABLED = "FALSE")

# Avoid oversubscription
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

# --- Step 0: cores + cluster ---
slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK")
n_cores <- if (slurm_cpus != "") as.numeric(slurm_cpus) else 4

message(paste0("--> Initializing Cluster with ", n_cores, " cores..."))
cl <- makeCluster(n_cores, type = "PSOCK", outfile = "")
on.exit(stopCluster(cl), add = TRUE)

# --- Step 1: Load Baseline (BEFORE exporting!) ---
path_m1 <- here::here("outputs", "models", "m_a5_all_controls_quad.rds")
if (!file.exists(path_m1)) stop("Baseline model not found at: ", path_m1)
m_init <- readRDS(path_m1)

# --- Step 2: Prep workers ---
message("--> Exporting environment to workers...")
clusterEvalQ(cl, { library(lcmm); NULL })

# Export objects workers need (now m_init exists)
clusterExport(
  cl,
  varlist = c("model_data", "formula_A_all_controls_quad", "m_init"),
  envir = environment()
)

# (Optional debug)
print(clusterCall(cl, function() paste("worker", Sys.getpid(), "libpaths:", paste(.libPaths(), collapse=" | "))))

# --- Step 3: Fit models ---
for (k in 2:4) {
  message(paste0("\n===== Fitting Model with ", k, " Classes ====="))

  gs_call <- substitute(
    lcmm::gridsearch(
      m = multlcmm(
        data    = model_data,
        subject = "ID",
        link    = "5-equi-splines",
        fixed   = formula_A_all_controls_quad,
        mixture = ~ time + I(time^2),
        random  = ~ time,
        ng      = K
      ),
      rep     = 32,
      maxiter = 500,
      minit   = m_init,
      cl      = cl
    ),
    list(K = k)
  )

  m_k <- eval(gs_call)

  saveRDS(m_k, here::here("outputs", "models", paste0("m_quad_", k, "class.rds")))
  print(summary(m_k))
}

message("\nAnalysis Complete.")
