#!/usr/bin/env Rscript

# 0. Parallel Environment Setup -------------------------------------------
Sys.setenv(RENV_CONFIG_SANDBOX_ENABLED = "FALSE")

# Avoid thread oversubscription
Sys.setenv(
  OMP_NUM_THREADS = "1",
  OPENBLAS_NUM_THREADS = "1",
  MKL_NUM_THREADS = "1",
  VECLIB_MAXIMUM_THREADS = "1",
  NUMEXPR_NUM_THREADS = "1"
)


# 1. Setup & Config -------------------------------------------------------
source("analysis/clustering_setup.R")
library(parallel)


# 2. Derive Filename Parameters (Global) ----------------------------------
# We define these early so we can use them for both loading (step 3) and saving (step 6)
quad_suffix <- if (opt$quadratic) "_quad" else ""
rand_label  <- if (opt$random == "1") "RI" else "RS"


# 3. Load Baseline (MANDATORY for gridsearch) -----------------------------
# lcmm::gridsearch REQUIRES a 1-class model to generate initial values.
baseline_filename <- paste0("model_", opt$transform, "_", opt$mode, "_", rand_label, "_", opt$controls, quad_suffix, ".rds")
path_m1 <- here::here("outputs", "models", baseline_filename)

if (!file.exists(path_m1)) {
  stop("\nCRITICAL ERROR: Baseline model not found at: ", path_m1, "\n",
       "You MUST run the 1-class model (run_model.R) with these exact settings first.\n",
       "Gridsearch cannot run without a baseline.")
}

message("--> Loading baseline model: ", baseline_filename)
m_init <- readRDS(path_m1)


# 4. Initialize Cluster ---------------------------------------------------
slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK")
n_cores <- if (slurm_cpus != "") as.numeric(slurm_cpus) else 4

message(paste0("--> Initializing Cluster with ", n_cores, " cores..."))
cl <- makeCluster(n_cores, type = "PSOCK", outfile = "")
on.exit(stopCluster(cl), add = TRUE)

# Export environment
clusterEvalQ(cl, { library(lcmm); NULL })
clusterExport(
  cl,
  # We export m_init (which might be NULL, that's fine)
  varlist = c("df", "final_formula", "random_formula", "m_init", "n_outcomes", "opt"),
  envir = environment()
)

# 5. Define Mixture -------------------------------------------------------
mixture_formula <- if (opt$mixture=="time") ~ time else ~ time + I(time^2)
mix_tag <- if (opt$mixture == "time_quad") "_mixQ" else "" # for the filename

# 6. Run Gridsearch -------------------------------------------------------
message("\n===== Fitting Model with ", opt$nclass, " Classes =====")
message("  Baseline Init: ", opt$use_baseline)
message("  Reps:          ", opt$rep)

# We use substitute to strictly inject the variables into the call
m_call <- substitute(
  lcmm::gridsearch(
    m = multlcmm(
      fixed   = final_formula,
      mixture = mixture_formula,
      random  = random_formula,
      subject = "ID",
      data    = df,
      link    = links,
      ng      = K
    ),
    rep     = reps,
    maxiter = max_it,
    minit   = m_init, # MANDATORY
    cl      = cl
  ),
  list(K = opt$nclass, links = rep(opt$link, n_outcomes), reps = opt$rep, max_it = opt$maxiter)
)

m_k <- eval(m_call)

# 7. Save Output ----------------------------------------------------------
filename <- paste0(
  "model_",
  opt$transform, "_",
  opt$mode, "_",
  rand_label, "_",
  opt$controls,
  quad_suffix,
  mix_tag,
  "_", opt$nclass, "class",
  ".rds"
)

output_path <- here::here("outputs", "models", filename)
message("Saving model to: ", output_path)
saveRDS(m_k, output_path)

message("\n===== Summary =====\n")
print(summary(m_k))
message("\nAnalysis Complete.")
