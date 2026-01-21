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


# 3. Handle Baseline Initialization ---------------------------------------
m_init <- NULL # Default to NULL (Fresh gridsearch)

if (opt$use_baseline) {

  baseline_filename <- paste0(
    "model_",
    opt$transform, "_",
    opt$mode, "_",
    rand_label, "_",
    opt$controls,
    quad_suffix,
    ".rds"
  )

  path_m1 <- here::here("outputs", "models", baseline_filename)

  if (file.exists(path_m1)) {
    message("--> Loading baseline model for initialization: ", baseline_filename)
    m_init <- readRDS(path_m1)
  } else {
    stop("\nCRITICAL ERROR: Baseline model not found at: ", path_m1, "\n",
         "To run without baseline, use: --use_baseline=FALSE")
  }

} else {
  message("--> Skipping baseline initialization (Starting fresh gridsearch).")
}


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

m_call <- substitute(
  multlcmm(
    fixed   = final_formula,
    mixture = mixture_formula,
    random  = random_formula,
    subject = "ID",
    data    = df,
    link    = links,
    ng      = K
  ),
  list(K = opt$nclass, links = rep(opt$link, n_outcomes))
)

# FIX: Build the argument list dynamically
args_list <- list(
  m       = m_call,
  rep     = opt$rep,
  maxiter = opt$maxiter,
  cl      = cl
)

# Only add minit if it is NOT NULL
if (!is.null(m_init)) {
  args_list$minit <- m_init
}

# Run using do.call
m_k <- do.call(lcmm::gridsearch, args_list)

# 7. Save Output ----------------------------------------------------------
filename <- paste0(
  "model_",
  opt$transform, "_",
  opt$mode, "_",
  rand_label, "_",
  opt$controls,
  quad_suffix,
  mix_tag,       # <--- Add this to distinguish mixture types
  "_", opt$nclass, "class",
  ".rds"
)

output_path <- here::here("outputs", "models", filename)
message("Saving model to: ", output_path)
saveRDS(m_k, output_path)

message("\n===== Summary =====\n")
print(summary(m_k))
message("\nAnalysis Complete.")
