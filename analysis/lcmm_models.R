# analysis/lcmm_models.R
library(lcmm)
library(parallel)

detect_cores <- function() {
  slurm_cpus <- Sys.getenv("SLURM_CPUS_PER_TASK", unset = "")
  if (nzchar(slurm_cpus)) as.numeric(slurm_cpus) else 1
}

fit_lcmm_baseline <- function(df, formulas, opt) {
  message("\n===== Fitting Baseline (K=1) =====")
  n_cores <- detect_cores()
  message("--> Utilizing nproc = ", n_cores, " for baseline optimization.")
  lcmm::multlcmm(
    data = df,
    subject = "ID",
    link = rep(opt$link, formulas$n_outcomes),
    fixed = formulas$fixed,
    random = formulas$random,
    ng = 1,
    maxiter = opt$maxiter,
    nproc = n_cores
  )
}

fit_lcmm_gridsearch <- function(df, formulas, m_init, opt) {
  message("\n===== Fitting Gridsearch (K=", opt$nclass, ") =====")
  n_cores <- detect_cores()
  message("--> Initializing Cluster with ", n_cores, " cores...")
  cl <- makeCluster(n_cores, type = "PSOCK", outfile = "")
  on.exit(stopCluster(cl), add = TRUE)
  clusterEvalQ(cl,{library(lcmm);NULL})

  # Export everything the parallel workers need
  clusterExport(
    cl = cl,
    varlist = c("df", "formulas", "m_init", "opt"),
    envir = environment()
  )

  # Strict substitution for parallel evaluation
  # Note: MUST USE multlcmm(), and not lcmm::multlcmm()
  m_call <- substitute(
    lcmm::gridsearch(
      m = multlcmm(
        fixed   = f_fixed,
        mixture = f_mix,
        random  = f_rand,
        subject = "ID",
        data    = df,
        link    = links,
        ng      = K
      ),
      rep     = reps,
      maxiter = max_it,
      minit   = m_init,
      cl      = cl
    ),
    list(
      K = opt$nclass, 
      links = rep(opt$link, formulas$n_outcomes), 
      reps = opt$rep, 
      max_it = opt$maxiter,
      f_fixed = formulas$fixed,
      f_mix = formulas$mixture,
      f_rand = formulas$random
    )
  )

  eval(m_call)
}
