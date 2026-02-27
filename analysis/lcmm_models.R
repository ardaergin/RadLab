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
    maxiter = opt$lcmm_maxiter,
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
        data    = df,
        subject = "ID",
        link    = links,
        fixed   = f_fixed,
        random  = f_rand,
        ng      = k,
        maxiter = lcmm_max_it,
        nproc   = 1,
        mixture = f_mix
      ),
      rep     = reps,
      maxiter = grid_max_it,
      minit   = m_init,
      cl      = cl
    ),
    list(
      links = rep(opt$link, formulas$n_outcomes),
      f_fixed = formulas$fixed,
      f_rand = formulas$random,
      k = opt$nclass,
      lcmm_max_it = opt$lcmm_maxiter,
      f_mix = formulas$mixture,
      grid_max_it = opt$grid_maxiter,
      reps = opt$rep
    )
  )

  eval(m_call)
}
