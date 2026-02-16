#!/usr/bin/env Rscript

# 1. Setup & Config -------------------------------------------------------
source("analysis/clustering_setup.R")

# 2. Run Model ------------------------------------------------------------
# We can use tryCatch so the script doesn't crash hard if convergence fails
m <- tryCatch({
  lcmm::multlcmm(
    data = df,
    subject = "ID",
    link = rep(opt$link, n_outcomes),
    fixed = final_formula,
    random = random_formula,
    ng = 1,
    maxiter = opt$maxiter
  )
}, error = function(e) {
  message("Error in model fitting: ", e$message)
  return(NULL)
})

# 3. Save Output ----------------------------------------------------------
if (!is.null(m)) {
  quad_suffix <- if (opt$quadratic) "_quad" else ""

  filename <- paste0(
    "model_",
    opt$transform, "_",
    opt$mode, "_",
    rand_label, "_",
    opt$controls,
    quad_suffix,
    ".rds"
  )

  output_path <- here("outputs", "models", filename)

  message("Saving model to: ", output_path)
  saveRDS(m, output_path)

  message("\n===== Summary =====\n")
  print(summary(m))
  message("\n===================\n")

} else {
  message("Skipping save due to errors.")
}
