#!/usr/bin/env Rscript

# Loading the shared setup (Libraries + Data + Preprocessing + Formulas)
source("analysis/setup.R")


# Class Enumeration (Quadratic Mixture) --------------------------------
message("\nRunning Class Enumeration (Quadratic Mixture)...")

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
message("\nStarting Tournament Loop (2 to 4 classes)...")

for (k in 2:4) {

  message(paste0("\n===== Fitting Model with ", k, " Classes ====="))

  # Gridsearch: Tries 30 random starting points to avoid local maxima
  # minit = m_init uses the parameters from the 1-class model as a seed
  m_k <- lcmm::gridsearch(
    rep = 30, maxiter = 200, minit = m_init,
    multlcmm(
      data = model_data, subject = "ID", link = "5-equi-splines",
      # Fixed structure: Full controls + Quadratic Time
      fixed = formula_A_all_controls_quad,
      # Mixture: Allow Intercept, Slope, AND Curve to differ by class
      mixture = ~ time + I(time^2),
      random = ~ time, ng = k
    )
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
