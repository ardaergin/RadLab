#!/usr/bin/env Rscript

# Loading the shared setup (Libraries + Data + Preprocessing + Formulas)
source("analysis/setup.R")


# Model Set A: Hellinger, Univariate (ENNA) -----------------------------
message("\nRunning Model Set A: Hellinger, Univariate (ENNA)...")

# --- Step 1: Run Baselines (1 Class) ---

# Model A1: Base (Random Intercept)
message("Fitting A1 (Base, RI)...")
m_a1 <- lcmm::multlcmm(
  data = model_data, subject = "ID", link = "5-equi-splines",
  fixed = formula_A_base, random = ~ 1, ng = 1
)
saveRDS(m_a1, here("outputs", "models", "m_a1_base_RI.rds"))

message("\n===== Summary for Model A1 (Base, RI) =====\n")
print(summary(m_a1))
message("\n===== End of Summary for Model A1 =====\n")


# Model A2: Base (Random Slope)
message("Fitting A2 (Base, RI + RS)...")
m_a2 <- lcmm::multlcmm(
  data = model_data, subject = "ID", link = "5-equi-splines",
  fixed = formula_A_base, random = ~ time, ng = 1
)
saveRDS(m_a2, here("outputs", "models", "m_a2_base_RS.rds"))

message("\n===== Summary for Model A2 (Base, RI + RS) =====\n")
print(summary(m_a2))
message("\n===== End of Summary for Model A2 =====\n")


# --- Step 2: Test Fixed Effects (1 Class) ---

# Model A3: Scenario Controls
message("Fitting A3 (Scenario Controls)...")
m_a3 <- lcmm::multlcmm(
  data = model_data, subject = "ID", link = "5-equi-splines",
  fixed = formula_A_scenario_controls, random = ~ time, ng = 1
)
saveRDS(m_a3, here("outputs", "models", "m_a3_scenario_controls.rds"))

message("\n===== Summary for Model A3 (Scenario Controls) =====\n")
print(summary(m_a3))
message("\n===== End of Summary for Model A3 =====\n")


# Model A4: Experiment Controls
message("Fitting A4 (Experiment Controls)...")
m_a4 <- lcmm::multlcmm(
  data = model_data, subject = "ID", link = "5-equi-splines",
  fixed = formula_A_experiment_control, random = ~ time, ng = 1
)
saveRDS(m_a4, here("outputs", "models", "m_a4_experiment_controls.rds"))

message("\n===== Summary for Model A4 (Experiment Controls) =====\n")
print(summary(m_a4))
message("\n===== End of Summary for Model A4 =====\n")


# Model A5: Full Controls (Linear)
message("Fitting A5 (Full Controls)...")
m_a5 <- lcmm::multlcmm(
  data = model_data, subject = "ID", link = "5-equi-splines",
  fixed = formula_A_all_controls, random = ~ time, ng = 1,
  maxiter = 500 # this specific model did not converge with maxiter=100
  # So, just for the sake of being able to compare it with other models,
  # I am increasing the maxiter, so we find the 'least bad' linear fit.
)
saveRDS(m_a5, here("outputs", "models", "m_a5_all_controls.rds"))

message("\n===== Summary for Model A5 (Linear Full) =====\n")
print(summary(m_a5))
message("\n===== End of Summary for Model A5 =====\n")


# --- Step 3: Test Quadratic Shape ---

# Model A6: Full Controls (Quadratic)
message("Fitting A6 (Full Controls + Quadratic)...")
m_a5_quad <- lcmm::multlcmm(
  data = model_data, subject = "ID", link = "5-equi-splines",
  fixed = formula_A_all_controls_quad, random = ~ time, ng = 1
)
saveRDS(m_a5_quad, here("outputs", "models", "m_a5_all_controls_quad.rds"))

message("\n===== Summary for Model A5_Quad (Quadratic Full) =====\n")
print(summary(m_a5_quad))
message("\n===== End of Summary for Model A5_Quad =====\n")

message("\nAnalysis Complete.")
