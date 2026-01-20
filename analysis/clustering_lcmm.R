#!/usr/bin/env Rscript

# 1. Setup ----------------------------------------------------------------
message("Loading packages...")
library(devtools)
library(here)
devtools::load_all(here())
library(dplyr)
library(lcmm)

# 2. Load Data ------------------------------------------------------------
data_path <- here("data", "experiments_with_transforms.rds")

if (!file.exists(data_path)) {
  stop("Data file not found at: ", data_path)
}
message("Reading data from: ", data_path)
df_raw <- readRDS(data_path)

# Ensure models folder exist (to save the outputs)
if (!dir.exists(here("outputs", "models"))) dir.create(here("outputs", "models"), recursive = TRUE)

# 3. Preprocessing --------------------------------------------------------
message("Preprocessing data...")

model_data <- df_raw %>%
  dplyr::select(
    ID, time, experiment,
    ina_sqrt, na_sqrt, nna_sqrt, enna_sqrt, # Dependent variables
    excluded, injustice, personal, violence # Control variables
  ) %>%
  # Ensure numeric
  mutate(
    ID = as.numeric(as.factor(ID)),
    time = as.numeric(as.factor(time)),
    experiment = as.factor(experiment) # Must keep as factor
  )

# Normalize time
model_data$time <- model_data$time / max(model_data$time)

# Checks:
message("N experiments: ", length(unique(model_data$experiment)))
message("N participants: ", length(unique(model_data$ID)))
message("N unique time points: ", length(unique(model_data$time)))





# 4. Model Set A: Hellinger, Univariate (ENNA) -----------------------------
message("\nRunning Model Set A: Hellinger, Univariate (ENNA)...")

# --- Step 1: Define Formulas ---

# Base: Just time
formula_A_base <- enna_sqrt ~ time

# Option 1: Scenario-controls only (psychological factors)
formula_A_scenario_controls <- enna_sqrt ~ time +
  excluded + injustice + personal + violence

# Option 2: Experiment-control only (contextual factor)
formula_A_experiment_control <- enna_sqrt ~ time +
  experiment

# Option 3: Full Model (all controls)
formula_A_all_controls <- enna_sqrt ~ time +
  experiment +
  excluded + injustice + personal + violence

# Option 4: Quadratic formula
formula_A_all_controls_quad <- enna_sqrt ~ time + I(time^2) +
  experiment +
  excluded + injustice + personal + violence


# --- Step 2: Run Baselines (1 Class) ---

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


# --- Step 3: Test Fixed Effects (1 Class) ---

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
message("  - Fitting A5 (Full Controls)...")
m_a5 <- lcmm::multlcmm(
  data = model_data, subject = "ID", link = "5-equi-splines",
  fixed = formula_A_all_controls, random = ~ time, ng = 1
)
saveRDS(m_a5, here("outputs", "models", "m_a5_all_controls.rds"))

message("\n===== Summary for Model A5 (Linear Full) =====\n")
print(summary(m_a5))
message("\n===== End of Summary for Model A5 =====\n")


# --- Step 4: Quadratic Shape ---

# Model A5_Quad: Full Controls (Quadratic)
message("  - Fitting A5_Quad (Quadratic)...")
m_a5_quad <- lcmm::multlcmm(
  data = model_data, subject = "ID", link = "5-equi-splines",
  fixed = formula_A_all_controls_quad, random = ~ time, ng = 1
)
saveRDS(m_a5_quad, here("outputs", "models", "m_a5_all_controls_quad.rds"))

message("\n===== Summary for Model A5_Quad (Quadratic Full) =====\n")
print(summary(m_a5_quad))
message("\n===== End of Summary for Model A5_Quad =====\n")

