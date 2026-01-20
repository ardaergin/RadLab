
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
    ina_sqrt, na_sqrt, nna_sqrt, enna_sqrt,
    excluded, injustice, personal, violence
  ) %>%
  mutate(
    ID = as.numeric(as.factor(ID)),
    time = as.numeric(as.factor(time)),
    experiment = as.factor(experiment)
  )

# Normalize time
model_data$time <- model_data$time / max(model_data$time)

# Checks:
message("Setup Complete.")
message("N experiments: ", length(unique(model_data$experiment)))
message("N participants: ", length(unique(model_data$ID)))

# 4. Define Formulas ------------------------------------------------------

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
