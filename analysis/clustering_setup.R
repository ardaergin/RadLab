
# 1. Setup ----------------------------------------------------------------
message("Loading packages...")
library(here)
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

# Get Command Line Arguments
args <- commandArgs(trailingOnly = TRUE)
# Default to setting 1 (Multivariate) if no argument is passed
setting <- if(length(args) > 0) as.numeric(args[1]) else 2

if (setting == 1) {
  message("Running in UNIVARIATE mode (enna_sqrt only)")
  outcomes <- "enna_sqrt"
  n_outcomes <- 1
} else if (setting == 2) {
  message("Running in MULTIVARIATE mode (All 4 DVs)")
  outcomes <- "ina_sqrt + na_sqrt + nna_sqrt + enna_sqrt"
  n_outcomes <- 4
} else {
  stop("Invalid setting provided. Use 1 for Univariate or 2 for Multivariate.")
}

# Base: Just time
formula_base <- as.formula(paste(
  outcomes, "~ time"
))

# 1: Scenario-controls only (psychological factors)
formula_scenario_controls <- as.formula(paste0(
  outcomes, " ~ time + excluded + injustice + personal + violence"
))

# 2: Experiment-control only (contextual factor)
formula_experiment_control <- as.formula(paste0(
  outcomes, " ~ time + experiment"
))

# 3: Full Model (all controls)
formula_all_controls <- as.formula(paste0(
  outcomes, " ~ time + experiment + excluded + injustice + personal + violence"
))

# 4: Quadratic formula
formula_all_controls_quad <- as.formula(paste0(
  outcomes, " ~ time + I(time^2) + experiment + excluded + injustice + personal + violence"
))
