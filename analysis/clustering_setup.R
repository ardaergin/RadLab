# analysis/clustering_setup.R

# 1. Setup & Config -------------------------------------------------------
source("analysis/config.R")

message("Loading analysis packages...")
library(dplyr)
library(lcmm)
library(here)


# 2. Load Data ------------------------------------------------------------
file_path <- here::here("data", opt$file)
if (!file.exists(file_path)) {
  stop("Data file not found at: ", file_path)
}
message("Reading data from: ", file_path)
df <- readRDS(file_path)

# Ensure models folder exist (to save the outputs)
if (!dir.exists(here("outputs", "models"))) dir.create(here("outputs", "models"), recursive = TRUE)

# 3. Preprocessing --------------------------------------------------------
message("Preprocessing data...")

if (opt$transform == "hellinger") {
  multivariate_vars <- c(
    "ina_hellinger_inv",
    "na_hellinger_inv",
    "nna_hellinger",
    "enna_hellinger"
  )
} else if (opt$transform == "ilr") {
  multivariate_vars <- paste0("ilr", "_", opt$basis, "_", 1:3)
} else {
  stop("Unknown transform type: ", opt$transform)
}

# Check variables
vars_to_check <- c(
  multivariate_vars,
  "excluded", "injustice", "personal", "violence"
)
print(summary(df[, vars_to_check]))

message("Setup Complete.")
message("N experiments: ", length(unique(df$experiment)))
message("N participants: ", length(unique(df$ID)))


# 4. Define Formulas ------------------------------------------------------

# A. Determine Outcomes
if (opt$mode == "univariate") {
  if (opt$transform == "ilr") {
    stop("Invalid mode: ILR does not support univariate ENNA test.")
  }
  message("-> Mode: UNIVARIATE (hellinger)")
  # Univariate always targets ENNA in your workflow
  outcome_vars <- "enna_hellinger"
  outcomes_str <- outcome_vars

} else if (opt$mode == "multivariate") {
  message("-> Mode: MULTIVARIATE")
  outcome_vars <- multivariate_vars
  outcomes_str <- paste(outcome_vars, collapse = " + ")

} else {
  stop("Invalid mode: ", opt$mode)
}

# B. Determine Time Structure
time_term <- if(opt$quadratic) "time + I(time^2)" else "time"

# C. Determine Controls
if (opt$controls == "none") {
  rhs <- time_term
} else if (opt$controls == "scenario") {
  rhs <- paste(time_term, "+ excluded + injustice + personal + violence")
} else if (opt$controls == "experiment") {
  rhs <- paste(time_term, "+ experiment")
} else if (opt$controls == "all") {
  rhs <- paste(time_term, "+ experiment + excluded + injustice + personal + violence")
} else {
  stop("Invalid controls setting: ", opt$controls)
}

# D. Create Final Formula
final_formula <- as.formula(paste(outcomes_str, "~", rhs))

# E. Derive Internal Parameters
n_outcomes <- length(outcome_vars)

# F. Define Random Formula safely
if (opt$random == "1") {
  random_formula <- ~ 1
  rand_label     <- "RI" # Random Intercept
} else if (opt$random == "time") {
  random_formula <- ~ time
  rand_label     <- "RS" # Random Slope (Linear)
} else if (opt$random == "time_quad") {
  random_formula <- ~ time + I(time^2)
  rand_label     <- "RQ" # Random Quadratic
} else {
  stop("Invalid --random option provided.")
}

message("------------------------------------------------")
message("CONFIGURATION:")
message("  Mode: ", opt$mode)
message("  Link: ", opt$link)
message("  Controls: ", opt$controls)
message("  Quadratic: ", opt$quadratic)
message("FORMULA:")
message("  Fixed Effects:  ", format(final_formula))
message("  Random Effects: ", format(random_formula))
message("------------------------------------------------")
