# analysis/formula_builder.R
library(here)

build_lcmm_formulas <- function(opt) {
  # 1. Outcomes
  if (opt$transform == "hellinger") {
    hellinger_map <- c(
      "ina"   = "ina_hellinger_inv",
      "na"    = "na_hellinger_inv",
      "nna"   = "nna_hellinger",
      "enna"  = "enna_hellinger"
    )
    if (opt$outcomes == "all") {
      outcome_vars <- unname(hellinger_map)
    } else {
      selected <- trimws(unlist(strsplit(opt$outcomes, ",")))
      if (!all(selected %in% names(hellinger_map))) {
        stop("Invalid --outcomes for hellinger.")
      }
      outcome_vars <- unname(hellinger_map[selected])
    }
  } else if (opt$transform == "ilr") {
    if (opt$outcomes == "all") {
      selected_idx <- c("1", "2", "3")
    } else {
      selected_idx <- trimws(unlist(strsplit(opt$outcomes, ",")))
      if (!all(selected_idx %in% c("1", "2", "3"))) {
        stop("Invalid --outcomes for ILR.")
      }
    }
    outcome_vars <- paste0("ilr_", opt$basis, "_", selected_idx)
  } else {
    stop("Unknown transform type.")
  }

  # 2. Controls & Time
  time_term <- if (opt$quadratic) "time + I(time^2)" else "time"
  rhs <- switch(
    opt$controls,
    "none" = time_term,
    "scenario" = paste(
      time_term,
      "+ excluded + injustice + personal + violence"
    ),
    "experiment" = paste(
      time_term,
      "+ experiment"
    ),
    "all" = paste(
      time_term,
      "+ experiment + excluded + injustice + personal + violence"
    ),
    stop("Invalid controls setting.")
  )

  # 3. Final & Random Formulas
  fixed_formula <- as.formula(
    paste(paste(outcome_vars, collapse = " + "), "~", rhs)
  )

  random_formula <- switch(
    as.character(opt$random),
    "1" = ~ 1,
    "2" = ~ time,
    "3" = ~ time + I(time^2),
    stop("Invalid --random option.")
  )

  # 4. Mixture Formula (Only used if K > 1)
  mixture_formula <- switch(
    as.character(opt$mixture),
    "1" = ~ 1,
    "2" = ~ time,
    "3" = ~ time + I(time^2),
    NULL
  )

  list(
    fixed      = fixed_formula,
    random     = random_formula,
    mixture    = mixture_formula,
    n_outcomes = length(outcome_vars)
  )
}
