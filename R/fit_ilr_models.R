#' Fit Hierarchical Models for ILR Variables with Optional Covariates
#'
#' This function fits hierarchical linear mixed models for a specified set of ILR-transformed
#' variables (e.g., `ilr1`, `ilr2`, `ilr3`). The models include random intercepts by `Experiment`
#' and random slopes and intercepts by `ID`. Additional control and covariate variables can be specified.
#'
#' @param data A data frame containing the relevant data for modeling.
#' @param ilr_vars A character vector specifying the ILR-transformed dependent variables.
#'                 Default is `c("ilr1", "ilr2", "ilr3")`.
#' @param control_vars A character vector specifying additional control variables to include
#'                     in the fixed effects. Default is `c("excluded", "injustice", "personal", "violence")`.
#' @param extra_covariates A character vector specifying any additional covariate variables (e.g., `age`, `gender`).
#'                         Default is `NULL`.
#' @param time_var A string specifying the time variable in the data. Default is `"time"`.
#' @param id_var A string specifying the participant ID variable. Default is `"ID"`.
#' @param experiment_var A string specifying the experiment variable. Default is `"Experiment"`.
#'
#' @return A list containing the fitted models for each specified ILR variable.
#' @import lme4
#' @export
#' @examples
#' # Example usage with extra covariates
#' \dontrun{
#' data <- your_dataframe
#' models <- fit_ilr_models(data = data, extra_covariates = c("age", "gender"))
#' summary(models$ilr1_model)
#' }
fit_ilr_models <- function(data,
                           ilr_vars = c("ilr1", "ilr2", "ilr3"),
                           control_vars = c("excluded", "injustice", "personal", "violence"),
                           extra_covariates = NULL,
                           time_var = "time",
                           id_var = "ID",
                           experiment_var = "Experiment") {

  # Combine all covariates
  all_covariates <- c(time_var, control_vars, extra_covariates)

  # Helper function to fit an individual model for a given ILR variable
  fit_single_model <- function(ilr_var) {
    # Create the formula dynamically
    formula <- as.formula(
      paste(ilr_var, "~",
            paste(all_covariates, collapse = " + "),
            "+ (1 +", time_var, "|", id_var, ") + (1 |", experiment_var, ")")
    )

    # Fit the model using lmer
    model <- lmer(formula, data = data)
    return(model)
  }

  # Fit models for each specified ILR variable
  models <- lapply(ilr_vars, fit_single_model)

  # Name the models in the output list for easy reference
  names(models) <- paste(ilr_vars, "model", sep = "_")

  return(models)
}
