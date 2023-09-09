#' Get Summaries for the LM(ER) Models
#'
#' @param models_list The model list derived from the RadLab::run_models() function.
#' @param the_model The best fitted model.
#'
#' @return Prints the summary() and anova() for each action option.
#' @export
#'
#' @examples

get_model_results <- function(models_list,
                              the_model = "M_LMER_complex"){

  action_options = c(
    "INACTION",
    "NORMATIVE ACTION",
    "NON-NORMATIVE ACTION",
    "EXTREME NON-NORMATIVE ACTION")
  # For loop for the 4 action options
  for (i in 1:4){

    cat("\n", "\n", "**********", action_options[i], "**********", "\n", "\n")

    print(summary(models_list[[i]][[the_model]]))

    cat("\n", "\n", "------------------------------", "\n", "\n")

    print(stats::anova(models_list[[i]][[the_model]]))

    cat("\n", "\n", "------------------------------", "\n", "\n")

    cat("Marginal (R2m) and Conditional (R2c) R-squared:", "\n", "\n")

    print(MuMIn::r.squaredGLMM(models_list[[i]][[the_model]]))

    cat("\n", "\n", "\n", "===========================", "\n", "\n", "\n", "\n", "\n")
  }
}
