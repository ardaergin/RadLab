get_pretty_summaries <- function(
    models_list,
    action_comparison = TRUE,
    model_comparison = FALSE
    ){

  if (action_comparison == TRUE){
    cat(
      sjPlot::tab_model(
        models_list$ina$M_LMER_complex,
        models_list$na$M_LMER_complex,
        models_list$nna$M_LMER_complex,
        models_list$enna$M_LMER_complex)$knitr,
      "\n--------\n"
      )
  }

  if (model_comparison == TRUE){
    # For some reason, the for loop does not work here

    cat("\n--------\n")
    # INA
    cat(
      sjPlot::tab_model(
        models_list[["ina"]][["M_LM"]],
        models_list[["ina"]][["M_LMER_simple"]],
        models_list[["ina"]][["M_LMER_complex"]])$knitr,
      "\n--------\n"
    )
    # NA
    cat(
      sjPlot::tab_model(
        models_list[["na"]][["M_LM"]],
        models_list[["na"]][["M_LMER_simple"]],
        models_list[["na"]][["M_LMER_complex"]])$knitr,
      "\n--------\n"
    )
    # NNA
    cat(
      sjPlot::tab_model(
        models_list[["nna"]][["M_LM"]],
        models_list[["nna"]][["M_LMER_simple"]],
        models_list[["nna"]][["M_LMER_complex"]])$knitr,
      "\n--------\n"
    )
    # ENNA
    cat(
      sjPlot::tab_model(
        models_list[["enna"]][["M_LM"]],
        models_list[["enna"]][["M_LMER_simple"]],
        models_list[["enna"]][["M_LMER_complex"]])$knitr,
      "\n--------\n"
    )

  }
}
