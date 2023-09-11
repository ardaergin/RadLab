
get_model_comparisons <- function(models_list){
  # For some reason, the for loop does not work here

  # INA
  cat(
    sjPlot::tab_model(
      models_list[["ina"]][["M_LM"]],
      models_list[["ina"]][["M_LMER_simple"]],
      models_list[["ina"]][["M_LMER_complex"]])$knitr,
    "\n"
  )
  # NA
  cat(
    sjPlot::tab_model(
      models_list[["na"]][["M_LM"]],
      models_list[["na"]][["M_LMER_simple"]],
      models_list[["na"]][["M_LMER_complex"]])$knitr,
    "\n"
  )
  # NNA
  cat(
    sjPlot::tab_model(
      models_list[["nna"]][["M_LM"]],
      models_list[["nna"]][["M_LMER_simple"]],
      models_list[["nna"]][["M_LMER_complex"]])$knitr,
    "\n"
  )
  # ENNA
  cat(
    sjPlot::tab_model(
      models_list[["enna"]][["M_LM"]],
      models_list[["enna"]][["M_LMER_simple"]],
      models_list[["enna"]][["M_LMER_complex"]])$knitr,
    "\n"
  )
}


