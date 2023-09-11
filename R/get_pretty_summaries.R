get_pretty_summaries <- function(
    models_list
    ){
  cat(
    sjPlot::tab_model(
      models_list$ina$M_LMER_complex,
      models_list$na$M_LMER_complex,
      models_list$nna$M_LMER_complex,
      models_list$enna$M_LMER_complex)$knitr,
    "\n--------\n"
  )
}
