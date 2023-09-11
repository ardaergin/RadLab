get_pretty_summaries <- function(
    models_list,
    action_comparison = TRUE,
    model_comparison = FALSE
    ){

  if (action_comparison == TRUE){
    cat(tab_model(r)$knitr,"\n--------\n")
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
      for(i in 1:4){
        cat(
          sjPlot::tab_model(
            models_list[[i]][["M_LM"]],
            models_list[[i]][["M_LMER_simple"]],
            models_list[[i]][["M_LMER_complex"]])$knitr,
          "\n--------\n"
        )
      }
  }
}
