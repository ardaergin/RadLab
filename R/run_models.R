

run_models <- function(data){

  # Setup
  action_options = c("ina", "na", "nna", "enna")
  output = list()

  cat("==============================", "\n")
  cat("Running Models...", "\n", "\n")

  ##### For Loop #####

  for (action_option in action_options){
    # Creating a list for action option
    models = list()

    ########## Linear Model ##########
    # Print the Model Name
    cat(paste("Running Linear Model for", action_option, "\n"))

    # The Model
    M_LM <- stats::lm(
      formula = paste(
        # Dependent Variable:
        action_option, " ~
          time +
          excluded + injustice + personal + violence +
          gender_f + age"),
      data = data)

    # Saving it to the List
    models[["M_LM"]] = M_LM

    ########## Simple Linear Mixed Effect Model ##########
    # Print the Model Name
    cat(paste("Running Simple Linear Mixed Effect Model for", action_option, "\n"))

    # The Model
    M_LMER_simple <- lmerTest::lmer(
      formula = paste(
        action_option, " ~
        time +
        excluded + injustice + personal + violence +
        gender_f + age +
        (1|ID)"),
      REML = F,
      data = data)

    # Saving it to the List
    models[["M_LMER_simple"]] = M_LMER_simple

    ########## Complex Linear Mixed Effect Model ##########
    # Print the Model Name
    cat(paste("Running Complex Linear Mixed Effect Model for", action_option, "\n"))

    # The Model
    M_LMER_complex <- lmerTest::lmer(
      formula = paste(
        action_option, " ~
        time +
        excluded + injustice + personal + violence +
        gender_f + age +
        (1 + time|ID)"),
      REML = F,
      control = lmerControl(optimizer = "Nelder_Mead"),
      data = data)

    # Saving it to the List
    models[["M_LMER_complex"]] = M_LMER_complex

    # Saving Everything into Big List
    output[[action_option]] = models


    cat("\n", "==============================", "\n", sep = "")
  }

  ##### Tests #####
  cat("Running Model Comparisons...", "\n", "\n")

  for (action_option in action_options){

    # First Identifier Print
    cat("Model Comparison for ", action_option, ":", "\n",
        sep = "")

    # C1
    results_1 <- stats::anova(
      output[[action_option]][[2]],
      output[[action_option]][[1]])

    # if this comparison is significant...
    if (results_1$`Pr(>Chisq)`[2] < 0.05){

      # run the next comparison
      results_2 <- stats::anova(
        output[[action_option]][[3]],
        output[[action_option]][[2]])

      # if this second comparison is significant...
      if (results_2$`Pr(>Chisq)`[2] < 0.05){
        # third model is best
        cat("The Complex Linear Mixed Effect Model is the best")

      } else {
        # second model is best
        cat("The Simple Linear Mixed Effect Model is the best")
      }

      # else the Linear model is best:
    } else {
      cat("The Simple Linear Model is the best")
    }

    cat("\n", "---------------", "\n", sep = "")
    # End of For Loop for model comparison:
  }

  return(output)
}

