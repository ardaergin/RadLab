#' Graph Vignette Means from Data
#'
#' @param data A data frame with only data from vignette responses.
#' @param name_data A data frame with names of vignettes and their corresponding order
#' (e.g., for row 1, column 1, it is "Bakery", and row 1, column 2 is 1).
#' @param n_vignettes Total number of vignettes.
#' @param label_x Logical; if TRUE, labels the x-axis with vignette names. Defaults to TRUE.
#'
#' @return A ggplot2 object displaying the means of vignette responses across action types.
#' @export
#'
#' @examples
#' # Example usage:
#' plot_vignette_means(
#'   data = d_study1[, 1:60],
#'   name_data = o_study1,
#'   n_vignettes = 15
#' )

plot_vignette_means <- function(
    data,
    name_data,
    n_vignettes,
    label_x = TRUE){

  ##### Calculating #####
  # Calculating means and storing them:
  v.means_raw <- colMeans(data, na.rm = TRUE)

  # Creating a data frame of vignette means:
  v.means <- as.data.frame(
    matrix(0,
           ncol = 5,
           nrow = n_vignettes))

  # Giving name to columns
  colnames(v.means) <- c(
    'inaction',
    'normative',
    'nna',
    'enna',
    'order')

  # Loop to assign means:
  for (i in 1:4) {
    v.means[i] <- v.means_raw[seq(i, n_vignettes*4, 4)]
  }
  v.means[5] <- 1:n_vignettes


  # Checking if they all sum up to a 100:
  if (all(rowSums(v.means[1:4]) == 100)){
    print("all rows sum up to a 100, so everything is good!")
  } else{
    print(rowSums(v.means[1:4]) == 100)
  }

  # Adding the Names of Vignettes
  v.means_with_names <- merge(
    v.means,
    name_data,
    by.x = "order",
    by.y = "order")


  ##### Plotting #####
  the_plot <- ggplot2::autoplot(
    zoo::zoo(v.means_with_names[2:5]),
    facet = NULL) +

    ggplot2::geom_point() +

    ggplot2::theme_classic() +

    ggplot2::theme(
      axis.text.x = ggplot2::element_text(
        color = "black", size = 9,
        angle = 45, hjust = 1)) +

    ggplot2::scale_color_discrete(
      name = "Action Type",
      labels = c("Inaction",
                 "Normative Action",
                 "Non-normative Action",
                 "Extreme Non-normative Action"))

  if (label_x) {
    the_plot <- the_plot + ggplot2::scale_x_discrete(
      limits = factor(1:n_vignettes),
      labels = v.means_with_names$Vignette)
  }

  return(the_plot)
}

