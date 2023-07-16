#' Wide to Long Transformation for RadLab Vignette Data
#'
#' @param data the data frame that you want to transform from the wide format to the long format
#' @param name_data a data frame with names of vignettes and their corresponding order
#' @param n_vignettes Total number of vignettes
#' @param IDvar ID variable
#'
#' @return A long data frame suitable for modelling
#' @export
#'
#' @examples d_hartog_long <- d_hartog %>% RadLab::wide_to_long(
#'     name_data = sv_hartog,
#'     n_vignettes = 15)

wide_to_long <- function(
    data,
    name_data,
    n_vignettes,
    IDvar = "ID"){

  # Making sure its in the data frame format
  # Causes problems with "reshape" if otherwise
  data <- as.data.frame(data)


  ##### Changing Column Names #####
  # Renaming Inaction
  for (i in 1:n_vignettes){
    colnames(data)[seq(1, n_vignettes*4, 4)][i] <- paste("ina", i, sep = "_")
  }
  # Renaming Normative Action
  for (i in 1:n_vignettes){
    colnames(data)[seq(2, n_vignettes*4, 4)][i] <- paste("na", i, sep = "_")
  }
  # Renaming Non-Normative Action
  for (i in 1:n_vignettes){
    colnames(data)[seq(3, n_vignettes*4, 4)][i] <- paste("nna", i, sep = "_")
  }
  # Renaming Extreme Non-Normative Action
  for (i in 1:n_vignettes){
    colnames(data)[seq(4, n_vignettes*4, 4)][i] <- paste("enna", i, sep = "_")
  }

  middle <- stats::reshape(
    data = data,
    direction = 'long',
    varying = 1:(n_vignettes*4),
    timevar = 'time',
    times = 1:n_vignettes,
    v.names = c("ina",
                "na",
                "nna",
                "enna"),
    idvar = 'ID')


  ##### Merging with the Pilot Responses #####
  final <- merge(
    middle, name_data,

    # ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION ATTENTION
    # If there is an error, first check if these column names are correct
    by.x = "time",
    by.y = "order")

  ##### Converting to Factors #####
  # final$time <- factor(final$time, ordered = TRUE)

  final$ID <- factor(final$ID)

  final$Vignette <- factor(final$Vignette)

  return(final)
}
