#' Wide to Long Transformation for RadLab Vignette Data
#'
#' @description
#' Transforms interleaved vignette data from wide to long format.
#' **Note:** This function assumes that the vignette response columns
#' (4 per vignette) are the FIRST columns in the data frame.
#'
#' @param data A data frame where the first (n_vignettes * 4) columns are
#' interleaved vignette responses (ina, na, nna, enna).
#' @param name_data A data frame with names of vignettes and their
#' corresponding order (must contain 'order' and 'Vignette' columns).
#' @param n_vignettes Total number of vignettes (integer).
#' @param IDvar Character string specifying the name of the ID variable.
#' Defaults to "ID".
#'
#' @return A long-format data frame with columns: ID, time, ina, na, nna, enna,
#' and Vignette name.
#' @export
#'
#' @examples data_long <- data_wide %>% RadLab::wide_to_long(
#'   name_data = name_data,
#'   n_vignettes = 15,
#'   IDvar = "ID"
#' )

wide_to_long <- function(
    data,
    name_data,
    n_vignettes,
    IDvar = "ID"){

  # Making sure its in the data frame format
  # Causes problems with stats::reshape() if otherwise
  data <- as.data.frame(data)

  # Define the interleaved mapping via indices
  varying_list <- list(
    seq(1, n_vignettes * 4, 4), # Mapping for v.name "ina"
    seq(2, n_vignettes * 4, 4), # Mapping for v.name "na"
    seq(3, n_vignettes * 4, 4), # Mapping for v.name "nna"
    seq(4, n_vignettes * 4, 4)  # Mapping for v.name "enna"
  )

  # Reshape using the mapping list
  middle <- stats::reshape(
    data = data,
    direction = 'long',
    varying = varying_list,
    timevar = 'time',
    times = 1:n_vignettes,
    v.names = c("ina", "na", "nna", "enna"),
    idvar = IDvar
  )

  # Merge with vignette names on time=order.
  final <- merge(middle, name_data, by.x = "time", by.y = "order")

  # Factorize
  final$ID <- factor(final$ID)
  final$Vignette <- factor(final$Vignette)

  return(final)
}
