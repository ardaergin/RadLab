#' Add Row-wise Sums to a Dataframe
#'
#' @param data A dataframe or tibble.
#' @param var_name The name of the new variable to be created (as a string).
#' @param item_names A character vector of the column names to sum up.
#' @param ignore_na Logical; defaults to FALSE. If FALSE, any NA in the row results
#' in an NA sum. If TRUE, the sum ignores NAs.
#'
#' @return A new dataframe that includes the new variable and is ungrouped.
#' @export
#'
#' @examples
#' # Example usage:
#' my_data <- data.frame(a = c(1, 2, NA), b = c(3, 4, 5))
#'
#' my_data %>%
#'   add_row_sums(
#'     var_name = "total",
#'     item_names = c("a", "b"),
#'     ignore_na = TRUE
#'   )
add_row_sums <- function(
    data,
    var_name,
    item_names,
    ignore_na = FALSE) {

  data <- data %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      !!var_name := sum(
        dplyr::c_across(dplyr::all_of(item_names)),
        na.rm = ignore_na)) %>%
    dplyr::ungroup() # Ensures subsequent operations are efficient

  return(data)
}
