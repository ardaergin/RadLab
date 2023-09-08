#' Sum the Rows of a Dataframe
#'
#' @param data a dataframe
#' @param var_name the name of the new variable that is to be created
#' @param item_names the name of the items to sum up the points for
#' @param ignore_na FALSE by default:
#' if there is one NA, the total sum will be NA.
#' if TRUE, the sum ignores the NA and the sum only involves the non-NA.
#'
#' @return a new dataframe that includes the new variable
#' @export
#'
#' @examples
row_sums <- function(
    data,
    var_name,
    item_names,
    ignore_na = F){

  data <- data %>%
    dplyr::rowwise() %>%
    dplyr::mutate(
      !!var_name := sum(
        dplyr::c_across(dplyr::all_of(item_names)),
        na.rm = ignore_na))

  return(data)
}
