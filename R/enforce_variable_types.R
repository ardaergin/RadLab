#' Enforce Variable Types for Analysis
#'
#' @description
#' Iterates through a data frame and ensures every column is correctly classed.
#' It converts blank strings to `NA`, transforms labelled variables into
#' standard R factors, and forces all other columns to numeric.
#'
#' @param data A data frame containing character, factor, or labelled variables.
#'
#' @return A data frame where all columns are strictly either factors or numeric.
#' @export
#'
#' @examples
#' # d_study1 <- enforce_variable_types(d_study1_raw)

enforce_variable_types <- function(data) {

  ##### Fixing "" data (if there is any) #####
  data <- as.data.frame(data) %>%
    dplyr::mutate_if(is.character, ~dplyr::na_if(., ""))

  ##### Identifying Factor Columns #####
  # Columns already recognized as factors by R
  factor.columns <- names(data)[which(sapply(data, is.factor))]

  ##### Loop through each column name #####
  for (col_name in names(data)) {

    ##### Factor Columns Conversion #####
    if (col_name %in% factor.columns) {
      data[[col_name]] <- labelled::to_factor(data[[col_name]])
    }
    ##### Numeric Columns Conversion #####
    else {
      # This removes the 'labelled' attribute and enforces numeric type
      data[[col_name]] <- as.numeric(data[[col_name]])
    }
  }

  return(data)
}
