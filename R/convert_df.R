#' Convert a SPSS Data Frame (imported with haven::) Into an Analysis-Ready One
#'
#' @param data Your data file that either has numeric or factor columns. The character columns will be NA.
#'
#' @return A data frame that is haven-labelled variables are converted.
#' @export
#'
#' @examples converted_dataframe <- convert.df(spss_imported_dataframe)

convert_df <- function(data) {

  ##### Fixing "" data (if there is any) #####
  data <- as.data.frame(data) %>%
    dplyr::mutate_if(is.character, ~dplyr::na_if(., ""))

  ##### Identifying Factor Columns #####
  factor.columns <-
    names(data)[which(sapply(data, is.factor))]

  ##### Loop #####
  for (names in names(data)) {

    ##### Factor Columns Conversion #####
    if (any(names %in% factor.columns)) {
      data[names] <-
        labelled::to_factor(data[[names]])
    }
    ##### Numeric Columns Conversion #####
    else {
      data[names] <-
        as.numeric(data[[names]])
    }
  }

  return(data)
}
