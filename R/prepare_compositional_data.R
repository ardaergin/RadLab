#' Prepare Compositional Data with ilr Transformation
#'
#' This function processes compositional data by calculating proportions,
#' handling zeros, performing isometric log-ratio (ilr) transformation,
#' and returning a data frame with the original data and ilr coordinates.
#'
#' @param data A data frame containing the compositional data with columns
#'   \code{ina}, \code{na}, \code{nna}, and \code{enna}.
#' @return A data frame containing the original data and ilr-transformed coordinates.
#' @export
#'
#' @import dplyr
#' @importFrom compositions ilr acomp
#' @importFrom zCompositions cmultRepl
#'
#' @examples
#' # Example usage
#' data <- data.frame(
#'   ID = 1:5,
#'   ina = c(10, 0, 5, 20, 0),
#'   na = c(40, 50, 45, 30, 50),
#'   nna = c(30, 40, 35, 40, 30),
#'   enna = c(20, 10, 15, 10, 20)
#' )
#' df_ilr <- prepare_compositional_data(data)
prepare_compositional_data <- function(data) {

  # Check if required columns exist
  required_cols <- c("ina", "na", "nna", "enna")
  missing_cols <- setdiff(required_cols, names(data))
  if (length(missing_cols) > 0) {
    stop("The following required columns are missing from the data: ",
         paste(missing_cols, collapse = ", "))
  }

  # Check for missing values (NAs) in the required columns
  if (any(is.na(data[required_cols]))) {
    stop("The input data contains missing values (NAs). ",
         "Please handle missing values before proceeding with the ILR transformation.")
  }


  # Step 1: Data preparation
  df <- data %>%
    dplyr::mutate(
      inaction_prop = ina / 100,
      normative_prop = na / 100,
      nonnormative_prop = nna / 100,
      extreme_nonnormative_prop = enna / 100
    )

  # Step 2: Create a matrix of the compositional data
  comp_matrix <- as.matrix(
    df[, c('inaction_prop',
           'normative_prop',
           'nonnormative_prop',
           'extreme_nonnormative_prop')]
  )

  # Step 3: Replace zeros using Count Zero Multiplicative method (CZM)
  cat("Replacing zeros using Count Zero Multiplicative method (CZM)... \n")

  comp_matrix_nozeros <- zCompositions::cmultRepl(
    comp_matrix,
    method = "CZM",
    output = "prop",
    label = 0,
    z.delete = FALSE
  )

  # Step 4: Create acomp object
  comp_data <- compositions::acomp(comp_matrix_nozeros)

  # Step 5: Perform ilr transformation
  cat("Performing ilr-transformation... \n")

  ilr_data <- compositions::ilr(comp_data)

  # Step 6: Convert ilr_data to a data frame and name the columns
  ilr_df <- as.data.frame(ilr_data)
  colnames(ilr_df) <- paste0('ilr', 1:ncol(ilr_df))

  # Step 7: Combine ilr coordinates with the original data
  df_ilr <- cbind(df, ilr_df)

  # Return the resulting data frame
  cat("Your data is ready for CoDA!")

  return(df_ilr)
}
