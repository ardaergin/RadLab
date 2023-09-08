#' Merge Study with Pilot Scores
#'
#' @param excel_file The Excel file that includes the order of the vignettes in each study.
#' @param v_means_pilot A data frame that includes the pilot study vignette means
#' (the extremity scores).
#' @param study_number The corresponding column number in the excel sheet.
#' Note that the first column in the excel sheet is the names of the vignettes.
#'
#' @return a data frame that matches the vignette names & order within the particular study
#' with the extremity scores (means) derived from the pilot study.
#' @export
#'
#' @examples o_study1 <- RadLab::merge_with_pilot(studies_all, v.means_pilot, 1)
merge_with_pilot <- function(
    excel_file,
    v_means_pilot,
    study_number){

  # Selecting only the non-NA (filled) cells in the SPECIFIC COLUMN of the excel
  # "column_number + 2" because
  ## the first column should be vignette names
  ## and the second column is the data for the pilot study
  o_study_0 <- excel_file[
    !is.na(excel_file[, study_number + 2]), c(1, study_number + 2)]

  # Merging with the Pilot Scores
  o_study <- base::merge(
    v_means_pilot,
    o_study_0,
    by.x = "order",
    by.y = colnames(o_study_0[2]))

  return(o_study)
}
