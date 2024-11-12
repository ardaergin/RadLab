#' Exclude Participants Based on Duration
#'
#' This function filters out participants with duration values that are unusually
#' fast or slow, defined as being less than half or more than twice the median
#' duration. It returns a filtered dataset and displays information on the number
#' of excluded cases as well as before-and-after histograms of the `duration_column`.
#'
#' @param data A data frame containing survey data.
#' @param duration_column A string specifying the column name in `data` representing the duration.
#' @return A filtered data frame with participants outside the defined duration range excluded.
#' @examples
#' duration_exclusion(data = my_data, duration_column = "completion_time")
#' @export
duration_exclusion <- function(data, duration_column) {
  # Calculate median, upper bound, and lower bound
  med <- median(data[[duration_column]], na.rm = TRUE)
  upper_bound <- med * 2
  lower_bound <- med / 2

  # Print median, upper bound, and lower bound
  message("Median duration: ", med, " seconds")
  message("Upper bound: ", upper_bound, " seconds")
  message("Lower bound: ", lower_bound, " seconds")

  # Filter data based on calculated bounds
  data_filtered <- data %>% filter(
    .data[[duration_column]] < upper_bound & .data[[duration_column]] > lower_bound
  )

  # Print the number of excluded participants
  excluded_count <- nrow(data) - nrow(data_filtered)
  message("Number of participants excluded: ", excluded_count)

  # Plot histograms before and after filtering
  library(ggplot2)
  p1 <- ggplot(data, aes(x = .data[[duration_column]])) +
    geom_histogram(binwidth = 100, fill = "darkgreen", alpha = 0.5) +
    ggtitle("Before Filtering") +
    xlab("Duration (seconds)") +
    ylab("Frequency")

  p2 <- ggplot(data_filtered, aes(x = .data[[duration_column]])) +
    geom_histogram(binwidth = 100, fill = "firebrick", alpha = 0.5) +
    ggtitle("After Filtering") +
    xlab("Duration (seconds)") +
    ylab("Frequency")

  # Display plots side by side
  gridExtra::grid.arrange(p1, p2, ncol = 2)

  return(data_filtered)
}
