% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/duration_exclusion.R
\name{duration_exclusion}
\alias{duration_exclusion}
\title{Exclude Participants Based on Duration}
\usage{
duration_exclusion(data, duration_column)
}
\arguments{
\item{data}{A data frame containing survey data.}

\item{duration_column}{A string specifying the column name in `data` representing the duration.}
}
\value{
A filtered data frame with participants outside the defined duration range excluded.
}
\description{
This function filters out participants with duration values that are unusually
fast or slow, defined as being less than half or more than twice the median
duration. It returns a filtered dataset and displays information on the number
of excluded cases as well as before-and-after histograms of the `duration_column`.
}
\examples{
duration_exclusion(data = my_data, duration_column = "completion_time")
}
