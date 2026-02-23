#' Plot Stacked Histograms
#'
#' Creates a vertical stack of histograms for exploring distributions across multiple variables.
#'
#' @param data A data frame containing the variables to plot.
#' @param cols A character vector of column names to include in the stack.
#' @param labels A character vector of labels for the facet headers. Must match the length of `cols`.
#' @param colors A character vector of hex colors for the fill. Must match the length of `cols`.
#' @param title The main title of the plot.
#' @param subtitle An optional subtitle for the plot. Default is NULL.
#' @param xlab The x-axis label. Default is "Value".
#' @param x_breaks A numeric vector specifying where the x-axis ticks/gridlines should appear.
#' @param x_limits A numeric vector of length 2 specifying the x-axis boundaries (e.g., c(-5, 5)).
#' @param show_zero_line Logical; if TRUE, adds a dashed vertical line at 0 (useful for ILR).
#' @param bins Integer; the number of histogram bins. Default is 60.
#' @param exclude_values A numeric vector of values to exclude from the plot (e.g., c(0, 100)).
#'                       This filters specific observations (in long format) to allow the y-axis
#'                       to rescale, revealing the distribution of the remaining points.
#'
#' @return A ggplot object.
#' @export
#'
#' @examples
#' plot_stacked_distributions(df, cols = c("ina", "na"), labels = c("Inaction", "Normative"),
#'                            colors = c("gray", "blue"), title = "Test Plot")

plot_stacked_distributions <- function(
    data,
    cols,
    labels,
    colors,
    title,
    subtitle = NULL,
    xlab = "Value",
    x_breaks = seq(0, 100, 10),
    x_limits = NULL,
    show_zero_line = FALSE,
    bins = 60,
    exclude_values = NULL
) {

  # 1. Prepare Data (Reshape FIRST, so filtering doesn't drop whole participants)
  long_df <- data %>%
    dplyr::select(dplyr::all_of(cols)) %>%
    tidyr::pivot_longer(cols = dplyr::everything(), names_to = "Component", values_to = "Value") %>%
    dplyr::mutate(Component = factor(Component, levels = cols, labels = labels))

  # 2. Filter specific VALUES, not participants
  if (!is.null(exclude_values)) {
    long_df <- long_df %>%
      dplyr::filter(!Value %in% exclude_values)
  }

  # 3. Build Base Plot
  p <- ggplot2::ggplot(long_df, ggplot2::aes(x = Value, fill = Component))

  if (show_zero_line) {
    p <- p + ggplot2::geom_vline(xintercept = 0, linetype = "dashed", color = "black", alpha = 0.3)
  }

  p <- p +
    # Use 'free_y' so each facet scales to its own data, not the massive zero-count
    ggplot2::geom_histogram(bins = bins, color = "white", linewidth = 0.2, alpha = 0.9, boundary = 0) +
    ggplot2::facet_wrap(~Component, nrow = length(cols), scales = "free_y") +
    ggplot2::scale_fill_manual(values = colors) +
    ggplot2::scale_x_continuous(breaks = x_breaks, expand = ggplot2::expansion(mult = c(0.05, 0.05))) +
    ggplot2::labs(title = title, subtitle = subtitle, x = xlab, y = "Frequency (Count)") +
    ggplot2::theme_minimal(base_size = 14) +
    ggplot2::theme(
      legend.position = "none",
      strip.text = ggplot2::element_text(face = "bold", size = 13, hjust = 0, margin = ggplot2::margin(b = 10)),
      panel.grid.major.y = ggplot2::element_blank(),
      panel.grid.minor.y = ggplot2::element_blank(),
      panel.grid.major.x = ggplot2::element_line(color = "gray92", linewidth = 0.5),
      panel.grid.minor.x = ggplot2::element_blank(),
      axis.line.x = ggplot2::element_line(color = "black", linewidth = 0.6),
      plot.title = ggplot2::element_text(face = "bold", size = 18, margin = ggplot2::margin(b = 10)),
      plot.subtitle = ggplot2::element_text(size = 12, color = "gray30", margin = ggplot2::margin(b = 20)),
      panel.spacing = ggplot2::unit(2, "lines"),
      plot.margin = ggplot2::margin(t = 20, r = 30, b = 20, l = 30)
    )

  if (!is.null(x_limits)) {
    p <- p + ggplot2::coord_cartesian(xlim = x_limits)
  }

  return(p)
}
