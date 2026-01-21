
suppressPackageStartupMessages({
  library(optparse)
})

option_list <- list(
  # --- Input/Output ---
  make_option(
    c("-f", "--file"), type = "character",
    default = "experiments_with_transforms.rds",
    help = "Path to input data file. [default= %default]"
  ),

  # --- Transform ---
  make_option(
    c("-t", "--transform"), type = "character",
    default = "sqrt",
    help = "Transformation to use: 'sqrt' or 'ilr'. [default= %default]"
  ),

  make_option(
    c("--basis"), type = "character",
    default = "theory",
    help = "For ILR transform only: 'theory' or 'empirical'. [default= %default]"
  ),

  # --- Model Settings ---
  make_option(
    c("-m", "--mode"), type = "character",
    default = "multivariate",
    help = "Outcome mode: 'univariate' (enna only) or 'multivariate' (all 4). [default= %default]"
  ),

  make_option(
    c("-l", "--link"), type = "character",
    default = "linear",
    help = "Link function (e.g. 'linear', '5-equi-splines'). [default= %default]"
  ),

  make_option(
    c("-r", "--random"), type = "character",
    default = "time",
    help = "Random effect: '1' (Intercept only) or 'time' (Intercept + Slope). [default= %default]"
  ),

  make_option(
    "--mixture", type = "character",
    default = "time",
    help = "Mixture: 'time', 'time_quad' (time + I(time^2)). [default= %default]"
  ),

  # --- Formula Controls ---
  make_option(
    c("-c", "--controls"), type = "character",
    default = "all",
    help = "Controls to include: 'none', 'scenario', 'experiment', or 'all' [default= %default]"
  ),

  make_option(
    c("-q", "--quadratic"), type = "logical",
    default = FALSE, action = "store_true",
    help = "Include quadratic time term I(time^2)? [default= %default]"
  ),

  # --- Clustering Settings ---
  make_option(
    c("-k", "--nclass"), type = "integer",
    default = 2,
    help = "Number of latent classes to fit. [default= %default]"
  ),

  make_option(
    c("--rep"), type = "integer",
    default = 30,
    help = "Number of gridsearch repetitions. [default= %default]"
  ),

  make_option(
    c("--maxiter"), type = "integer",
    default = 200,
    help = "Maximum iterations for optimization. [default= %default]"
  ),
  make_option(
    c("--use_baseline"), type = "logical",
    default = FALSE, action = "store_true",
    help = "Use the 1-class model for initialization? [default= %default]")
)

opt_parser <- OptionParser(option_list = option_list)
opt <- parse_args(opt_parser)
