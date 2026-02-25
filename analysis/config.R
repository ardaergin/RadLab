# analysis/config.R
library(optparse)

get_option_list <- function() {
  list(
    # Input / Output
    make_option(
      c("-f", "--file"), type = "character",
      default = "experiments_processed.rds",
      help = "
          Path to input data file.
          [default= %default]"
    ),

    # Transform
    make_option(
      c("-t", "--transform"), type = "character",
      default = "hellinger",
      help = "
          CoDA transformation to use. Options: 
          - 'hellinger'
          - 'ilr'. 
          [default= %default]"
    ),

    make_option(
      c("--basis"), type = "character",
      default = "A",
      help = "
          The basis matrix to use (for ILR transform only).
          I.e., the name that follows 'ilr_' (e.g., 'A', 'B'). 
          [default= %default]"
    ),

    # Model Settings
    make_option(
      c("-o", "--outcomes"), type = "character",
      default = "all",
      help = "
          Comma-separated outcomes to include. 
          For hellinger: 'ina,na,nna,enna' or 'all'. 
          For ilr: '1,2,3' or 'all'. 
          [default= %default]"
    ),

    make_option(
      c("-l", "--link"), type = "character",
      default = "linear",
      help = "
          Link function to use in lcmm::multlcmm 
          E.g. 'linear', '5-equi-splines', etc. 
          [default= %default]"
    ),

    make_option(
      c("--random"), type = "character",
      default = "2",
      help = "
          Random effect setting. Options: 
          - '1' (intercept only), 
          - '2' (intercept + slope)
          - '3' (time + I(time^2))
          [default= %default]"
    ),

    make_option(
      c("--mixture"), type = "character",
      default = "2",
      help = "
          Mixture setting. Options: 
          - '1' (intercept only)
          - '2' (intercept + slope)
          - '3' (time + I(time^2))
          [default= %default]"
    ),

    # Formula Controls
    make_option(
      c("-c", "--controls"), type = "character",
      default = "all",
      help = "
          Controls to include. Options: 
          - 'none'
          - 'scenario'
          - 'experiment'
          - 'all' 
          [default= %default]"
    ),

    make_option(
      c("-q", "--quadratic"), type = "logical",
      default = FALSE, action = "store_true",
      help = "
          Boolean flag to include quadratic time term I(time^2).
          [default= %default]"
    ),

    # Clustering Settings
    make_option(
      c("-k", "--nclass"), type = "integer",
      default = 1,
      help = "
          Number of latent classes to fit. 
          [default= %default]"
    ),

    make_option(
      c("--rep"), type = "integer",
      default = 30,
      help = "
          Number of gridsearch repetitions. 
          [default= %default]"
    ),

    make_option(
      c("--maxiter"), type = "integer",
      default = 200,
      help = "
          Maximum iterations for optimization. 
          [default= %default]"
    )
  )
}
