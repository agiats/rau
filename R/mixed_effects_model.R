# Required libraries
library(lme4)
library(lmerTest)  # For p-values in mixed models
library(tidyverse)

# Read and prepare data
prepare_data <- function(file_path) {
  # Read CSV file
  data <- read.csv(file_path)

  # Z-score standardization function
  z_score <- function(x) {
    (x - mean(x, na.rm = TRUE)) / sd(x, na.rm = TRUE)
  }

  # Create standardized variables
  data <- data %>%
    mutate(
      Q_z = z_score(n_states),
      Sigma_z = z_score(N_sym),
      ExpLen_z = z_score(mean_length),
      HA_z = z_score(global_entropy),
      local2_z = z_score(`2_local_entropy`),
      local3_z = z_score(`3_local_entropy`),
      local4_z = z_score(`4_local_entropy`)
    )

  # Extract grammar name and seed
  data <- data %>%
    mutate(
      grammar_name = sub("_s\\d+$", "", grammar_name),
      seed = as.numeric(sub(".*_s(\\d+)$", "\\1", grammar_name))
    )

  return(data)
}

# Fit mixed effects model
fit_mixed_model <- function(data) {
  model <- lmer(KL_divergence ~ Q_z + Sigma_z + ExpLen_z + HA_z +
                local2_z + local3_z + local4_z +
                (1 | grammar_name) + (1 | seed),
                data = data)

  return(model)
}

# Main analysis
main <- function() {
  # Read data
  data <- prepare_data("path/to/your/data.csv")

  # Fit model
  model <- fit_mixed_model(data)

  # Print summary
  print(summary(model))

  # Print random effects
  print("Random Effects:")
  print(VarCorr(model))

  # Calculate R-squared
  r2 <- r.squaredGLMM(model)
  print("R-squared:")
  print(r2)
}

# Run analysis
main()
