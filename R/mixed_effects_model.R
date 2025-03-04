# Required libraries
library(lme4)
library(lmerTest)  # For p-values in mixed models
library(tidyverse)

# Read and prepare data
prepare_data <- function(file_path) {
  # Read CSV file
  data <- read.csv(file_path)

  # Calculate KL divergence
  data <- data %>%
    mutate(KL_divergence = cross_entropy_per_token_base_2 - next_symbol_entropy)

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
      NextSym_z = z_score(next_symbol_entropy),
      Local2_z = z_score(`X2_local_entropy`),
      Local3_z = z_score(`X3_local_entropy`),
      Local4_z = z_score(`X4_local_entropy`),
      Local5_z = z_score(`X5_local_entropy`)
    )


  return(data)
}

# Fit and compare models
compare_models <- function(data) {
  # Base model with random intercept only
  base_model <- lmer(KL_divergence ~ Q_z * Sigma_z + NextSym_z + (1|Q_z:Sigma_z),
                   data = data, REML = FALSE)

  # List of local entropy variables to test
  entropy_vars <- c("Local2_z", "Local3_z", "Local4_z", "Local5_z")

  # Store results
  results <- list()
  results[["base"]] <- base_model

  for (entropy_var in entropy_vars) {
    formula <- as.formula(paste("KL_divergence ~ Q_z * Sigma_z + NextSym_z +",
                              entropy_var,
                              "+ (1 | Q_z:Sigma_z)"))  # 同じランダム効果構造

    current_model <- lmer(formula, data = data, REML = FALSE)
    results[[entropy_var]] <- current_model

    cat("\n--- Testing", entropy_var, "---\n")
    print(anova(base_model, current_model))
  }

  # Compare model fits
  cat("\n=== Model Information Criteria ===\n")
  model_fits <- sapply(results, function(m) {
    c(AIC = AIC(m),
      BIC = BIC(m),
      logLik = as.numeric(logLik(m)),
      df = attr(logLik(m), "df"))
  })
  print(round(model_fits, 2))

  # Print summary of coefficients for all models with significance stars
  cat("\n=== Model Coefficients ===\n")
  for (name in names(results)) {
    cat("\n---", name, "---\n")
    coef_summary <- summary(results[[name]])

    # mixed-effects modelの場合の処理
    if (inherits(results[[name]], "lmerMod")) {
      # lmerTestのsummaryからp値を取得
      p_values <- coef_summary$coefficients[,"Pr(>|t|)"]
      # 有意性を示す記号を追加
      signif <- symnum(p_values,
                      corr = FALSE,
                      na = FALSE,
                      cutpoints = c(0, 0.001, 0.01, 0.05, 0.1, 1),
                      symbols = c("***", "**", "*", ".", " "))
      # 係数と標準誤差を整形
      coef_table <- cbind(
        "Estimate" = sprintf("%.3e", coef_summary$coefficients[,"Estimate"]),
        "Std.Error" = sprintf("%.3e", coef_summary$coefficients[,"Std. Error"]),
        "t value" = sprintf("%.3f", coef_summary$coefficients[,"t value"]),
        "Pr(>|t|)" = sprintf("%.3e", p_values),
        "Signif." = signif
      )
      print(coef_table, quote = FALSE)
    }
  }

  cat("\nSignif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1\n")

  return(results)
}

# Main analysis
main <- function(file_path) {
  # Read and prepare data
  data <- prepare_data(file_path)

  # Perform model comparisons
  models <- compare_models(data)

  # Return models for further analysis if needed
  return(models)
}

# Run analysis
models <- main("/Users/agiats/Projects/lm_inductive_bias/results/PFSA/collected_results_test.csv")
