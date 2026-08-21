#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(slopeaudit)
})

B <- as.integer(Sys.getenv("SLOPEAUDIT_Q1_B", "999"))
SEED <- as.integer(Sys.getenv("SLOPEAUDIT_Q1_SEED", "20260804"))
OUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_OUT",
  "validation/extended/results/q1_sleepstudy"
)

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

ctrl <- lmerControl(
  optimizer = "nloptwrap",
  check.conv.singular = "message"
)

full <- lmer(
  Reaction ~ Days + (1 + Days | Subject),
  data = sleepstudy,
  REML = TRUE,
  control = ctrl
)

null <- lmer(
  Reaction ~ Days + (1 | Subject),
  data = sleepstudy,
  REML = TRUE,
  control = ctrl
)

obs <- slopeaudit:::.sa_q1_observed_lr(
  full,
  null,
  target = "Days",
  diagnostic_control = ctrl
)

if (!isTRUE(obs$valid_for_inference)) {
  stop("Observed sleepstudy Question-1 pair is not valid for inference.")
}

historical_LR <- 42.837
if (abs(obs$LR_raw - historical_LR) > 0.05) {
  stop(
    sprintf(
      "Observed LR %.6f is unexpectedly far from historical anchor %.3f.",
      obs$LR_raw,
      historical_LR
    )
  )
}

ans <- slopeaudit:::.sa_q1_bootstrap(
  full,
  null,
  target = "Days",
  refit_control = ctrl,
  B = B,
  seed = SEED,
  retain_replicates = TRUE,
  progress = TRUE
)

saveRDS(ans, file.path(OUT, "q1_sleepstudy_result.rds"))
write.csv(
  ans$replicates,
  file.path(OUT, "q1_sleepstudy_replicates.csv"),
  row.names = FALSE
)

summary_row <- data.frame(
  observed_LR = ans$observed$LR_raw,
  B_requested = ans$bootstrap$B_requested,
  denominator_drop = ans$bootstrap$denominator_drop,
  denominator_zero = ans$bootstrap$denominator_zero,
  usable_rate_drop = ans$bootstrap$usable_rate_drop,
  usable_rate_zero = ans$bootstrap$usable_rate_zero,
  exceedances_drop = ans$bootstrap$exceedances_drop,
  exceedances_zero = ans$bootstrap$exceedances_zero,
  p_drop = ans$bootstrap$p_drop,
  p_zero = ans$bootstrap$p_zero,
  attainable_floor_drop = ans$bootstrap$attainable_floor_drop,
  attainable_floor_zero = ans$bootstrap$attainable_floor_zero,
  mc_drop_low = ans$bootstrap$mc_interval_drop[[1]],
  mc_drop_high = ans$bootstrap$mc_interval_drop[[2]],
  small_negative = ans$bootstrap$n_small_negative,
  material_negative = ans$bootstrap$n_material_negative,
  failures = ans$bootstrap$n_failure,
  escalation_recommended = ans$bootstrap$escalation$recommended,
  escalation_reason = ans$bootstrap$escalation$reason,
  stringsAsFactors = FALSE
)

write.csv(
  summary_row,
  file.path(OUT, "q1_sleepstudy_summary.csv"),
  row.names = FALSE
)

print(summary_row)
