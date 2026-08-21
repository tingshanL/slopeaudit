#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(slopeaudit)
})

B <- as.integer(Sys.getenv("SLOPEAUDIT_Q1_B", "999"))
SEED <- as.integer(Sys.getenv("SLOPEAUDIT_Q1_SEED", "20260803"))
FIXTURE <- Sys.getenv(
  "SLOPEAUDIT_BTHEB_FIXTURE",
  "validation/reference/btheb_primary/analysis_data.csv"
)
OUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_OUT",
  "validation/extended/results/q1_btheb"
)

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

dat <- read.csv(FIXTURE, stringsAsFactors = FALSE)

needed <- c(
  "subject",
  "bdi",
  "month",
  "month_c",
  "occasion",
  "treatment"
)

missing <- setdiff(needed, names(dat))
if (length(missing)) {
  stop(
    "Beat-the-Blues fixture is missing: ",
    paste(missing, collapse = ", ")
  )
}

dat$subject <- factor(dat$subject)
dat$occasion <- factor(
  as.character(dat$occasion),
  levels = c("0", "2", "3", "5", "8")
)
dat$treatment <- factor(
  dat$treatment,
  levels = c("TAU", "BtheB")
)

ctrl <- lmerControl(
  optimizer = "nloptwrap",
  optCtrl = list(maxeval = 200000),
  check.conv.singular = "message"
)

# treatment * occasion spans the same saturated treatment-by-occasion mean
# space in both full and null fits. The LR comparison uses the identical X
# basis on both sides.
full <- lmer(
  bdi ~ treatment * occasion + (1 + month_c | subject),
  data = dat,
  REML = TRUE,
  control = ctrl
)

null <- lmer(
  bdi ~ treatment * occasion + (1 | subject),
  data = dat,
  REML = TRUE,
  control = ctrl
)

obs <- slopeaudit:::.sa_q1_observed_lr(
  full,
  null,
  target = "month_c",
  diagnostic_control = ctrl
)

if (!isTRUE(obs$valid_for_inference)) {
  stop("Observed Beat-the-Blues Question-1 pair is not valid.")
}

historical_LR <- 4.072
if (abs(obs$LR_raw - historical_LR) > 0.10) {
  warning(
    sprintf(
      paste0(
        "Observed R/lme4 LR %.6f differs from historical Python anchor %.3f. ",
        "Do not freeze a manuscript value until this discrepancy is adjudicated."
      ),
      obs$LR_raw,
      historical_LR
    )
  )
}

ans <- slopeaudit:::.sa_q1_bootstrap(
  full,
  null,
  target = "month_c",
  refit_control = ctrl,
  B = B,
  seed = SEED,
  retain_replicates = TRUE,
  progress = TRUE
)

saveRDS(ans, file.path(OUT, "q1_btheb_result.rds"))
write.csv(
  ans$replicates,
  file.path(OUT, "q1_btheb_replicates.csv"),
  row.names = FALSE
)

summary_row <- data.frame(
  observed_LR = ans$observed$LR_raw,
  historical_observed_LR = historical_LR,
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
  file.path(OUT, "q1_btheb_summary.csv"),
  row.names = FALSE
)

print(summary_row)
