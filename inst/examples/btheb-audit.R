# Complete irregular-schedule example.
# The decomposition runs immediately. Set SLOPEAUDIT_RUN_BOOTSTRAP=true to run
# the bootstrap; the manuscript workflow escalates an ambiguous B=999 run to
# B=4999 explicitly rather than changing B silently.

library(lme4)
library(slopeaudit)

path <- system.file("extdata", "btheb_primary.csv", package = "slopeaudit")
btheb <- read.csv(path, stringsAsFactors = FALSE)
btheb$subject <- factor(btheb$subject)
btheb$occasion <- factor(
  btheb$occasion,
  levels = c("0", "2", "3", "5", "8")
)
btheb$treatment <- factor(btheb$treatment, levels = c("TAU", "BtheB"))

control <- lmerControl(
  optimizer = "nloptwrap",
  optCtrl = list(maxeval = 200000),
  check.conv.singular = "message"
)

full <- lmer(
  bdi ~ treatment * occasion + (1 + month_c | subject),
  data = btheb,
  REML = TRUE,
  control = control
)
null <- lmer(
  bdi ~ treatment * occasion + (1 | subject),
  data = btheb,
  REML = TRUE,
  control = control
)

month_target <- list(
  basis_type = "linear",
  outcome_units = "BDI score points",
  time_or_exposure_scale = "1 month",
  origin_or_reference = "Month 4 (month_c = 0)",
  reference_window = "Months 0--8",
  estimand = "participant-specific linear change in BDI score per month"
)

sources <- slope_decomposition(full, "month_c", month_target)
print(sources)

run_bootstrap <- identical(
  tolower(Sys.getenv("SLOPEAUDIT_RUN_BOOTSTRAP", "false")),
  "true"
)

if (run_bootstrap) {
  B <- as.integer(Sys.getenv("SLOPEAUDIT_EXAMPLE_B", "999"))
  audit <- slope_audit(
    full = full,
    null = null,
    target = "month_c",
    target_metadata = month_target,
    B = B,
    seed = 20260803L,
    progress = interactive()
  )
  print(audit)
  print(summary(audit))
}
