# Complete sleepstudy example.
# The decomposition runs immediately. Set SLOPEAUDIT_RUN_BOOTSTRAP=true to run
# the validated bootstrap; optionally set SLOPEAUDIT_EXAMPLE_B for a smoke test.

library(lme4)
library(slopeaudit)

control <- lmerControl(
  optimizer = "nloptwrap",
  optCtrl = list(maxeval = 200000),
  check.conv.singular = "message"
)

full <- lmer(
  Reaction ~ Days + (1 + Days | Subject),
  data = sleepstudy,
  REML = TRUE,
  control = control
)
null <- lmer(
  Reaction ~ Days + (1 | Subject),
  data = sleepstudy,
  REML = TRUE,
  control = control
)

days_target <- list(
  basis_type = "linear",
  outcome_units = "milliseconds",
  time_or_exposure_scale = "1 day",
  origin_or_reference = "Day 0",
  reference_window = "Days 0--9",
  estimand = "participant-specific linear change in Reaction per day"
)

sources <- slope_decomposition(full, "Days", days_target)
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
    target = "Days",
    target_metadata = days_target,
    B = B,
    seed = 20260803L,
    progress = interactive()
  )
  print(audit)
  print(summary(audit))
}
