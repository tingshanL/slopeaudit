q1_sleepstudy_control <- function() {
  lme4::lmerControl(
    optimizer = "nloptwrap",
    check.conv.singular = "message"
  )
}


q1_sleepstudy_pair <- function(REML = TRUE) {
  ctrl <- q1_sleepstudy_control()

  full <- lme4::lmer(
    Reaction ~ Days + (1 + Days | Subject),
    data = lme4::sleepstudy,
    REML = REML,
    control = ctrl
  )

  null <- lme4::lmer(
    Reaction ~ Days + (1 | Subject),
    data = lme4::sleepstudy,
    REML = REML,
    control = ctrl
  )

  list(full = full, null = null, control = ctrl)
}
