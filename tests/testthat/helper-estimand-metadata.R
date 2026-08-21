sleepstudy_estimand_metadata <- function() {
  list(
    basis_type = "linear",
    outcome_units = "milliseconds",
    time_or_exposure_scale = "1 day",
    origin_or_reference = "Day 0",
    reference_window = "Days 0--9",
    estimand = "participant-specific linear change in Reaction per day",
    basis_expression = "Days"
  )
}
