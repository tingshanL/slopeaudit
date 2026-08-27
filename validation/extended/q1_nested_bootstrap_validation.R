#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(slopeaudit)
})

read_integer_env <- function(name, default, minimum = 1L) {
  raw <- Sys.getenv(name, unset = as.character(default))
  value <- suppressWarnings(as.numeric(raw))
  if (length(value) != 1L || !is.finite(value) ||
      value != floor(value) || value < minimum ||
      value > .Machine$integer.max) {
    stop(sprintf("%s must be an integer >= %d.", name, minimum))
  }
  as.integer(value)
}

read_numeric_env <- function(name, default) {
  raw <- Sys.getenv(name, unset = as.character(default))
  value <- suppressWarnings(as.numeric(raw))
  if (length(value) != 1L || !is.finite(value)) {
    stop(sprintf("%s must be one finite number.", name))
  }
  value
}

MODE <- tolower(Sys.getenv("SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_MODE", "smoke"))
mode_defaults <- list(
  smoke = list(outer = 2L, usable_B = 39L, workers = 1L),
  pilot = list(outer = 20L, usable_B = 99L, workers = 1L),
  full = list(outer = 999L, usable_B = 999L, workers = 1L)
)
if (!MODE %in% names(mode_defaults)) {
  stop("SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_MODE must be smoke, pilot, or full.")
}

defaults <- mode_defaults[[MODE]]
OUTER_N <- read_integer_env(
  "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_OUTER",
  defaults$outer
)
USABLE_B <- read_integer_env(
  "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_B_USABLE",
  defaults$usable_B
)
ATTEMPT_CAP <- read_integer_env(
  "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_ATTEMPT_CAP",
  USABLE_B + max(50L, ceiling(0.25 * USABLE_B)),
  minimum = USABLE_B
)
WORKERS <- read_integer_env(
  "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_WORKERS",
  defaults$workers
)
CHECKPOINT_BATCH <- read_integer_env(
  "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_CHECKPOINT_BATCH",
  max(1L, WORKERS)
)
MASTER_SEED <- read_integer_env(
  "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_SEED",
  20260812L,
  minimum = 0L
)
ALPHA <- read_numeric_env("SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_ALPHA", 0.05)
if (ALPHA <= 0 || ALPHA >= 1) {
  stop("SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_ALPHA must lie strictly between 0 and 1.")
}
MAX_OBSERVED_UNRESOLVED_RATE <- read_numeric_env(
  "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_MAX_OBSERVED_UNRESOLVED_RATE",
  0.005
)
if (MAX_OBSERVED_UNRESOLVED_RATE < 0 ||
    MAX_OBSERVED_UNRESOLVED_RATE >= 1) {
  stop(
    paste0(
      "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_MAX_OBSERVED_UNRESOLVED_RATE must lie ",
      "in [0, 1)."
    )
  )
}

rank_grid <- ALPHA * (USABLE_B + 1)
if (abs(rank_grid - round(rank_grid)) > 100 * .Machine$double.eps) {
  stop(
    paste0(
      "The primary Monte Carlo rank grid is not exact at this alpha: ",
      "alpha * (B_usable + 1) = ", format(rank_grid, digits = 16), "."
    )
  )
}

if (WORKERS > 1L && .Platform$OS.type == "windows") {
  stop("Parallel outer execution currently requires macOS/Linux.")
}
if (WORKERS > 2L) {
  warning("More than two outer workers are not recommended on the 8-GB M2.")
}

OUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_OUT",
  file.path(
    "validation",
    "extended",
    "results",
    paste0("q1_nested_bootstrap_", MODE, "_v14")
  )
)
FROZEN_PILOT_INPUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_V12_PILOT_INPUT",
  unset = ""
)
TARGET_FIXTURE <- file.path(
  "validation",
  "reference",
  "q1_nested_bootstrap_v14_targets",
  "q1_nested_bootstrap_targets.csv"
)
V11_EVIDENCE_DIR <- Sys.getenv(
  "SLOPEAUDIT_Q1_V11_EVIDENCE_DIR",
  unset = ""
)
FIXTURE_PATH <- Sys.getenv(
  "SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE",
  unset = ""
)

if (!nzchar(V11_EVIDENCE_DIR)) {
  stop(
    "Nested-bootstrap validation requires private v11 adjudication evidence. ",
    "Set SLOPEAUDIT_Q1_V11_EVIDENCE_DIR explicitly.",
    call. = FALSE
  )
}
if (MODE == "pilot" && !nzchar(FROZEN_PILOT_INPUT)) {
  stop(
    "pilot mode requires the private audited v12 pilot checkpoint directory. ",
    "Set SLOPEAUDIT_Q1_NESTED_BOOTSTRAP_V12_PILOT_INPUT explicitly.",
    call. = FALSE
  )
}
if (!nzchar(FIXTURE_PATH)) {
  stop(
    "Nested-bootstrap validation requires the private exact irregular schedule. ",
    "Set SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE explicitly.",
    call. = FALSE
  )
}

if (!file.exists(FIXTURE_PATH) || !file.exists(TARGET_FIXTURE)) {
  stop(
    paste0(
      "Missing nested-bootstrap validation fixture. Required paths:\n- ", FIXTURE_PATH,
      "\n- ", TARGET_FIXTURE
    ),
    call. = FALSE
  )
}

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

irregular <- utils::read.csv(
  FIXTURE_PATH,
  stringsAsFactors = FALSE
)
required_fixture_columns <- c("schedule_id", "measurement", "t5_mid")
if (!identical(names(irregular), required_fixture_columns)) {
  stop(
    paste0(
      "Irregular fixture columns must be exactly: ",
      paste(required_fixture_columns, collapse = ", ")
    )
  )
}
if (nrow(irregular) != 1805L ||
    length(unique(irregular$schedule_id)) != 691L ||
    any(!is.finite(irregular$t5_mid))) {
  stop("Irregular fixture must contain 691 schedules and 1,805 finite rows.")
}
if (any(table(irregular$schedule_id) < 2L)) {
  stop("Every irregular schedule must contain at least two observations.")
}

calibration_controls <- slopeaudit:::.sa_q1_calibration_rescue_controls()
initial_control <- calibration_controls$initial
rescue_control <- calibration_controls$rescue

v11_evidence_paths <- c(
  method_summary = file.path(
    V11_EVIDENCE_DIR,
    "q1_optimizer_rescue_method_summary.csv"
  ),
  operational_gate = file.path(
    V11_EVIDENCE_DIR,
    "q1_optimizer_rescue_operational_gate.csv"
  ),
  task_summary = file.path(
    V11_EVIDENCE_DIR,
    "q1_optimizer_rescue_task_summary.csv"
  )
)
if (any(!file.exists(v11_evidence_paths))) {
  stop("Missing frozen v11 optimizer-rescue adjudication evidence.")
}
v11_method_summary <- utils::read.csv(
  v11_evidence_paths[["method_summary"]],
  stringsAsFactors = FALSE,
  check.names = FALSE
)
v11_operational_gate <- utils::read.csv(
  v11_evidence_paths[["operational_gate"]],
  stringsAsFactors = FALSE,
  check.names = FALSE
)
v11_task_summary <- utils::read.csv(
  v11_evidence_paths[["task_summary"]],
  stringsAsFactors = FALSE,
  check.names = FALSE
)
v11_all <- v11_method_summary[
  v11_method_summary$original_diagnostic_class == "ALL",
  ,
  drop = FALSE
]
v11_current <- v11_all[v11_all$method_id == "current_nloptwrap", ]
v11_tight <- v11_all[v11_all$method_id == "tight_nloptwrap", ]
v11_bobyqa <- v11_all[v11_all$method_id == "bobyqa", ]
v11_resolution_counts <- table(v11_task_summary$resolution_pattern)
v11_disagreement <- v11_task_summary[
  v11_task_summary$resolution_pattern ==
    "both_candidates_primary_usable_but_disagree",
  ,
  drop = FALSE
]
v11_evidence_ok <- nrow(v11_current) == 1L &&
  nrow(v11_tight) == 1L && nrow(v11_bobyqa) == 1L &&
  v11_current$n == 908L && v11_current$failures == 866L &&
  v11_current$small_negative == 14L &&
  v11_current$material_negative == 28L &&
  v11_tight$primary_usable == 274L &&
  v11_bobyqa$primary_usable == 908L &&
  v11_bobyqa$failures == 0L &&
  v11_bobyqa$small_negative == 0L &&
  v11_bobyqa$material_negative == 0L &&
  nrow(v11_operational_gate) == 7L &&
  all(v11_operational_gate$pass) &&
  nrow(v11_task_summary) == 908L &&
  unname(v11_resolution_counts[["bobyqa_only"]]) == 634L &&
  unname(v11_resolution_counts[[
    "both_candidates_primary_usable_and_agree"
  ]]) == 219L &&
  unname(v11_resolution_counts[[
    "both_candidates_primary_usable_but_disagree"
  ]]) == 55L &&
  nrow(v11_disagreement) == 55L &&
  all(
    v11_disagreement$bobyqa_full_logLik >=
      v11_disagreement$tight_full_logLik - 1e-10
  ) &&
  max(abs(
    v11_disagreement$bobyqa_null_logLik -
      v11_disagreement$tight_null_logLik
  )) <= 1e-8
if (!isTRUE(v11_evidence_ok)) {
  stop("Frozen v11 evidence does not reproduce its adjudicated anchors.")
}

target_fixture <- utils::read.csv(
  TARGET_FIXTURE,
  stringsAsFactors = FALSE,
  check.names = FALSE,
  na.strings = "NA"
)
required_target_columns <- c(
  "cell_id", "reference_source", "reference_rate",
  "reference_interval_low", "reference_interval_high", "reference_n",
  "reference_supports", "interpretation"
)
if (!identical(names(target_fixture), required_target_columns) ||
    nrow(target_fixture) != 8L || anyDuplicated(target_fixture$cell_id) ||
    any(!is.finite(target_fixture$reference_rate))) {
  stop("The frozen nested-bootstrap validation target fixture failed its schema checks.")
}

cell_manifest <- data.frame(
  cell_id = c(
    "null_balanced_q0_0p5",
    "null_balanced_q0_4",
    "null_irregular_q0_0p5",
    "null_irregular_q0_4",
    "power_three_waves_two_years_rho0",
    "power_three_waves_two_years_rho0p5",
    "power_five_waves_five_years_rho0",
    "power_five_waves_five_years_rho0p5"
  ),
  cell_family = c(
    rep("null_size", 4L),
    rep("planning_power_validation", 4L)
  ),
  truth_role = c(
    rep("null", 4L),
    "alternative", "paired_sensitivity",
    "alternative", "paired_sensitivity"
  ),
  schedule = c(
    "balanced_three_wave",
    "balanced_three_wave",
    "irregular_fluid_cognition",
    "irregular_fluid_cognition",
    "three_waves_two_years",
    "three_waves_two_years",
    "five_waves_five_years",
    "five_waves_five_years"
  ),
  q0 = c(0.5, 4, 0.5, 4, rep(0.5, 4L)),
  slope_variance = c(rep(0, 4L), rep(0.01, 4L)),
  intercept_slope_correlation = c(rep(0, 5L), 0.5, 0, 0.5),
  residual_variance = rep(1, 8L),
  fixed_intercept = rep(0, 8L),
  fixed_slope = rep(0.10, 8L),
  pair_role = c(rep("none", 4L), "rho0", "rho0p5", "rho0", "rho0p5"),
  pair_base_cell = c(
    rep("", 4L),
    "power_three_waves_two_years_rho0",
    "power_three_waves_two_years_rho0",
    "power_five_waves_five_years_rho0",
    "power_five_waves_five_years_rho0"
  ),
  historical_k = c(47L, 48L, 41L, 45L, rep(NA_integer_, 4L)),
  historical_n = c(999L, 997L, 998L, 999L, rep(NA_integer_, 4L)),
  stringsAsFactors = FALSE
)
if (!identical(cell_manifest$cell_id, target_fixture$cell_id)) {
  stop("nested-bootstrap validation cell order differs from the frozen target fixture.")
}
cell_manifest <- cbind(
  cell_manifest,
  target_fixture[, setdiff(names(target_fixture), "cell_id"), drop = FALSE]
)

make_balanced_schedule <- function(n_id, times) {
  out <- do.call(
    rbind,
    lapply(seq_len(n_id), function(id) {
      data.frame(id = id, time = times)
    })
  )
  out$id <- factor(out$id)
  rownames(out) <- NULL
  out
}

schedules <- list(
  balanced_three_wave = make_balanced_schedule(
    600L,
    c(-0.5, 0, 0.5)
  ),
  irregular_fluid_cognition = transform(
    irregular,
    id = factor(schedule_id),
    time = t5_mid
  )[, c("id", "time")],
  three_waves_two_years = make_balanced_schedule(
    100L,
    c(0, 1, 2) - mean(c(0, 1, 2))
  ),
  five_waves_five_years = make_balanced_schedule(
    100L,
    seq(0, 5, length.out = 5) - mean(seq(0, 5, length.out = 5))
  )
)

analytic_planning_values <- function(cell) {
  if (cell$cell_family != "planning_power_validation") {
    return(c(R_direct = NA_real_, power = NA_real_))
  }
  schedule <- schedules[[cell$schedule]]
  one <- schedule[schedule$id == levels(schedule$id)[[1]], "time"]
  information <- sum((one - mean(one))^2) / cell$residual_variance
  direct_error_variance <- 1 / information
  r_direct <- cell$slope_variance /
    (cell$slope_variance + direct_error_variance)
  n_id <- nlevels(schedule$id)
  df <- n_id - 1L
  critical <- stats::qchisq(1 - ALPHA, df = df)
  power <- 1 - stats::pchisq(
    critical * (1 - r_direct),
    df = df
  )
  c(R_direct = r_direct, power = power)
}

planning <- t(vapply(
  seq_len(nrow(cell_manifest)),
  function(index) analytic_planning_values(cell_manifest[index, ]),
  numeric(2)
))
cell_manifest$analytic_R_direct <- planning[, "R_direct"]
cell_manifest$analytic_power <- planning[, "power"]
cell_manifest$n_participants <- vapply(
  cell_manifest$schedule,
  function(name) nlevels(schedules[[name]]$id),
  integer(1)
)
cell_manifest$n_observations <- vapply(
  cell_manifest$schedule,
  function(name) nrow(schedules[[name]]),
  integer(1)
)

# The `version` value below is the historical v14 configuration ID used by
# completed checkpoints.
scientific_config <- list(
  version = "q1_nested_bootstrap_nested_bootstrap_v14_candidate",
  mode = MODE,
  outer_per_cell = OUTER_N,
  usable_B = USABLE_B,
  attempt_cap = ATTEMPT_CAP,
  alpha = ALPHA,
  maximum_observed_unresolved_rate = MAX_OBSERVED_UNRESOLVED_RATE,
  rejection_rule = "plus-one p <= alpha",
  primary_inner_convention = "fixed positive/zero usable denominator",
  unresolved_inner_rule = paste0(
    "unresolved inner attempts trigger deterministic top-up; retain full ",
    "provenance and report worst-case decision bounds"
  ),
  unresolved_observed_rule = paste0(
    "do not impute; retain task as unresolved and include it in cell-level ",
    "support and interval bounds"
  ),
  master_seed = MASTER_SEED,
  irregular_fixture_md5 = unname(tools::md5sum(FIXTURE_PATH)),
  target_fixture_md5 = unname(tools::md5sum(TARGET_FIXTURE)),
  v11_evidence_md5 = unname(tools::md5sum(v11_evidence_paths)),
  production_R_md5 = unname(tools::md5sum(list.files(
    "R", full.names = TRUE, recursive = TRUE
  ))),
  cell_manifest = cell_manifest,
  initial_control = initial_control,
  rescue_control = rescue_control,
  rescue_rule = paste0(
    "initial nloptwrap full/null pair; if not primary-usable, refit both ",
    "with bobyqa; unresolved rescue triggers top-up"
  )
)

config_rds <- file.path(OUT, "q1_nested_bootstrap_configuration.rds")
if (file.exists(config_rds)) {
  prior_config <- readRDS(config_rds)
  if (!identical(prior_config, scientific_config)) {
    stop(
      "Existing checkpoint configuration differs from this invocation."
    )
  }
} else {
  saveRDS(scientific_config, config_rds, version = 3)
}
utils::write.csv(
  cell_manifest,
  file.path(OUT, "q1_nested_bootstrap_cell_manifest.csv"),
  row.names = FALSE
)

generate_outcome <- function(schedule, cell, seed) {
  set.seed(seed)
  n_id <- nlevels(schedule$id)
  z0 <- stats::rnorm(n_id)
  if (cell$slope_variance > 0) {
    z1 <- stats::rnorm(n_id)
    rho <- cell$intercept_slope_correlation
    b1 <- sqrt(cell$slope_variance) * z1
    b0 <- sqrt(cell$q0) * (sqrt(1 - rho^2) * z0 + rho * z1)
  } else {
    if (cell$intercept_slope_correlation != 0) {
      stop("A null cell cannot have nonzero intercept-slope correlation.")
    }
    b0 <- sqrt(cell$q0) * z0
    b1 <- rep(0, n_id)
  }
  id_index <- as.integer(schedule$id)
  schedule$y <- cell$fixed_intercept + cell$fixed_slope * schedule$time +
    b0[id_index] +
    b1[id_index] * schedule$time +
    stats::rnorm(
      nrow(schedule),
      0,
      sqrt(cell$residual_variance)
    )
  schedule
}

direct_slope_statistic <- function(data, residual_variance) {
  by_id <- split(data, data$id, drop = TRUE)
  sxx <- vapply(by_id, function(one) {
    sum((one$time - mean(one$time))^2)
  }, numeric(1))
  if (any(!is.finite(sxx)) || any(sxx <= 0) ||
      max(abs(sxx - sxx[[1]])) > 1e-12 * max(1, abs(sxx))) {
    return(NA_real_)
  }
  slopes <- vapply(by_id, function(one) {
    centered <- one$time - mean(one$time)
    sum(centered * one$y) / sum(centered^2)
  }, numeric(1))
  sum((slopes - mean(slopes))^2) / (residual_variance / sxx[[1]])
}

inner_decision_bounds <- function(p_drop, usable_B, attempts_evaluated) {
  if (!is.finite(p_drop) || !is.finite(usable_B) ||
      !is.finite(attempts_evaluated) || usable_B < 1L ||
      attempts_evaluated < usable_B) {
    return(c(
      exceedances = NA_real_, excluded = NA_real_,
      p_all_nonexceed = NA_real_, p_all_exceed = NA_real_,
      stable = NA
    ))
  }
  exceedances <- round(p_drop * (usable_B + 1) - 1)
  if (abs(p_drop - (exceedances + 1) / (usable_B + 1)) > 1e-10) {
    stop("A primary p-value does not lie on its declared plus-one grid.")
  }
  excluded <- attempts_evaluated - usable_B
  lower <- (exceedances + 1) / (attempts_evaluated + 1)
  upper <- (exceedances + excluded + 1) / (attempts_evaluated + 1)
  c(
    exceedances = exceedances,
    excluded = excluded,
    p_all_nonexceed = lower,
    p_all_exceed = upper,
    stable = (lower <= ALPHA) == (upper <= ALPHA)
  )
}

nested_bootstrap_bound_regression_checks <- local({
  stable_support <- inner_decision_bounds(0.01, 99L, 100L)
  cutoff_ambiguous <- inner_decision_bounds(0.05, 99L, 100L)
  stable_nonsupport <- inner_decision_bounds(0.11, 99L, 100L)
  stopifnot(
    isTRUE(as.logical(stable_support[["stable"]])),
    !isTRUE(as.logical(cutoff_ambiguous[["stable"]])),
    isTRUE(as.logical(stable_nonsupport[["stable"]])),
    cutoff_ambiguous[["p_all_nonexceed"]] <= ALPHA,
    cutoff_ambiguous[["p_all_exceed"]] > ALPHA
  )
  TRUE
})

empty_problem_rows <- function() {
  data.frame(
    task_id = character(),
    cell_id = character(),
    truth_role = character(),
    truth_rho = numeric(),
    pair_id = character(),
    outer = integer(),
    design_seed = integer(),
    bootstrap_seed = integer(),
    replicate = integer(),
    seed = integer(),
    status = character(),
    reason = character(),
    LR_raw = numeric(),
    LR_zero = numeric(),
    tolerance = numeric(),
    logLik_full = numeric(),
    logLik_null = numeric(),
    full_singular = logical(),
    null_singular = logical(),
    warnings = character(),
    messages = character(),
    fit_source = character(),
    rescue_triggered = logical(),
    rescue_succeeded = logical(),
    initial_status = character(),
    initial_reason = character(),
    initial_LR_raw = numeric(),
    initial_tolerance = numeric(),
    initial_logLik_full = numeric(),
    initial_logLik_null = numeric(),
    initial_full_valid = logical(),
    initial_null_valid = logical(),
    initial_full_singular = logical(),
    initial_null_singular = logical(),
    initial_full_optimizer = character(),
    initial_null_optimizer = character(),
    initial_optimizer_honored = logical(),
    initial_warnings = character(),
    initial_messages = character(),
    rescue_status = character(),
    rescue_reason = character(),
    rescue_LR_raw = numeric(),
    rescue_tolerance = numeric(),
    rescue_logLik_full = numeric(),
    rescue_logLik_null = numeric(),
    rescue_full_valid = logical(),
    rescue_null_valid = logical(),
    rescue_full_singular = logical(),
    rescue_null_singular = logical(),
    rescue_full_optimizer = character(),
    rescue_null_optimizer = character(),
    rescue_optimizer_honored = logical(),
    rescue_warnings = character(),
    rescue_messages = character(),
    stringsAsFactors = FALSE
  )
}

observed_diagnostic_fields <- function(observed) {
  value <- function(assessment, name, default = NA) {
    slopeaudit:::.sa_q1_assessment_value(assessment, name, default)
  }
  initial <- if (is.null(observed)) NULL else observed$initial
  rescue <- if (is.null(observed)) NULL else observed$rescue
  data.frame(
    observed_initial_status = value(initial, "status", "not_run"),
    observed_initial_reason = value(initial, "reason", ""),
    observed_initial_LR_raw = value(initial, "LR_raw", NA_real_),
    observed_initial_logLik_full = value(
      initial,
      "logLik_full",
      NA_real_
    ),
    observed_initial_logLik_null = value(
      initial,
      "logLik_null",
      NA_real_
    ),
    observed_initial_full_valid = value(initial, "full_valid", NA),
    observed_initial_null_valid = value(initial, "null_valid", NA),
    observed_initial_full_singular = value(
      initial,
      "full_singular",
      NA
    ),
    observed_initial_null_singular = value(
      initial,
      "null_singular",
      NA
    ),
    observed_initial_full_optimizer = value(
      initial,
      "full_optimizer",
      ""
    ),
    observed_initial_null_optimizer = value(
      initial,
      "null_optimizer",
      ""
    ),
    observed_initial_warnings = value(initial, "warnings", ""),
    observed_initial_messages = value(initial, "messages", ""),
    observed_rescue_status = value(rescue, "status", "not_run"),
    observed_rescue_reason = value(rescue, "reason", ""),
    observed_rescue_LR_raw = value(rescue, "LR_raw", NA_real_),
    observed_rescue_logLik_full = value(rescue, "logLik_full", NA_real_),
    observed_rescue_logLik_null = value(rescue, "logLik_null", NA_real_),
    observed_rescue_full_valid = value(rescue, "full_valid", NA),
    observed_rescue_null_valid = value(rescue, "null_valid", NA),
    observed_rescue_full_singular = value(rescue, "full_singular", NA),
    observed_rescue_null_singular = value(rescue, "null_singular", NA),
    observed_rescue_full_optimizer = value(rescue, "full_optimizer", ""),
    observed_rescue_null_optimizer = value(rescue, "null_optimizer", ""),
    observed_rescue_warnings = value(rescue, "warnings", ""),
    observed_rescue_messages = value(rescue, "messages", ""),
    stringsAsFactors = FALSE
  )
}

observed_failure <- function(
    cell,
    task,
    status,
    reason,
    elapsed_seconds,
    observed = NULL,
    direct_statistic = NA_real_) {
  core <- data.frame(
    task_id = task$task_id,
    cell_id = cell$cell_id,
    cell_family = cell$cell_family,
    truth_role = cell$truth_role,
    truth_rho = cell$intercept_slope_correlation,
    pair_id = task$pair_id,
    outer = task$outer,
    design_seed = task$design_seed,
    bootstrap_seed = task$bootstrap_seed,
    status = status,
    failure_reason = reason,
    direct_statistic = direct_statistic,
    observed_LR = NA_real_,
    observed_fit_source = if (is.null(observed)) "" else observed$source,
    observed_rescue_triggered = if (is.null(observed)) NA else
      observed$rescue_triggered,
    observed_rescue_succeeded = if (is.null(observed)) NA else
      observed$rescue_succeeded,
    full_singular = NA,
    lambda_min_over_trace = NA_real_,
    target_max_abs_correlation = NA_real_,
    usable_target = USABLE_B,
    attempt_cap = ATTEMPT_CAP,
    attempts_evaluated = NA_integer_,
    top_up_attempts = NA_integer_,
    denominator_drop = NA_integer_,
    denominator_zero = NA_integer_,
    denominator_all_zero = NA_integer_,
    p_drop = NA_real_,
    p_zero = NA_real_,
    p_all_zero = NA_real_,
    usable_exceedances = NA_integer_,
    excluded_inner_attempts = NA_integer_,
    p_if_excluded_all_nonexceed = NA_real_,
    p_if_excluded_all_exceed = NA_real_,
    inner_missingness_decision_stable = NA,
    reject_drop = NA,
    reject_zero = NA,
    reject_all_zero = NA,
    inner_failures = NA_integer_,
    small_negative = NA_integer_,
    material_negative = NA_integer_,
    inner_rescue_attempts = NA_integer_,
    inner_rescue_successes = NA_integer_,
    unresolved_after_rescue = NA_integer_,
    elapsed_seconds = elapsed_seconds,
    stringsAsFactors = FALSE
  )
  cbind(core, observed_diagnostic_fields(observed))
}

run_one <- function(task_index) {
  task <- task_manifest[task_index, ]
  cell <- cell_manifest[
    cell_manifest$cell_id == task$cell_id,
    ,
    drop = FALSE
  ]
  started <- proc.time()[["elapsed"]]
  dat <- generate_outcome(
    schedules[[cell$schedule]],
    cell,
    task$design_seed
  )
  direct_statistic <- if (cell$cell_family == "planning_power_validation") {
    direct_slope_statistic(dat, cell$residual_variance)
  } else {
    NA_real_
  }

  observed <- slopeaudit:::.sa_q1_fit_observed_pair_rescue(
    full_formula = y ~ time + (1 + time | id),
    null_formula = y ~ time + (1 | id),
    data = dat,
    target = "time",
    REML = TRUE,
    initial_control = initial_control,
    rescue_control = rescue_control
  )

  if (!identical(observed$status, "ok")) {
    return(list(
      summary = observed_failure(
        cell = cell,
        task = task,
        status = "observed_unresolved_after_rescue",
        reason = paste0(
          "initial=", observed$initial$status, ":", observed$initial$reason,
          ";rescue=", observed$rescue$status, ":", observed$rescue$reason
        ),
        elapsed_seconds = proc.time()[["elapsed"]] - started,
        observed = observed,
        direct_statistic = direct_statistic
      ),
      problems = empty_problem_rows()
    ))
  }

  full <- observed$full_fit
  covariance <- as.matrix(VarCorr(full)[[1]])
  geometry <- slopeaudit:::.sa_covariance_diagnostics(
    covariance,
    target_index = 2L
  )

  answer <- tryCatch(
    slopeaudit:::.sa_q1_bootstrap_fixed_usable_rescue(
      observed = observed,
      target = "time",
      usable_B = USABLE_B,
      attempt_cap = ATTEMPT_CAP,
      seed = task$bootstrap_seed,
      alpha = ALPHA,
      retain_replicates = TRUE,
      progress = FALSE,
      initial_control = initial_control,
      rescue_control = rescue_control
    ),
    error = function(error) error
  )

  if (inherits(answer, "error") || answer$status != "ok") {
    reason <- if (inherits(answer, "error")) {
      conditionMessage(answer)
    } else {
      paste0("status:", answer$status)
    }
    failure_status <- if (inherits(answer, "error")) {
      "q1_error"
    } else if (answer$status == "invalid_observed") {
      "q1_invalid_observed"
    } else {
      "attempt_cap_reached"
    }
    failure_summary <- observed_failure(
      cell = cell,
      task = task,
      status = failure_status,
      reason = reason,
      elapsed_seconds = proc.time()[["elapsed"]] - started,
      observed = observed,
      direct_statistic = direct_statistic
    )
    if (
      !inherits(answer, "error") &&
        answer$status == "attempt_cap_reached"
    ) {
      partial <- answer$bootstrap
      failure_summary$observed_LR <- answer$observed$LR_for_inference
      failure_summary$full_singular <-
        answer$observed$chosen$full_singular
      failure_summary$lambda_min_over_trace <-
        geometry$lambda_min_over_trace
      failure_summary$target_max_abs_correlation <-
        geometry$target_max_abs_correlation
      failure_summary$attempts_evaluated <- answer$attempts_evaluated
      failure_summary$top_up_attempts <- answer$top_up_attempts
      failure_summary$denominator_drop <- partial$denominator_drop
      failure_summary$denominator_zero <- partial$denominator_zero
      failure_summary$denominator_all_zero <-
        partial$denominator_all_zero
      failure_summary$p_drop <- partial$p_drop
      failure_summary$p_zero <- partial$p_zero
      failure_summary$p_all_zero <- partial$p_all_zero
      failure_summary$reject_drop <- slopeaudit:::.sa_q1_reject(
        partial$p_drop,
        ALPHA
      )
      failure_summary$reject_zero <- slopeaudit:::.sa_q1_reject(
        partial$p_zero,
        ALPHA
      )
      failure_summary$reject_all_zero <- slopeaudit:::.sa_q1_reject(
        partial$p_all_zero,
        ALPHA
      )
      failure_summary$inner_failures <- partial$n_failure
      failure_summary$small_negative <- partial$n_small_negative
      failure_summary$material_negative <- partial$n_material_negative
      failure_summary$inner_rescue_attempts <- answer$rescue_attempts
      failure_summary$inner_rescue_successes <- answer$rescue_successes
      failure_summary$unresolved_after_rescue <-
        answer$unresolved_after_rescue
    }
    return(list(
      summary = failure_summary,
      problems = if (
        !inherits(answer, "error") &&
          !is.null(answer$replicates)
      ) {
        problem <- answer$replicates[
          answer$replicates$rescue_triggered |
            !answer$replicates$status %in% c("positive", "zero"),
          ,
          drop = FALSE
        ]
        if (nrow(problem)) {
          cbind(
            data.frame(
              task_id = task$task_id,
              cell_id = cell$cell_id,
              truth_role = cell$truth_role,
              truth_rho = cell$intercept_slope_correlation,
              pair_id = task$pair_id,
              outer = task$outer,
              design_seed = task$design_seed,
              bootstrap_seed = task$bootstrap_seed
            ),
            problem
          )
        } else {
          empty_problem_rows()
        }
      } else {
        empty_problem_rows()
      }
    ))
  }

  bootstrap <- answer$bootstrap
  problems <- answer$replicates[
    answer$replicates$rescue_triggered |
      !answer$replicates$status %in% c("positive", "zero"),
    ,
    drop = FALSE
  ]
  if (nrow(problems)) {
    problems <- cbind(
      data.frame(
        task_id = task$task_id,
        cell_id = cell$cell_id,
        truth_role = cell$truth_role,
        truth_rho = cell$intercept_slope_correlation,
        pair_id = task$pair_id,
        outer = task$outer,
        design_seed = task$design_seed,
        bootstrap_seed = task$bootstrap_seed
      ),
      problems
    )
  } else {
    problems <- empty_problem_rows()
  }

  decision_bounds <- inner_decision_bounds(
    bootstrap$p_drop,
    answer$usable_target,
    answer$attempts_evaluated
  )
  summary <- data.frame(
    task_id = task$task_id,
    cell_id = cell$cell_id,
    cell_family = cell$cell_family,
    truth_role = cell$truth_role,
    truth_rho = cell$intercept_slope_correlation,
    pair_id = task$pair_id,
    outer = task$outer,
    design_seed = task$design_seed,
    bootstrap_seed = task$bootstrap_seed,
    status = "ok",
    failure_reason = "",
    direct_statistic = direct_statistic,
    observed_LR = answer$observed$LR_for_inference,
    observed_fit_source = answer$observed$source,
    observed_rescue_triggered = answer$observed$rescue_triggered,
    observed_rescue_succeeded = answer$observed$rescue_succeeded,
    full_singular = answer$observed$chosen$full_singular,
    lambda_min_over_trace = geometry$lambda_min_over_trace,
    target_max_abs_correlation = geometry$target_max_abs_correlation,
    usable_target = answer$usable_target,
    attempt_cap = answer$attempt_cap,
    attempts_evaluated = answer$attempts_evaluated,
    top_up_attempts = answer$top_up_attempts,
    denominator_drop = bootstrap$denominator_drop,
    denominator_zero = bootstrap$denominator_zero,
    denominator_all_zero = bootstrap$denominator_all_zero,
    p_drop = bootstrap$p_drop,
    p_zero = bootstrap$p_zero,
    p_all_zero = bootstrap$p_all_zero,
    usable_exceedances = as.integer(decision_bounds[["exceedances"]]),
    excluded_inner_attempts = as.integer(decision_bounds[["excluded"]]),
    p_if_excluded_all_nonexceed = decision_bounds[["p_all_nonexceed"]],
    p_if_excluded_all_exceed = decision_bounds[["p_all_exceed"]],
    inner_missingness_decision_stable = as.logical(
      decision_bounds[["stable"]]
    ),
    reject_drop = slopeaudit:::.sa_q1_reject(
      bootstrap$p_drop,
      ALPHA
    ),
    reject_zero = slopeaudit:::.sa_q1_reject(
      bootstrap$p_zero,
      ALPHA
    ),
    reject_all_zero = slopeaudit:::.sa_q1_reject(
      bootstrap$p_all_zero,
      ALPHA
    ),
    inner_failures = bootstrap$n_failure,
    small_negative = bootstrap$n_small_negative,
    material_negative = bootstrap$n_material_negative,
    inner_rescue_attempts = answer$rescue_attempts,
    inner_rescue_successes = answer$rescue_successes,
    unresolved_after_rescue = answer$unresolved_after_rescue,
    elapsed_seconds = proc.time()[["elapsed"]] - started,
    stringsAsFactors = FALSE
  )
  summary <- cbind(
    summary,
    observed_diagnostic_fields(answer$observed)
  )

  stopifnot(summary$denominator_drop == USABLE_B)
  stopifnot(
    summary$attempts_evaluated ==
      summary$denominator_drop + summary$inner_failures +
      summary$small_negative + summary$material_negative
  )

  list(summary = summary, problems = problems)
}

legacy_base_cell_ids <- c(
  "null_balanced_q0_0p5",
  "null_balanced_q0_4",
  "null_irregular_q0_0p5",
  "null_irregular_q0_4",
  "power_three_waves_two_years",
  "power_five_waves_five_years"
)
nested_bootstrap_base_cell_ids <- c(
  legacy_base_cell_ids[1:4],
  "power_three_waves_two_years_rho0",
  "power_five_waves_five_years_rho0"
)
N_BASE_RUNS <- length(nested_bootstrap_base_cell_ids) * OUTER_N
all_seeds <- slopeaudit:::.sa_q1_make_replicate_seeds(
  B = 2L * N_BASE_RUNS,
  seed = MASTER_SEED
)
base_task_manifest <- do.call(
  rbind,
  lapply(seq_along(nested_bootstrap_base_cell_ids), function(cell_index) {
    start <- (cell_index - 1L) * OUTER_N
    data.frame(
      task_index = start + seq_len(OUTER_N),
      cell_id = nested_bootstrap_base_cell_ids[[cell_index]],
      legacy_cell_id = legacy_base_cell_ids[[cell_index]],
      outer = seq_len(OUTER_N),
      stringsAsFactors = FALSE
    )
  })
)
base_task_manifest$design_seed <- all_seeds[base_task_manifest$task_index]
base_task_manifest$bootstrap_seed <-
  all_seeds[N_BASE_RUNS + base_task_manifest$task_index]

task_manifest <- do.call(rbind, lapply(
  seq_len(nrow(cell_manifest)),
  function(cell_index) {
    cell <- cell_manifest[cell_index, ]
    source_cell <- if (cell$pair_role == "rho0p5") {
      cell$pair_base_cell
    } else {
      cell$cell_id
    }
    source <- base_task_manifest[
      base_task_manifest$cell_id == source_cell,
      ,
      drop = FALSE
    ]
    if (nrow(source) != OUTER_N) {
      stop("Could not resolve one nested-bootstrap validation task seed source.")
    }
    data.frame(
      task_id = sprintf("nested_bootstrap_%s_%04d", cell$cell_id, source$outer),
      cell_id = cell$cell_id,
      truth_role = cell$truth_role,
      truth_rho = cell$intercept_slope_correlation,
      pair_id = if (cell$pair_role == "none") "" else sprintf(
        "%s::%04d", cell$pair_base_cell, source$outer
      ),
      outer = source$outer,
      design_seed = source$design_seed,
      bootstrap_seed = source$bootstrap_seed,
      seed_source_cell = source_cell,
      stringsAsFactors = FALSE
    )
  }
))
task_manifest$task_index <- seq_len(nrow(task_manifest))
task_manifest <- task_manifest[, c(
  "task_index", "task_id", "cell_id", "truth_role", "truth_rho",
  "pair_id", "outer", "design_seed", "bootstrap_seed", "seed_source_cell"
)]
N_RUNS <- nrow(task_manifest)

if (anyDuplicated(task_manifest$task_id) ||
    anyDuplicated(paste(task_manifest$cell_id, task_manifest$outer))) {
  stop("nested-bootstrap validation task IDs and cell-by-outer keys must be unique.")
}
seed_groups <- split(task_manifest, task_manifest$design_seed)
seed_reuse_ok <- vapply(seed_groups, function(group) {
  nrow(group) == 1L ||
    (nrow(group) == 2L && length(unique(group$pair_id)) == 1L &&
      nzchar(group$pair_id[[1]]) &&
      setequal(group$truth_role, c("alternative", "paired_sensitivity")) &&
      length(unique(group$bootstrap_seed)) == 1L)
}, logical(1))
if (!all(seed_reuse_ok)) {
  stop("Seeds may repeat only inside declared rho=0/rho=.5 pairs.")
}

if (MODE == "pilot") {
  frozen_task_path <- file.path(
    FROZEN_PILOT_INPUT,
    "q1_four_design_task_manifest.csv"
  )
  frozen_gate_path <- file.path(
    FROZEN_PILOT_INPUT,
    "q1_four_design_global_gate.csv"
  )
  if (!file.exists(frozen_task_path) || !file.exists(frozen_gate_path)) {
    stop("Missing frozen v12 pilot task/gate input.")
  }
  frozen_tasks <- utils::read.csv(
    frozen_task_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  current_legacy_tasks <- base_task_manifest[, c(
    "task_index", "legacy_cell_id", "outer", "design_seed", "bootstrap_seed"
  )]
  names(current_legacy_tasks)[[2]] <- "cell_id"
  same_tasks <- isTRUE(all.equal(
    current_legacy_tasks,
    frozen_tasks,
    check.attributes = TRUE
  ))
  frozen_gate <- utils::read.csv(
    frozen_gate_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  obsolete_target_gate <-
    frozen_gate$check == "all_cell_two_sided_intervals_include_targets"
  if (!same_tasks || OUTER_N != 20L || USABLE_B != 99L ||
      ATTEMPT_CAP != 149L || ALPHA != 0.05 ||
      nrow(frozen_gate) != 8L || sum(obsolete_target_gate) != 1L ||
      !all(frozen_gate$pass[!obsolete_target_gate])) {
    stop(
      paste0(
        "pilot must use the exact audited v12 120-task manifest, seeds, ",
        "B, cap, alpha, and operational gates. The obsolete analytic-power ",
        "target gate is intentionally replaced by the v13.3.1 references."
      )
    )
  }
}

utils::write.csv(
  task_manifest,
  file.path(OUT, "q1_nested_bootstrap_task_manifest.csv"),
  row.names = FALSE
)

checkpoint_path <- file.path(OUT, "q1_nested_bootstrap_checkpoint.rds")
outer_csv <- file.path(OUT, "q1_nested_bootstrap_outer_runs.csv")
problem_csv <- file.path(OUT, "q1_nested_bootstrap_rescue_attempts.csv")

upgrade_imported_outer <- function(imported) {
  map <- c(
    power_three_waves_two_years = "power_three_waves_two_years_rho0",
    power_five_waves_five_years = "power_five_waves_five_years_rho0"
  )
  replace <- imported$cell_id %in% names(map)
  imported$cell_id[replace] <- unname(map[imported$cell_id[replace]])
  metadata <- task_manifest[match(
    paste(imported$cell_id, imported$outer),
    paste(task_manifest$cell_id, task_manifest$outer)
  ), ]
  if (anyNA(metadata$task_id)) {
    stop("Frozen v12 pilot rows do not map to the nested-bootstrap validation manifest.")
  }
  imported$task_id <- metadata$task_id
  imported$truth_role <- metadata$truth_role
  imported$truth_rho <- metadata$truth_rho
  imported$pair_id <- metadata$pair_id
  imported$direct_statistic <- NA_real_
  for (index in which(imported$cell_family == "planning_power_validation")) {
    cell <- cell_manifest[cell_manifest$cell_id == imported$cell_id[[index]], ]
    dat <- generate_outcome(
      schedules[[cell$schedule]], cell, imported$design_seed[[index]]
    )
    imported$direct_statistic[[index]] <- direct_slope_statistic(
      dat, cell$residual_variance
    )
  }
  bounds <- t(vapply(seq_len(nrow(imported)), function(index) {
    inner_decision_bounds(
      imported$p_drop[[index]],
      imported$denominator_drop[[index]],
      imported$attempts_evaluated[[index]]
    )
  }, numeric(5)))
  imported$usable_exceedances <- as.integer(bounds[, "exceedances"])
  imported$excluded_inner_attempts <- as.integer(bounds[, "excluded"])
  imported$p_if_excluded_all_nonexceed <- bounds[, "p_all_nonexceed"]
  imported$p_if_excluded_all_exceed <- bounds[, "p_all_exceed"]
  imported$inner_missingness_decision_stable <-
    as.logical(bounds[, "stable"])
  imported
}

upgrade_imported_problems <- function(imported) {
  map <- c(
    power_three_waves_two_years = "power_three_waves_two_years_rho0",
    power_five_waves_five_years = "power_five_waves_five_years_rho0"
  )
  replace <- imported$cell_id %in% names(map)
  imported$cell_id[replace] <- unname(map[imported$cell_id[replace]])
  metadata <- task_manifest[match(
    paste(imported$cell_id, imported$outer),
    paste(task_manifest$cell_id, task_manifest$outer)
  ), ]
  imported$task_id <- metadata$task_id
  imported$truth_role <- metadata$truth_role
  imported$truth_rho <- metadata$truth_rho
  imported$pair_id <- metadata$pair_id
  imported[, names(empty_problem_rows()), drop = FALSE]
}

if (file.exists(checkpoint_path)) {
  checkpoint <- readRDS(checkpoint_path)
  outer_results <- checkpoint$outer_results
  inner_problems <- checkpoint$inner_problems
} else if (MODE == "pilot") {
  frozen_checkpoint_path <- file.path(
    FROZEN_PILOT_INPUT,
    "q1_four_design_checkpoint.rds"
  )
  frozen_checkpoint <- readRDS(frozen_checkpoint_path)
  if (nrow(frozen_checkpoint$outer_results) != 120L ||
      any(frozen_checkpoint$outer_results$status != "ok") ||
      any(frozen_checkpoint$outer_results$denominator_drop != 99L)) {
    stop("The frozen v12 pilot checkpoint failed its import contract.")
  }
  outer_results <- upgrade_imported_outer(frozen_checkpoint$outer_results)
  inner_problems <- upgrade_imported_problems(
    frozen_checkpoint$inner_problems
  )
  utils::write.csv(
    data.frame(
      source = "audited_v12_pilot_checkpoint",
      imported_outer_rows = nrow(outer_results),
      imported_problem_rows = nrow(inner_problems),
      checkpoint_md5 = unname(tools::md5sum(frozen_checkpoint_path)),
      stringsAsFactors = FALSE
    ),
    file.path(OUT, "q1_nested_bootstrap_import_provenance.csv"),
    row.names = FALSE
  )
} else {
  outer_results <- NULL
  inner_problems <- empty_problem_rows()
}

save_checkpoint <- function() {
  if (!is.null(outer_results) && nrow(outer_results)) {
    outer_results <<- outer_results[
      order(outer_results$cell_id, outer_results$outer),
      ,
      drop = FALSE
    ]
    rownames(outer_results) <<- NULL
  }
  if (nrow(inner_problems)) {
    inner_problems <<- inner_problems[
      order(
        inner_problems$cell_id,
        inner_problems$outer,
        inner_problems$replicate
      ),
      ,
      drop = FALSE
    ]
    rownames(inner_problems) <<- NULL
  }
  temporary <- paste0(checkpoint_path, ".tmp")
  saveRDS(
    list(
      outer_results = outer_results,
      inner_problems = inner_problems
    ),
    temporary,
    version = 3
  )
  if (!file.rename(temporary, checkpoint_path)) {
    stop("Could not publish the calibration checkpoint atomically.")
  }
  utils::write.csv(outer_results, outer_csv, row.names = FALSE)
  utils::write.csv(inner_problems, problem_csv, row.names = FALSE)
}

completed_keys <- if (!is.null(outer_results) && nrow(outer_results)) {
  paste(outer_results$cell_id, outer_results$outer, sep = "::")
} else {
  character()
}
task_keys <- paste(task_manifest$cell_id, task_manifest$outer, sep = "::")
missing <- task_manifest$task_index[!task_keys %in% completed_keys]

if (length(missing)) {
  progress_bar <- utils::txtProgressBar(
    min = 0,
    max = length(missing),
    initial = 0,
    style = 3
  )
  completed_now <- 0L

  batches <- split(
    missing,
    ceiling(seq_along(missing) / CHECKPOINT_BATCH)
  )
  for (batch in batches) {
    answers <- if (WORKERS == 1L) {
      lapply(batch, run_one)
    } else {
      parallel::mclapply(
        batch,
        run_one,
        mc.cores = WORKERS,
        mc.preschedule = FALSE,
        mc.set.seed = FALSE
      )
    }

    batch_outer <- do.call(
      rbind,
      lapply(answers, `[[`, "summary")
    )
    batch_problems <- do.call(
      rbind,
      lapply(answers, `[[`, "problems")
    )

    outer_results <- rbind(outer_results, batch_outer)
    if (!is.null(batch_problems) && nrow(batch_problems)) {
      inner_problems <- rbind(inner_problems, batch_problems)
    }
    save_checkpoint()

    completed_now <- completed_now + length(batch)
    utils::setTxtProgressBar(progress_bar, completed_now)
  }
  close(progress_bar)
}

exact_interval <- function(k, n) {
  if (!is.finite(k) || !is.finite(n) || n < 1L || k < 0L || k > n) {
    return(c(NA_real_, NA_real_))
  }
  slopeaudit:::.sa_q1_exact_binomial_interval(
    as.integer(k),
    as.integer(n),
    0.95
  )
}

safe_spearman <- function(x, y) {
  keep <- is.finite(x) & is.finite(y)
  x <- x[keep]
  y <- y[keep]
  if (length(x) < 5L || length(unique(x)) < 2L ||
      length(unique(y)) < 2L) {
    return(c(rho = NA_real_, n = length(x)))
  }
  c(
    rho = suppressWarnings(stats::cor(x, y, method = "spearman")),
    n = length(x)
  )
}

reference_compatibility <- function(
    stage_low,
    stage_high,
    reference_rate,
    reference_low,
    reference_high) {
  if (!is.finite(stage_low) || !is.finite(stage_high) ||
      !is.finite(reference_rate)) {
    return(FALSE)
  }
  if (is.finite(reference_low) && is.finite(reference_high)) {
    return(stage_low <= reference_high && stage_high >= reference_low)
  }
  reference_rate >= stage_low && reference_rate <= stage_high
}

summarize_cell <- function(index) {
  cell <- cell_manifest[index, ]
  all_cell <- outer_results[
    outer_results$cell_id == cell$cell_id,
    ,
    drop = FALSE
  ]
  valid <- all_cell[
    all_cell$status == "ok" &
      all_cell$denominator_drop == USABLE_B,
    ,
    drop = FALSE
  ]

  primary_supports <- sum(valid$reject_drop, na.rm = TRUE)
  n_valid <- nrow(valid)
  primary_interval <- exact_interval(primary_supports, n_valid)

  definitely_reject <- is.finite(valid$p_if_excluded_all_exceed) &
    valid$p_if_excluded_all_exceed <= ALPHA
  definitely_nonreject <- is.finite(valid$p_if_excluded_all_nonexceed) &
    valid$p_if_excluded_all_nonexceed > ALPHA
  ambiguous_valid <- !(definitely_reject | definitely_nonreject)
  unknown_outer <- OUTER_N - n_valid

  support_lower <- sum(definitely_reject)
  support_upper <- support_lower + sum(ambiguous_valid) + unknown_outer
  rate_lower <- support_lower / OUTER_N
  rate_upper <- support_upper / OUTER_N
  lower_interval <- exact_interval(support_lower, OUTER_N)
  upper_interval <- exact_interval(support_upper, OUTER_N)
  decision_interval_low <- lower_interval[[1]]
  decision_interval_high <- upper_interval[[2]]

  reference_rate <- cell$reference_rate
  reference_low <- cell$reference_interval_low
  reference_high <- cell$reference_interval_high
  reference_compatible <- reference_compatibility(
    decision_interval_low,
    decision_interval_high,
    reference_rate,
    reference_low,
    reference_high
  )
  compatible_if_all_unknown_nonreject <- reference_compatibility(
    lower_interval[[1]],
    lower_interval[[2]],
    reference_rate,
    reference_low,
    reference_high
  )
  compatible_if_all_unknown_reject <- reference_compatibility(
    upper_interval[[1]],
    upper_interval[[2]],
    reference_rate,
    reference_low,
    reference_high
  )
  issue_correlation <- safe_spearman(
    valid$excluded_inner_attempts,
    valid$observed_LR
  )

  data.frame(
    cell_id = cell$cell_id,
    cell_family = cell$cell_family,
    truth_role = cell$truth_role,
    truth_rho = cell$intercept_slope_correlation,
    n_requested = OUTER_N,
    n_completed = nrow(all_cell),
    n_valid = n_valid,
    validity_rate = n_valid / OUTER_N,
    observed_unresolved_after_rescue = sum(
      all_cell$status == "observed_unresolved_after_rescue"
    ),
    q1_invalid_observed = sum(all_cell$status == "q1_invalid_observed"),
    q1_errors = sum(all_cell$status == "q1_error"),
    attempt_cap_failures = sum(all_cell$status == "attempt_cap_reached"),
    primary_supports = primary_supports,
    primary_support_rate = if (n_valid) {
      primary_supports / n_valid
    } else {
      NA_real_
    },
    primary_interval_low = primary_interval[[1]],
    primary_interval_high = primary_interval[[2]],
    primary_MCSE = if (n_valid) {
      sqrt(
        (primary_supports / n_valid) *
          (1 - primary_supports / n_valid) / n_valid
      )
    } else {
      NA_real_
    },
    definitely_supports = support_lower,
    ambiguous_inner_decisions = sum(ambiguous_valid),
    unresolved_outer_decisions = unknown_outer,
    support_count_lower = support_lower,
    support_count_upper = support_upper,
    support_rate_lower = rate_lower,
    support_rate_upper = rate_upper,
    decision_bound_interval_low = decision_interval_low,
    decision_bound_interval_high = decision_interval_high,
    reference_source = cell$reference_source,
    reference_rate = reference_rate,
    reference_interval_low = reference_low,
    reference_interval_high = reference_high,
    reference_point_in_decision_bound_interval =
      reference_rate >= decision_interval_low &&
        reference_rate <= decision_interval_high,
    reference_compatible = reference_compatible,
    unresolved_changes_reference_compatibility =
      compatible_if_all_unknown_nonreject !=
        compatible_if_all_unknown_reject,
    total_inner_attempts = sum(valid$attempts_evaluated, na.rm = TRUE),
    total_excluded_inner_attempts = sum(
      valid$excluded_inner_attempts,
      na.rm = TRUE
    ),
    excluded_inner_attempt_rate = if (sum(
      valid$attempts_evaluated,
      na.rm = TRUE
    ) > 0) {
      sum(valid$excluded_inner_attempts, na.rm = TRUE) /
        sum(valid$attempts_evaluated, na.rm = TRUE)
    } else {
      NA_real_
    },
    outer_runs_with_inner_topups = sum(valid$excluded_inner_attempts > 0),
    maximum_top_up_attempts = if (n_valid) {
      max(valid$top_up_attempts)
    } else {
      NA_integer_
    },
    total_inner_failures = sum(valid$inner_failures, na.rm = TRUE),
    total_small_negative = sum(valid$small_negative, na.rm = TRUE),
    total_material_negative = sum(valid$material_negative, na.rm = TRUE),
    observed_rescue_attempts_all_runs = sum(
      all_cell$observed_rescue_triggered,
      na.rm = TRUE
    ),
    observed_rescue_successes_all_runs = sum(
      all_cell$observed_rescue_succeeded,
      na.rm = TRUE
    ),
    inner_rescue_attempts_all_runs = sum(
      all_cell$inner_rescue_attempts,
      na.rm = TRUE
    ),
    inner_rescue_successes_all_runs = sum(
      all_cell$inner_rescue_successes,
      na.rm = TRUE
    ),
    unresolved_inner_attempts_all_runs = sum(
      all_cell$unresolved_after_rescue,
      na.rm = TRUE
    ),
    drop_zero_decision_disagreements = sum(
      valid$reject_drop != valid$reject_zero,
      na.rm = TRUE
    ),
    drop_all_zero_decision_disagreements = sum(
      valid$reject_drop != valid$reject_all_zero,
      na.rm = TRUE
    ),
    issue_count_observed_LR_spearman = issue_correlation[["rho"]],
    issue_count_observed_LR_n = issue_correlation[["n"]],
    analytic_R_direct = cell$analytic_R_direct,
    analytic_direct_channel_benchmark = cell$analytic_power,
    historical_k = cell$historical_k,
    historical_n = cell$historical_n,
    historical_rate = cell$historical_k / cell$historical_n,
    mean_elapsed_seconds = if (n_valid) {
      mean(valid$elapsed_seconds)
    } else {
      NA_real_
    },
    operational_cell_pass =
      nrow(all_cell) == OUTER_N &&
      sum(all_cell$status == "attempt_cap_reached") == 0L &&
      sum(all_cell$status == "q1_error") == 0L &&
      unknown_outer / OUTER_N <= MAX_OBSERVED_UNRESOLVED_RATE &&
      all(valid$denominator_drop == USABLE_B),
    stringsAsFactors = FALSE
  )
}

nested_bootstrap_summary <- do.call(
  rbind,
  lapply(seq_len(nrow(cell_manifest)), summarize_cell)
)
utils::write.csv(
  nested_bootstrap_summary,
  file.path(OUT, "q1_nested_bootstrap_summary.csv"),
  row.names = FALSE
)

paired_rows <- lapply(
  c("three_waves_two_years", "five_waves_five_years"),
  function(design_id) {
    rho0_id <- paste0("power_", design_id, "_rho0")
    rho5_id <- paste0("power_", design_id, "_rho0p5")
    rho0 <- outer_results[
      outer_results$cell_id == rho0_id,
      c(
        "pair_id", "status", "design_seed", "bootstrap_seed",
        "direct_statistic", "reject_drop",
        "inner_missingness_decision_stable"
      ),
      drop = FALSE
    ]
    rho5 <- outer_results[
      outer_results$cell_id == rho5_id,
      c(
        "pair_id", "status", "design_seed", "bootstrap_seed",
        "direct_statistic", "reject_drop",
        "inner_missingness_decision_stable"
      ),
      drop = FALSE
    ]
    paired <- merge(
      rho0,
      rho5,
      by = "pair_id",
      suffixes = c("_rho0", "_rho0p5"),
      all = TRUE,
      sort = TRUE
    )
    valid <- paired[
      paired$status_rho0 == "ok" & paired$status_rho0p5 == "ok",
      ,
      drop = FALSE
    ]
    difference <- if (nrow(valid)) {
      as.numeric(valid$reject_drop_rho0p5) -
        as.numeric(valid$reject_drop_rho0)
    } else {
      numeric()
    }
    direct_difference <- paired$direct_statistic_rho0p5 -
      paired$direct_statistic_rho0

    data.frame(
      design_id = design_id,
      n_requested_pairs = OUTER_N,
      n_manifest_pairs = nrow(paired),
      n_both_valid = nrow(valid),
      power_rho0 = if (nrow(valid)) {
        mean(valid$reject_drop_rho0)
      } else {
        NA_real_
      },
      power_rho0p5 = if (nrow(valid)) {
        mean(valid$reject_drop_rho0p5)
      } else {
        NA_real_
      },
      difference_rho0p5_minus_rho0 = if (nrow(valid)) {
        mean(difference)
      } else {
        NA_real_
      },
      paired_SE = if (length(difference) > 1L) {
        stats::sd(difference) / sqrt(length(difference))
      } else {
        NA_real_
      },
      rho0_only_supports = if (nrow(valid)) {
        sum(valid$reject_drop_rho0 & !valid$reject_drop_rho0p5)
      } else {
        NA_integer_
      },
      rho0p5_only_supports = if (nrow(valid)) {
        sum(!valid$reject_drop_rho0 & valid$reject_drop_rho0p5)
      } else {
        NA_integer_
      },
      ambiguous_inner_decisions_in_valid_pairs = if (nrow(valid)) {
        sum(
          !valid$inner_missingness_decision_stable_rho0 |
            !valid$inner_missingness_decision_stable_rho0p5
        )
      } else {
        NA_integer_
      },
      design_seeds_identical = nrow(paired) == OUTER_N &&
        all(paired$design_seed_rho0 == paired$design_seed_rho0p5),
      bootstrap_seeds_identical = nrow(paired) == OUTER_N &&
        all(paired$bootstrap_seed_rho0 == paired$bootstrap_seed_rho0p5),
      max_abs_direct_statistic_difference = if (
        nrow(paired) && all(is.finite(direct_difference))
      ) {
        max(abs(direct_difference))
      } else {
        NA_real_
      },
      direct_channel_exactly_paired_within_1e_10 =
        nrow(paired) == OUTER_N &&
        all(is.finite(direct_difference)) &&
        max(abs(direct_difference)) <= 1e-10,
      stringsAsFactors = FALSE
    )
  }
)
paired_rho_comparison <- do.call(rbind, paired_rows)
utils::write.csv(
  paired_rho_comparison,
  file.path(OUT, "q1_nested_bootstrap_paired_rho_comparison.csv"),
  row.names = FALSE
)

pooled_null <- nested_bootstrap_summary[
  nested_bootstrap_summary$cell_family == "null_size",
  ,
  drop = FALSE
]
pooled_null_requested <- sum(pooled_null$n_requested)
pooled_null_valid <- sum(pooled_null$n_valid)
pooled_null_primary_supports <- sum(pooled_null$primary_supports)
pooled_null_lower <- sum(pooled_null$support_count_lower)
pooled_null_upper <- sum(pooled_null$support_count_upper)
pooled_null_primary_interval <- exact_interval(
  pooled_null_primary_supports,
  pooled_null_valid
)
pooled_null_lower_interval <- exact_interval(
  pooled_null_lower,
  pooled_null_requested
)
pooled_null_upper_interval <- exact_interval(
  pooled_null_upper,
  pooled_null_requested
)
pooled_null_summary <- data.frame(
  analysis_id = "ALL_FOUR_NULL_CELLS",
  n_requested = pooled_null_requested,
  n_valid = pooled_null_valid,
  primary_false_supports = pooled_null_primary_supports,
  primary_false_support_rate = pooled_null_primary_supports /
    pooled_null_valid,
  primary_interval_low = pooled_null_primary_interval[[1]],
  primary_interval_high = pooled_null_primary_interval[[2]],
  false_support_count_lower = pooled_null_lower,
  false_support_count_upper = pooled_null_upper,
  false_support_rate_lower = pooled_null_lower / pooled_null_requested,
  false_support_rate_upper = pooled_null_upper / pooled_null_requested,
  decision_bound_interval_low = pooled_null_lower_interval[[1]],
  decision_bound_interval_high = pooled_null_upper_interval[[2]],
  target_alpha = ALPHA,
  target_compatible = ALPHA >= pooled_null_lower_interval[[1]] &&
    ALPHA <= pooled_null_upper_interval[[2]],
  total_excluded_inner_attempts = sum(
    pooled_null$total_excluded_inner_attempts
  ),
  ambiguous_inner_decisions = sum(pooled_null$ambiguous_inner_decisions),
  unresolved_outer_decisions = sum(pooled_null$unresolved_outer_decisions),
  stringsAsFactors = FALSE
)
utils::write.csv(
  pooled_null_summary,
  file.path(OUT, "q1_nested_bootstrap_pooled_null_summary.csv"),
  row.names = FALSE
)

valid_all <- outer_results[
  outer_results$status == "ok" &
    outer_results$denominator_drop == USABLE_B,
  ,
  drop = FALSE
]
attempt_arithmetic_ok <- nrow(valid_all) == 0L || all(
  valid_all$attempts_evaluated ==
    valid_all$denominator_drop +
    valid_all$inner_failures +
    valid_all$small_negative +
    valid_all$material_negative
)
topup_arithmetic_ok <- nrow(valid_all) == 0L || all(
  valid_all$top_up_attempts == valid_all$excluded_inner_attempts &
    valid_all$excluded_inner_attempts ==
      valid_all$inner_failures +
      valid_all$small_negative +
      valid_all$material_negative
)
task_contract_ok <- nrow(outer_results) == N_RUNS &&
  !anyDuplicated(outer_results$task_id) &&
  setequal(outer_results$task_id, task_manifest$task_id)

all_run_diagnostics <- data.frame(
  n_tasks = nrow(outer_results),
  n_valid = nrow(valid_all),
  observed_unresolved_after_rescue = sum(
    outer_results$status == "observed_unresolved_after_rescue"
  ),
  q1_invalid_observed = sum(
    outer_results$status == "q1_invalid_observed"
  ),
  q1_errors = sum(outer_results$status == "q1_error"),
  attempt_cap_failures = sum(
    outer_results$status == "attempt_cap_reached"
  ),
  inner_attempts_valid_runs = sum(
    valid_all$attempts_evaluated,
    na.rm = TRUE
  ),
  excluded_inner_attempts_valid_runs = sum(
    valid_all$excluded_inner_attempts,
    na.rm = TRUE
  ),
  excluded_inner_attempt_rate = sum(
    valid_all$excluded_inner_attempts,
    na.rm = TRUE
  ) / sum(valid_all$attempts_evaluated, na.rm = TRUE),
  outer_runs_with_topups = sum(
    valid_all$excluded_inner_attempts > 0,
    na.rm = TRUE
  ),
  ambiguous_inner_decisions = sum(
    !valid_all$inner_missingness_decision_stable,
    na.rm = TRUE
  ),
  observed_rescue_attempts = sum(
    outer_results$observed_rescue_triggered,
    na.rm = TRUE
  ),
  observed_rescue_successes = sum(
    outer_results$observed_rescue_succeeded,
    na.rm = TRUE
  ),
  inner_rescue_attempts = sum(
    outer_results$inner_rescue_attempts,
    na.rm = TRUE
  ),
  inner_rescue_successes = sum(
    outer_results$inner_rescue_successes,
    na.rm = TRUE
  ),
  unresolved_inner_attempts = sum(
    outer_results$unresolved_after_rescue,
    na.rm = TRUE
  ),
  retained_rescue_or_problem_rows = nrow(inner_problems),
  attempt_arithmetic_ok = attempt_arithmetic_ok,
  topup_arithmetic_ok = topup_arithmetic_ok,
  stringsAsFactors = FALSE
)
utils::write.csv(
  all_run_diagnostics,
  file.path(OUT, "q1_nested_bootstrap_all_run_diagnostics.csv"),
  row.names = FALSE
)

reference_gate <- data.frame(
  cell_id = nested_bootstrap_summary$cell_id,
  mode = MODE,
  enforced = MODE != "smoke",
  reference_source = nested_bootstrap_summary$reference_source,
  reference_rate = nested_bootstrap_summary$reference_rate,
  nested_bootstrap_decision_interval_low =
    nested_bootstrap_summary$decision_bound_interval_low,
  nested_bootstrap_decision_interval_high =
    nested_bootstrap_summary$decision_bound_interval_high,
  reference_compatible = nested_bootstrap_summary$reference_compatible,
  unresolved_changes_reference_compatibility =
    nested_bootstrap_summary$unresolved_changes_reference_compatibility,
  pass = if (MODE == "smoke") {
    rep(TRUE, nrow(nested_bootstrap_summary))
  } else {
    nested_bootstrap_summary$reference_compatible
  },
  stringsAsFactors = FALSE
)
utils::write.csv(
  reference_gate,
  file.path(OUT, "q1_nested_bootstrap_reference_gate.csv"),
  row.names = FALSE
)

maximum_observed_issue_rate <- max(
  (
    nested_bootstrap_summary$observed_unresolved_after_rescue +
      nested_bootstrap_summary$q1_invalid_observed
  ) / nested_bootstrap_summary$n_requested
)
operational_gate <- data.frame(
  check = c(
    "task_manifest_complete_unique_and_exact",
    "every_valid_outer_run_reached_fixed_usable_B",
    "no_attempt_cap_failure_or_q1_error",
    "observed_unresolved_rate_within_declared_limit",
    "inner_attempt_and_topup_arithmetic_reconciles",
    "unresolved_inner_attempts_are_bounded_not_silently_imputed",
    "paired_rho_seeds_and_direct_channel_are_identical",
    "all_cell_operational_checks_pass",
    "pooled_null_decision_bounds_include_alpha"
  ),
  value = c(
    paste0(nrow(outer_results), "/", N_RUNS),
    paste0(
      sum(valid_all$denominator_drop == USABLE_B), "/",
      nrow(valid_all), " valid runs"
    ),
    paste0(
      all_run_diagnostics$attempt_cap_failures, " caps; ",
      all_run_diagnostics$q1_errors, " errors"
    ),
    paste0(
      format(maximum_observed_issue_rate, digits = 6),
      " maximum; limit ",
      MAX_OBSERVED_UNRESOLVED_RATE
    ),
    paste0(
      all_run_diagnostics$excluded_inner_attempts_valid_runs,
      " excluded/top-up attempts"
    ),
    paste0(
      all_run_diagnostics$ambiguous_inner_decisions,
      " outer decisions remain worst-case ambiguous"
    ),
    paste0(
      max(paired_rho_comparison$max_abs_direct_statistic_difference),
      " maximum direct-statistic difference"
    ),
    paste0(
      sum(nested_bootstrap_summary$operational_cell_pass), "/",
      nrow(nested_bootstrap_summary)
    ),
    paste0(
      pooled_null_summary$decision_bound_interval_low,
      " to ",
      pooled_null_summary$decision_bound_interval_high
    )
  ),
  pass = c(
    task_contract_ok,
    nrow(valid_all) > 0L &&
      all(valid_all$denominator_drop == USABLE_B),
    all_run_diagnostics$attempt_cap_failures == 0L &&
      all_run_diagnostics$q1_errors == 0L,
    maximum_observed_issue_rate <= MAX_OBSERVED_UNRESOLVED_RATE,
    attempt_arithmetic_ok && topup_arithmetic_ok,
    all(
      is.finite(valid_all$p_if_excluded_all_nonexceed) &
        is.finite(valid_all$p_if_excluded_all_exceed) &
        valid_all$p_if_excluded_all_nonexceed <=
          valid_all$p_if_excluded_all_exceed
    ),
    all(paired_rho_comparison$design_seeds_identical) &&
      all(paired_rho_comparison$bootstrap_seeds_identical) &&
      all(
        paired_rho_comparison$direct_channel_exactly_paired_within_1e_10
      ),
    all(nested_bootstrap_summary$operational_cell_pass),
    isTRUE(pooled_null_summary$target_compatible)
  ),
  stringsAsFactors = FALSE
)
utils::write.csv(
  operational_gate,
  file.path(OUT, "q1_nested_bootstrap_operational_gate.csv"),
  row.names = FALSE
)

correlation_rows <- lapply(
  c("ALL_NULL_CELLS", cell_manifest$cell_id),
  function(cell_id) {
    subset <- if (cell_id == "ALL_NULL_CELLS") {
      valid_all[valid_all$cell_family == "null_size", , drop = FALSE]
    } else {
      valid_all[valid_all$cell_id == cell_id, , drop = FALSE]
    }
    correlation <- safe_spearman(
      subset$excluded_inner_attempts,
      subset$observed_LR
    )
    data.frame(
      cell_id = cell_id,
      n = correlation[["n"]],
      spearman_rho = correlation[["rho"]],
      interpretation = paste0(
        "Indirect diagnostic only: association between excluded/top-up ",
        "inner attempts and the outer observed LR does not prove that ",
        "failures are independent of their unobserved LR values."
      ),
      stringsAsFactors = FALSE
    )
  }
)
utils::write.csv(
  do.call(rbind, correlation_rows),
  file.path(OUT, "q1_nested_bootstrap_failure_association.csv"),
  row.names = FALSE
)

if (MODE == "pilot") {
  frozen_outer <- utils::read.csv(
    file.path(FROZEN_PILOT_INPUT, "q1_four_design_outer_runs.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  mapped <- outer_results[
    outer_results$truth_role != "paired_sensitivity",
    ,
    drop = FALSE
  ]
  mapped$legacy_cell_id <- mapped$cell_id
  mapped$legacy_cell_id[
    mapped$legacy_cell_id == "power_three_waves_two_years_rho0"
  ] <- "power_three_waves_two_years"
  mapped$legacy_cell_id[
    mapped$legacy_cell_id == "power_five_waves_five_years_rho0"
  ] <- "power_five_waves_five_years"
  compare_fields <- c(
    "status", "observed_LR", "p_drop", "reject_drop",
    "attempts_evaluated", "inner_failures", "small_negative",
    "material_negative"
  )
  comparison <- merge(
    mapped[, c(
      "legacy_cell_id", "outer", "design_seed", "bootstrap_seed",
      compare_fields
    )],
    frozen_outer[, c(
      "cell_id", "outer", "design_seed", "bootstrap_seed", compare_fields
    )],
    by.x = c(
      "legacy_cell_id", "outer", "design_seed", "bootstrap_seed"
    ),
    by.y = c("cell_id", "outer", "design_seed", "bootstrap_seed"),
    suffixes = c("_current", "_v12"),
    all = TRUE,
    sort = TRUE
  )
  comparison$scientific_fields_identical <- vapply(
    seq_len(nrow(comparison)),
    function(index) all(vapply(compare_fields, function(field) {
      x <- comparison[[paste0(field, "_current")]][[index]]
      y <- comparison[[paste0(field, "_v12")]][[index]]
      isTRUE(all.equal(x, y, tolerance = 0, check.attributes = FALSE))
    }, logical(1))),
    logical(1)
  )
  utils::write.csv(
    comparison,
    file.path(OUT, "q1_nested_bootstrap_v12_pilot_comparison.csv"),
    row.names = FALSE
  )
  if (nrow(comparison) != 120L ||
      !all(comparison$scientific_fields_identical)) {
    stop("The imported v12 pilot rows changed scientifically.")
  }
}

overall_pass <- all(operational_gate$pass %in% TRUE) &&
  all(reference_gate$pass %in% TRUE)
run_status <- data.frame(
  mode = MODE,
  status = if (overall_pass) "PASS" else "DIAGNOSE",
  operational_checks_passed = sum(operational_gate$pass),
  operational_checks_total = nrow(operational_gate),
  reference_checks_enforced = sum(reference_gate$enforced),
  reference_checks_passed = sum(reference_gate$pass),
  reference_checks_total = nrow(reference_gate),
  note = if (MODE == "smoke") {
    "Smoke references are descriptive; only operational checks are enforced."
  } else {
    paste0(
      "Power references are independently estimated statistic-level ",
      "benchmarks; interval compatibility is enforced."
    )
  },
  stringsAsFactors = FALSE
)
utils::write.csv(
  run_status,
  file.path(OUT, "q1_nested_bootstrap_run_status.csv"),
  row.names = FALSE
)

writeLines(
  capture.output(sessionInfo()),
  file.path(OUT, "sessionInfo_q1_nested_bootstrap_v14.txt")
)

print(nested_bootstrap_summary)
print(pooled_null_summary)
print(paired_rho_comparison)
print(operational_gate)
print(reference_gate)
if (overall_pass) {
  message(
    paste0(
      "PASS: v14 nested-bootstrap validation ", MODE,
      " nested-bootstrap diagnostic completed operationally."
    )
  )
} else {
  message(
    paste0(
      "DIAGNOSE: v14 nested-bootstrap validation ", MODE,
      " completed, but one or more declared gates failed."
    )
  )
}
