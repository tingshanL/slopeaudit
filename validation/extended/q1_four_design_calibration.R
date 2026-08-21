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

MODE <- tolower(Sys.getenv("SLOPEAUDIT_Q1_CAL_MODE", "pilot_repeat"))
mode_defaults <- list(
  smoke = list(outer = 2L, usable_B = 39L, workers = 1L),
  pilot_repeat = list(outer = 20L, usable_B = 99L, workers = 1L),
  full = list(outer = 999L, usable_B = 999L, workers = 1L)
)
if (!MODE %in% names(mode_defaults)) {
  stop("SLOPEAUDIT_Q1_CAL_MODE must be smoke, pilot_repeat, or full.")
}

defaults <- mode_defaults[[MODE]]
OUTER_N <- read_integer_env(
  "SLOPEAUDIT_Q1_CAL_OUTER",
  defaults$outer
)
USABLE_B <- read_integer_env(
  "SLOPEAUDIT_Q1_CAL_B_USABLE",
  defaults$usable_B
)
ATTEMPT_CAP <- read_integer_env(
  "SLOPEAUDIT_Q1_CAL_ATTEMPT_CAP",
  USABLE_B + max(50L, ceiling(0.25 * USABLE_B)),
  minimum = USABLE_B
)
WORKERS <- read_integer_env(
  "SLOPEAUDIT_Q1_CAL_WORKERS",
  defaults$workers
)
CHECKPOINT_BATCH <- read_integer_env(
  "SLOPEAUDIT_Q1_CAL_CHECKPOINT_BATCH",
  max(1L, WORKERS)
)
MASTER_SEED <- read_integer_env(
  "SLOPEAUDIT_Q1_CAL_SEED",
  20260812L,
  minimum = 0L
)
ALPHA <- read_numeric_env("SLOPEAUDIT_Q1_CAL_ALPHA", 0.05)
if (ALPHA <= 0 || ALPHA >= 1) {
  stop("SLOPEAUDIT_Q1_CAL_ALPHA must lie strictly between 0 and 1.")
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
  "SLOPEAUDIT_Q1_OUT",
  file.path(
    "validation",
    "extended",
    "results",
    paste0("q1_four_design_", MODE, "_v12")
  )
)
FROZEN_PILOT_INPUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_PILOT_INPUT",
  unset = ""
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
    "This historical calibration workflow requires private v11 adjudication ",
    "evidence. Set SLOPEAUDIT_Q1_V11_EVIDENCE_DIR explicitly.",
    call. = FALSE
  )
}
if (MODE == "pilot_repeat" && !nzchar(FROZEN_PILOT_INPUT)) {
  stop(
    "pilot_repeat requires the private frozen v10 pilot input. ",
    "Set SLOPEAUDIT_Q1_PILOT_INPUT explicitly.",
    call. = FALSE
  )
}
if (!nzchar(FIXTURE_PATH)) {
  stop(
    "This historical calibration workflow requires the private exact ",
    "irregular schedule. Set SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE explicitly.",
    call. = FALSE
  )
}

if (!file.exists(FIXTURE_PATH)) {
  stop(
    paste0(
      "Missing exact irregular schedule fixture: ", FIXTURE_PATH, "\n",
      "Provide the authorized private frozen schedule. ",
      "Do not substitute an approximate schedule."
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

cell_manifest <- data.frame(
  cell_id = c(
    "null_balanced_q0_0p5",
    "null_balanced_q0_4",
    "null_irregular_q0_0p5",
    "null_irregular_q0_4",
    "power_three_waves_two_years",
    "power_five_waves_five_years"
  ),
  cell_family = c(
    rep("null_size", 4L),
    rep("planning_power_validation", 2L)
  ),
  schedule = c(
    "balanced_three_wave",
    "balanced_three_wave",
    "irregular_fluid_cognition",
    "irregular_fluid_cognition",
    "three_waves_two_years",
    "five_waves_five_years"
  ),
  q0 = c(0.5, 4, 0.5, 4, 0.5, 0.5),
  slope_variance = c(0, 0, 0, 0, 0.01, 0.01),
  residual_variance = rep(1, 6L),
  fixed_slope = rep(0.10, 6L),
  historical_k = c(47L, 48L, 41L, 45L, NA, NA),
  historical_n = c(999L, 997L, 998L, 999L, NA, NA),
  stringsAsFactors = FALSE
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

scientific_config <- list(
  version = "q1_four_design_calibration_v12_rescue_candidate",
  mode = MODE,
  outer_per_cell = OUTER_N,
  usable_B = USABLE_B,
  attempt_cap = ATTEMPT_CAP,
  alpha = ALPHA,
  rejection_rule = "plus-one p <= alpha",
  primary_inner_convention = "fixed positive/zero usable denominator",
  master_seed = MASTER_SEED,
  irregular_fixture_md5 = unname(tools::md5sum(FIXTURE_PATH)),
  v11_evidence_md5 = unname(tools::md5sum(v11_evidence_paths)),
  cell_manifest = cell_manifest,
  initial_control = initial_control,
  rescue_control = rescue_control,
  rescue_rule = paste0(
    "initial nloptwrap full/null pair; if not primary-usable, refit both ",
    "with bobyqa; unresolved rescue triggers top-up"
  )
)

config_rds <- file.path(OUT, "q1_four_design_configuration.rds")
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
  file.path(OUT, "q1_four_design_cell_manifest.csv"),
  row.names = FALSE
)

generate_outcome <- function(schedule, cell, seed) {
  set.seed(seed)
  n_id <- nlevels(schedule$id)
  b0 <- stats::rnorm(n_id, 0, sqrt(cell$q0))
  b1 <- if (cell$slope_variance > 0) {
    stats::rnorm(n_id, 0, sqrt(cell$slope_variance))
  } else {
    rep(0, n_id)
  }
  id_index <- as.integer(schedule$id)
  schedule$y <- cell$fixed_slope * schedule$time +
    b0[id_index] +
    b1[id_index] * schedule$time +
    stats::rnorm(
      nrow(schedule),
      0,
      sqrt(cell$residual_variance)
    )
  schedule
}

empty_problem_rows <- function() {
  data.frame(
    cell_id = character(),
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
    outer,
    design_seed,
    bootstrap_seed,
    status,
    reason,
    elapsed_seconds,
    observed = NULL) {
  core <- data.frame(
    cell_id = cell$cell_id,
    cell_family = cell$cell_family,
    outer = outer,
    design_seed = design_seed,
    bootstrap_seed = bootstrap_seed,
    status = status,
    failure_reason = reason,
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
        cell,
        task$outer,
        task$design_seed,
        task$bootstrap_seed,
        "observed_unresolved_after_rescue",
        paste0(
          "initial=", observed$initial$status, ":", observed$initial$reason,
          ";rescue=", observed$rescue$status, ":", observed$rescue$reason
        ),
        proc.time()[["elapsed"]] - started,
        observed = observed
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
      cell,
      task$outer,
      task$design_seed,
      task$bootstrap_seed,
      failure_status,
      reason,
      proc.time()[["elapsed"]] - started,
      observed = observed
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
              cell_id = cell$cell_id,
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
        cell_id = cell$cell_id,
        outer = task$outer,
        design_seed = task$design_seed,
        bootstrap_seed = task$bootstrap_seed
      ),
      problems
    )
  } else {
    problems <- empty_problem_rows()
  }

  summary <- data.frame(
    cell_id = cell$cell_id,
    cell_family = cell$cell_family,
    outer = task$outer,
    design_seed = task$design_seed,
    bootstrap_seed = task$bootstrap_seed,
    status = "ok",
    failure_reason = "",
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

N_RUNS <- nrow(cell_manifest) * OUTER_N
all_seeds <- slopeaudit:::.sa_q1_make_replicate_seeds(
  B = 2L * N_RUNS,
  seed = MASTER_SEED
)
task_manifest <- do.call(
  rbind,
  lapply(seq_len(nrow(cell_manifest)), function(cell_index) {
    start <- (cell_index - 1L) * OUTER_N
    data.frame(
      task_index = start + seq_len(OUTER_N),
      cell_id = cell_manifest$cell_id[[cell_index]],
      outer = seq_len(OUTER_N),
      stringsAsFactors = FALSE
    )
  })
)
task_manifest$design_seed <- all_seeds[task_manifest$task_index]
task_manifest$bootstrap_seed <- all_seeds[N_RUNS + task_manifest$task_index]

if (MODE == "pilot_repeat") {
  frozen_task_path <- file.path(
    FROZEN_PILOT_INPUT,
    "q1_four_design_task_manifest.csv"
  )
  frozen_config_path <- file.path(
    FROZEN_PILOT_INPUT,
    "q1_four_design_configuration.rds"
  )
  if (!file.exists(frozen_task_path) || !file.exists(frozen_config_path)) {
    stop("Missing frozen v10 pilot task/configuration input.")
  }
  frozen_tasks <- utils::read.csv(
    frozen_task_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  frozen_config <- readRDS(frozen_config_path)
  same_tasks <- isTRUE(all.equal(
    task_manifest,
    frozen_tasks,
    check.attributes = TRUE
  ))
  if (!same_tasks || MASTER_SEED != frozen_config$master_seed ||
      OUTER_N != frozen_config$outer_per_cell ||
      USABLE_B != frozen_config$usable_B ||
      ATTEMPT_CAP != frozen_config$attempt_cap ||
      ALPHA != frozen_config$alpha) {
    stop(
      paste0(
        "pilot_repeat must use the exact v10 120-task manifest, seeds, ",
        "B, cap, and alpha."
      )
    )
  }
}

utils::write.csv(
  task_manifest,
  file.path(OUT, "q1_four_design_task_manifest.csv"),
  row.names = FALSE
)

checkpoint_path <- file.path(OUT, "q1_four_design_checkpoint.rds")
outer_csv <- file.path(OUT, "q1_four_design_outer_runs.csv")
problem_csv <- file.path(OUT, "q1_four_design_rescue_attempts.csv")

if (file.exists(checkpoint_path)) {
  checkpoint <- readRDS(checkpoint_path)
  outer_results <- checkpoint$outer_results
  inner_problems <- checkpoint$inner_problems
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
  slopeaudit:::.sa_q1_exact_binomial_interval(k, n, 0.95)
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

summary_rows <- lapply(seq_len(nrow(cell_manifest)), function(index) {
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
  k_drop <- sum(valid$reject_drop)
  n_valid <- nrow(valid)
  interval <- exact_interval(k_drop, n_valid)
  issue_correlation <- safe_spearman(
    valid$top_up_attempts,
    valid$observed_LR
  )

  target_rate <- if (cell$cell_family == "null_size") {
    ALPHA
  } else {
    cell$analytic_power
  }
  interval_excludes_target <- if (n_valid) {
    target_rate < interval[[1]] || target_rate > interval[[2]]
  } else {
    NA
  }

  data.frame(
    cell_id = cell$cell_id,
    cell_family = cell$cell_family,
    n_requested = OUTER_N,
    n_completed = nrow(all_cell),
    n_valid = n_valid,
    validity_rate = n_valid / OUTER_N,
    observed_unresolved_after_rescue = sum(
      all_cell$status == "observed_unresolved_after_rescue"
    ),
    q1_errors = sum(all_cell$status == "q1_error"),
    q1_invalid_observed = sum(
      all_cell$status == "q1_invalid_observed"
    ),
    attempt_cap_failures = sum(
      all_cell$status == "attempt_cap_reached"
    ),
    support_drop = k_drop,
    support_rate_drop = if (n_valid) k_drop / n_valid else NA_real_,
    support_interval_low = interval[[1]],
    support_interval_high = interval[[2]],
    support_MCSE = if (n_valid) {
      sqrt((k_drop / n_valid) * (1 - k_drop / n_valid) / n_valid)
    } else {
      NA_real_
    },
    target_rate = target_rate,
    interval_excludes_target = interval_excludes_target,
    support_zero = if (n_valid) sum(valid$reject_zero) else NA_integer_,
    support_all_zero = if (n_valid) {
      sum(valid$reject_all_zero)
    } else {
      NA_integer_
    },
    drop_zero_decision_disagreements = if (n_valid) {
      sum(valid$reject_drop != valid$reject_zero)
    } else {
      NA_integer_
    },
    drop_all_zero_decision_disagreements = if (n_valid) {
      sum(valid$reject_drop != valid$reject_all_zero)
    } else {
      NA_integer_
    },
    total_top_up_attempts = if (n_valid) {
      sum(valid$top_up_attempts)
    } else {
      NA_integer_
    },
    maximum_top_up_attempts = if (n_valid) {
      max(valid$top_up_attempts)
    } else {
      NA_integer_
    },
    total_inner_failures = if (n_valid) {
      sum(valid$inner_failures)
    } else {
      NA_integer_
    },
    total_small_negative = if (n_valid) {
      sum(valid$small_negative)
    } else {
      NA_integer_
    },
    total_material_negative = if (n_valid) {
      sum(valid$material_negative)
    } else {
      NA_integer_
    },
    outer_runs_with_material_negative = if (n_valid) {
      sum(valid$material_negative > 0)
    } else {
      NA_integer_
    },
    observed_rescue_attempts_all_runs = sum(
      all_cell$observed_rescue_triggered,
      na.rm = TRUE
    ),
    observed_rescue_successes_all_runs = sum(
      all_cell$observed_rescue_succeeded,
      na.rm = TRUE
    ),
    inner_attempts_all_runs = sum(
      all_cell$attempts_evaluated,
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
    unresolved_after_rescue_all_runs = sum(
      all_cell$unresolved_after_rescue,
      na.rm = TRUE
    ),
    total_inner_failures_all_runs = sum(
      all_cell$inner_failures,
      na.rm = TRUE
    ),
    total_small_negative_all_runs = sum(
      all_cell$small_negative,
      na.rm = TRUE
    ),
    total_material_negative_all_runs = sum(
      all_cell$material_negative,
      na.rm = TRUE
    ),
    issue_count_observed_LR_spearman = issue_correlation[["rho"]],
    issue_count_observed_LR_n = issue_correlation[["n"]],
    analytic_R_direct = cell$analytic_R_direct,
    analytic_power = cell$analytic_power,
    historical_k = cell$historical_k,
    historical_n = cell$historical_n,
    historical_rate = cell$historical_k / cell$historical_n,
    mean_elapsed_seconds = if (n_valid) {
      mean(valid$elapsed_seconds)
    } else {
      NA_real_
    },
    provisional_gate = if (
      nrow(all_cell) != OUTER_N ||
      n_valid / OUTER_N < 0.95 ||
      any(all_cell$status == "attempt_cap_reached") ||
      any(all_cell$status == "observed_unresolved_after_rescue") ||
      any(all_cell$status == "q1_error") ||
      sum(all_cell$unresolved_after_rescue, na.rm = TRUE) > 0L ||
      isTRUE(interval_excludes_target) ||
      (n_valid > 0L && any(
        valid$reject_drop != valid$reject_all_zero
      ))
    ) {
      "DIAGNOSE"
    } else {
      "PASS"
    },
    stringsAsFactors = FALSE
  )
})

calibration_summary <- do.call(rbind, summary_rows)
utils::write.csv(
  calibration_summary,
  file.path(OUT, "q1_four_design_summary.csv"),
  row.names = FALSE
)

pooled_null_all <- outer_results[
  outer_results$cell_family == "null_size",
  ,
  drop = FALSE
]
pooled_null_valid <- pooled_null_all[
  pooled_null_all$status == "ok" &
    pooled_null_all$denominator_drop == USABLE_B,
  ,
  drop = FALSE
]
pooled_null_k <- sum(pooled_null_valid$reject_drop)
pooled_null_n <- nrow(pooled_null_valid)
pooled_null_interval <- exact_interval(pooled_null_k, pooled_null_n)
pooled_null_excludes_alpha <- if (pooled_null_n) {
  ALPHA < pooled_null_interval[[1]] || ALPHA > pooled_null_interval[[2]]
} else {
  NA
}
pooled_null_summary <- data.frame(
  analysis_id = "ALL_NULL_CELLS",
  n_requested = nrow(pooled_null_all),
  n_valid = pooled_null_n,
  validity_rate = pooled_null_n / nrow(pooled_null_all),
  false_supports = pooled_null_k,
  false_support_rate = if (pooled_null_n) {
    pooled_null_k / pooled_null_n
  } else {
    NA_real_
  },
  interval_low = pooled_null_interval[[1]],
  interval_high = pooled_null_interval[[2]],
  target_alpha = ALPHA,
  interval_excludes_alpha = pooled_null_excludes_alpha,
  attempt_cap_failures = sum(
    pooled_null_all$status == "attempt_cap_reached"
  ),
  observed_unresolved_after_rescue = sum(
    pooled_null_all$status == "observed_unresolved_after_rescue"
  ),
  unresolved_inner_rescues = sum(
    pooled_null_all$unresolved_after_rescue,
    na.rm = TRUE
  ),
  stringsAsFactors = FALSE
)
utils::write.csv(
  pooled_null_summary,
  file.path(OUT, "q1_four_design_pooled_null_summary.csv"),
  row.names = FALSE
)

all_run_diagnostics <- data.frame(
  n_outer = nrow(outer_results),
  n_valid = sum(outer_results$status == "ok"),
  attempt_cap_failures = sum(
    outer_results$status == "attempt_cap_reached"
  ),
  observed_unresolved_after_rescue = sum(
    outer_results$status == "observed_unresolved_after_rescue"
  ),
  q1_errors = sum(outer_results$status == "q1_error"),
  inner_attempts = sum(outer_results$attempts_evaluated, na.rm = TRUE),
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
  unresolved_after_rescue = sum(
    outer_results$unresolved_after_rescue,
    na.rm = TRUE
  ),
  final_inner_failures = sum(
    outer_results$inner_failures,
    na.rm = TRUE
  ),
  final_small_negative = sum(
    outer_results$small_negative,
    na.rm = TRUE
  ),
  final_material_negative = sum(
    outer_results$material_negative,
    na.rm = TRUE
  ),
  retained_rescue_or_problem_rows = nrow(inner_problems),
  stringsAsFactors = FALSE
)
utils::write.csv(
  all_run_diagnostics,
  file.path(OUT, "q1_four_design_all_run_diagnostics.csv"),
  row.names = FALSE
)

if (MODE == "pilot_repeat") {
  frozen_outer <- utils::read.csv(
    file.path(FROZEN_PILOT_INPUT, "q1_four_design_outer_runs.csv"),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  baseline <- frozen_outer[, c(
    "cell_id", "outer", "design_seed", "bootstrap_seed", "status",
    "observed_LR", "p_drop", "reject_drop", "attempts_evaluated",
    "inner_failures", "small_negative", "material_negative"
  )]
  names(baseline)[5:ncol(baseline)] <- paste0(
    "v10_",
    names(baseline)[5:ncol(baseline)]
  )
  comparison <- merge(
    outer_results,
    baseline,
    by = c("cell_id", "outer", "design_seed", "bootstrap_seed"),
    all = TRUE,
    sort = TRUE
  )
  comparison$p_drop_difference_v12_minus_v10 <-
    comparison$p_drop - comparison$v10_p_drop
  comparison$decision_changed <-
    comparison$reject_drop != comparison$v10_reject_drop
  utils::write.csv(
    comparison,
    file.path(OUT, "q1_four_design_v10_v12_comparison.csv"),
    row.names = FALSE
  )
}

cell_intervals_ok <- all(
  !is.na(calibration_summary$interval_excludes_target) &
    !calibration_summary$interval_excludes_target
)
global_gate <- data.frame(
  check = c(
    "all_tasks_completed",
    "at_least_95_percent_valid_each_cell",
    "no_attempt_cap_or_observed_rescue_failures",
    "no_final_inner_failures_after_rescue",
    "no_final_negative_LRs_after_rescue",
    "no_primary_sensitivity_decision_disagreement",
    "all_cell_two_sided_intervals_include_targets",
    "pooled_null_two_sided_interval_includes_alpha"
  ),
  value = c(
    paste0(nrow(outer_results), "/", N_RUNS),
    paste0(min(calibration_summary$validity_rate), " minimum"),
    paste0(
      all_run_diagnostics$attempt_cap_failures, " caps; ",
      all_run_diagnostics$observed_unresolved_after_rescue,
      " observed unresolved"
    ),
    paste0(
      all_run_diagnostics$final_inner_failures,
      " final failures; ",
      all_run_diagnostics$unresolved_after_rescue,
      " unresolved bobyqa"
    ),
    paste0(
      all_run_diagnostics$final_small_negative,
      " small; ",
      all_run_diagnostics$final_material_negative,
      " material"
    ),
    as.character(sum(
      calibration_summary$drop_all_zero_decision_disagreements,
      na.rm = TRUE
    )),
    paste0(
      sum(calibration_summary$interval_excludes_target, na.rm = TRUE),
      " excluded"
    ),
    paste0(
      pooled_null_summary$interval_low,
      " to ",
      pooled_null_summary$interval_high,
      "; target ",
      ALPHA
    )
  ),
  pass = c(
    nrow(outer_results) == N_RUNS,
    all(calibration_summary$validity_rate >= 0.95),
    all_run_diagnostics$attempt_cap_failures == 0L &&
      all_run_diagnostics$observed_unresolved_after_rescue == 0L &&
      all_run_diagnostics$q1_errors == 0L,
    all_run_diagnostics$final_inner_failures == 0L,
    all_run_diagnostics$final_small_negative == 0L &&
      all_run_diagnostics$final_material_negative == 0L,
    sum(
      calibration_summary$drop_all_zero_decision_disagreements,
      na.rm = TRUE
    ) == 0L,
    cell_intervals_ok,
    isFALSE(pooled_null_excludes_alpha)
  ),
  stringsAsFactors = FALSE
)
utils::write.csv(
  global_gate,
  file.path(OUT, "q1_four_design_global_gate.csv"),
  row.names = FALSE
)

correlation_rows <- lapply(
  c("ALL_NULL_CELLS", cell_manifest$cell_id),
  function(cell_id) {
    subset <- if (cell_id == "ALL_NULL_CELLS") {
      outer_results[
        outer_results$cell_family == "null_size" &
          outer_results$status == "ok",
        ,
        drop = FALSE
      ]
    } else {
      outer_results[
        outer_results$cell_id == cell_id &
          outer_results$status == "ok",
        ,
        drop = FALSE
      ]
    }
    correlation <- safe_spearman(
      subset$top_up_attempts,
      subset$observed_LR
    )
    data.frame(
      cell_id = cell_id,
      n = correlation[["n"]],
      spearman_rho = correlation[["rho"]],
      interpretation = paste0(
        "Indirect diagnostic only: association between the number of ",
        "excluded/top-up inner attempts and the outer observed LR does not ",
        "prove that failures are independent of their unobserved LR values."
      ),
      stringsAsFactors = FALSE
    )
  }
)
utils::write.csv(
  do.call(rbind, correlation_rows),
  file.path(OUT, "q1_four_design_failure_association.csv"),
  row.names = FALSE
)

writeLines(
  capture.output(sessionInfo()),
  file.path(OUT, "sessionInfo_q1_four_design_v12.txt")
)

print(calibration_summary)
print(pooled_null_summary)
print(global_gate)
if (all(global_gate$pass)) {
  message("PASS: v12 calibration run passed every declared global gate.")
} else {
  message("DIAGNOSE: v12 completed, but one or more global gates failed.")
}
