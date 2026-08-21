#!/usr/bin/env Rscript

# V13.3.1 focused power diagnostic for the two iid-residual Table-1 rows.
#
# This is not a nested parametric-bootstrap experiment.  It simulates one
# independent null ensemble per design, uses that ensemble to calibrate each
# statistic to empirical size .05, and evaluates all three statistics on the
# same alternative datasets.  The result isolates the power of the test
# statistic/model specification from inner-bootstrap Monte Carlo noise.

options(stringsAsFactors = FALSE)

script_argument <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (!length(script_argument)) {
  stop("Run this file with Rscript.", call. = FALSE)
}
SCRIPT_PATH <- normalizePath(
  sub("^--file=", "", script_argument[[1]]),
  winslash = "/",
  mustWork = TRUE
)
PACKAGE_ROOT <- normalizePath(
  file.path(dirname(SCRIPT_PATH), "..", ".."),
  winslash = "/",
  mustWork = TRUE
)

suppressPackageStartupMessages({
  library(slopeaudit)
  library(lme4)
})

namespace <- asNamespace("slopeaudit")
required_internal <- c(
  ".sa_q1_calibration_rescue_controls",
  ".sa_q1_fit_observed_pair_rescue",
  ".sa_q1_fit_diagnostics",
  ".sa_q1_classify_lr"
)
missing_internal <- required_internal[
  !vapply(
    required_internal,
    exists,
    logical(1),
    envir = namespace,
    inherits = FALSE
  )
]
if (length(missing_internal)) {
  stop(
    paste0(
      "The installed slopeaudit package predates v12/v13. Missing: ",
      paste(missing_internal, collapse = ", "),
      ". Run `R CMD INSTALL .` from the unpacked package root."
    ),
    call. = FALSE
  )
}

HELPER_PATH <- file.path(
  PACKAGE_ROOT,
  "validation",
  "extended",
  "q1_table1_power_helpers.R"
)
if (!file.exists(HELPER_PATH)) {
  stop("Missing v13 power helper file: ", HELPER_PATH, call. = FALSE)
}
source(HELPER_PATH, local = FALSE)
q1p_run_self_tests()

mode_value <- Sys.getenv("SLOPEAUDIT_Q1_POWER_MODE", "repair")
MODE <- match.arg(mode_value, c("smoke", "repair", "extend"))

workers_value <- suppressWarnings(as.integer(
  Sys.getenv("SLOPEAUDIT_Q1_POWER_WORKERS", "1")
))
if (length(workers_value) != 1L || is.na(workers_value) || workers_value < 1L) {
  stop("SLOPEAUDIT_Q1_POWER_WORKERS must be one positive integer.",
       call. = FALSE)
}
WORKERS <- workers_value
if (.Platform$OS.type == "windows" && WORKERS > 1L) {
  stop("This script supports more than one worker only through Unix forking.",
       call. = FALSE)
}

OUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_POWER_OUT",
  file.path(
    PACKAGE_ROOT,
    "validation",
    "extended",
    "results",
    paste0("q1_table1_power_", MODE, "_v13_3_1")
  )
)
if (!grepl("^/", OUT)) {
  OUT <- file.path(PACKAGE_ROOT, OUT)
}
OUT <- normalizePath(OUT, winslash = "/", mustWork = FALSE)
dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

ALPHA <- 0.05
M_NULL <- if (identical(MODE, "smoke")) {
  20L
} else if (identical(MODE, "repair")) {
  1000L
} else {
  10000L
}
M_ALTERNATIVE <- if (identical(MODE, "smoke")) 20L else 1000L
LEGACY_MASTER_SEED <- 13082026L
NULL_TOPUP_MASTER_SEED <- 13082027L
BATCH_SIZE <- max(1L, 5L * WORKERS)

controls <- slopeaudit:::.sa_q1_calibration_rescue_controls()
slopeaudit:::.sa_q1_validate_calibration_rescue_controls(
  controls$initial,
  controls$rescue
)

schedules <- list(
  three_waves_two_years = q1p_make_balanced_schedule(
    100L,
    c(0, 1, 2)
  ),
  five_waves_five_years = q1p_make_balanced_schedule(
    100L,
    seq(0, 5, length.out = 5)
  )
)

included_designs <- data.frame(
  design_id = c("three_waves_two_years", "five_waves_five_years"),
  table_label = c("3 waves / 2 years", "5 waves / 5 years"),
  n_participants = c(100L, 100L),
  wave_times = c("0;1;2", "0;1.25;2.5;3.75;5"),
  random_intercept_variance = c(0.5, 0.5),
  random_slope_variance = c(0.01, 0.01),
  residual_variance = c(1, 1),
  residual_model = c("iid Gaussian", "iid Gaussian"),
  diagnostic_status = c("included", "included"),
  exclusion_reason = c("", ""),
  stringsAsFactors = FALSE
)

analytic <- t(vapply(seq_len(nrow(included_designs)), function(index) {
  design <- included_designs[index, ]
  times <- as.numeric(strsplit(design$wave_times, ";", fixed = TRUE)[[1]])
  q1p_analytic_values(
    n_participants = design$n_participants,
    times = times,
    random_slope_variance = design$random_slope_variance,
    residual_variance = design$residual_variance,
    alpha = ALPHA
  )
}, numeric(6)))
included_designs <- cbind(included_designs, as.data.frame(analytic))

excluded_designs <- data.frame(
  design_id = c("ten_daily_observations", "thirty_day_ema"),
  table_label = c("10 daily observations", "30-day EMA"),
  n_participants = c(NA_integer_, NA_integer_),
  wave_times = c(NA_character_, NA_character_),
  random_intercept_variance = c(NA_real_, NA_real_),
  random_slope_variance = c(NA_real_, NA_real_),
  residual_variance = c(NA_real_, NA_real_),
  residual_model = c("AR(1)", "AR(1)"),
  diagnostic_status = c("excluded", "excluded"),
  exclusion_reason = rep(
    paste0(
      "The manuscript row assumes AR(1) residuals, but the current ",
      "lme4 Question-1 backend fits iid residuals. Including this row ",
      "would confound test power with residual-model misspecification."
    ),
    2L
  ),
  information = c(NA_real_, NA_real_),
  direct_error_variance = c(NA_real_, NA_real_),
  R_direct = c(NA_real_, NA_real_),
  analytic_direct_power = c(NA_real_, NA_real_),
  df = c(NA_real_, NA_real_),
  theoretical_direct_critical = c(NA_real_, NA_real_),
  stringsAsFactors = FALSE
)
design_manifest <- rbind(included_designs, excluded_designs)

make_scenario <- function(
    scenario_id,
    design_id,
    truth_role,
    slope_variance,
    rho,
    requested,
    coupling,
    interpretation) {
  data.frame(
    scenario_id = scenario_id,
    design_id = design_id,
    truth_role = truth_role,
    random_intercept_variance = 0.5,
    random_slope_variance = slope_variance,
    intercept_slope_correlation = rho,
    residual_variance = 1,
    fixed_intercept = 0,
    fixed_slope = 0.1,
    requested = as.integer(requested),
    coupling = coupling,
    interpretation = interpretation,
    stringsAsFactors = FALSE
  )
}

legacy_scenario_manifest <- do.call(rbind, list(
  make_scenario(
    "three_waves_two_years_null", "three_waves_two_years", "null",
    0, 0, 1000L, "legacy_hold_intercept", "Empirical null calibration"
  ),
  make_scenario(
    "three_waves_two_years_alt_rho0", "three_waves_two_years",
    "alternative", 0.01, 0, 1000L, "legacy_hold_intercept",
    "Primary Table-1 alternative; true intercept-slope covariance is zero"
  ),
  make_scenario(
    "five_waves_five_years_null", "five_waves_five_years", "null",
    0, 0, 1000L, "legacy_hold_intercept", "Empirical null calibration"
  ),
  make_scenario(
    "five_waves_five_years_alt_rho0", "five_waves_five_years",
    "alternative", 0.01, 0, 1000L, "legacy_hold_intercept",
    "Primary Table-1 alternative; true intercept-slope covariance is zero"
  ),
  make_scenario(
    "five_waves_five_years_alt_rho0p5", "five_waves_five_years",
    "sensitivity", 0.01, 0.5, 1000L, "legacy_hold_intercept",
    paste0(
      "V13.2 independent sensitivity retained for replication; rho=.5; ",
      "not paired to the rho=0 datasets"
    )
  )
))

paired_scenarios <- do.call(rbind, lapply(
  c("three_waves_two_years", "five_waves_five_years"),
  function(design_id) make_scenario(
    paste0(design_id, "_alt_rho0p5_paired"),
    design_id,
    "paired_sensitivity",
    0.01,
    0.5,
    M_ALTERNATIVE,
    "paired_hold_slope",
    paste0(
      "Paired rho=.5 sensitivity: same slope effects and residual draws as ",
      "the design's rho=0 alternative; only the intercept construction changes"
    )
  )
))

if (identical(MODE, "smoke")) {
  scenario_manifest <- rbind(
    transform(
      legacy_scenario_manifest[
        legacy_scenario_manifest$truth_role %in% c("null", "alternative"),
      ],
      requested = M_ALTERNATIVE
    ),
    paired_scenarios
  )
} else if (identical(MODE, "repair")) {
  scenario_manifest <- legacy_scenario_manifest
} else {
  scenario_manifest <- rbind(
    transform(
      legacy_scenario_manifest,
      requested = ifelse(truth_role == "null", M_NULL, M_ALTERNATIVE)
    ),
    paired_scenarios
  )
}
scenario_manifest$intercept_slope_covariance <-
  scenario_manifest$intercept_slope_correlation *
  sqrt(
    scenario_manifest$random_intercept_variance *
      scenario_manifest$random_slope_variance
  )

make_tasks <- function(
    scenario,
    replicates,
    seeds,
    task_ids,
    task_origin,
    pair_id = "") {
  stopifnot(length(replicates) == length(seeds),
            length(replicates) == length(task_ids))
  data.frame(
    task_id = task_ids,
    scenario_id = scenario$scenario_id,
    design_id = scenario$design_id,
    truth_role = scenario$truth_role,
    replicate = as.integer(replicates),
    seed = as.integer(seeds),
    coupling = scenario$coupling,
    task_origin = task_origin,
    pair_id = if (length(pair_id) == 1L) rep(pair_id, length(replicates)) else
      pair_id,
    stringsAsFactors = FALSE
  )
}

legacy_seeds <- slopeaudit:::.sa_q1_make_replicate_seeds(
  5000L,
  LEGACY_MASTER_SEED
)
legacy_tasks <- do.call(rbind, lapply(
  seq_len(nrow(legacy_scenario_manifest)),
  function(index) {
    scenario <- legacy_scenario_manifest[index, ]
    offset <- (index - 1L) * 1000L
    reps <- seq_len(1000L)
    make_tasks(
      scenario,
      reps,
      legacy_seeds[offset + reps],
      sprintf("task_%05d", offset + reps),
      "v13_2_legacy",
      if (scenario$truth_role == "alternative") {
        paste0(scenario$design_id, "::", sprintf("%04d", reps))
      } else ""
    )
  }
))

if (identical(MODE, "repair")) {
  task_manifest <- legacy_tasks
} else if (identical(MODE, "smoke")) {
  base_scenarios <- scenario_manifest[
    scenario_manifest$truth_role %in% c("null", "alternative"),
  ]
  unique_seeds <- slopeaudit:::.sa_q1_make_replicate_seeds(
    nrow(base_scenarios) * M_ALTERNATIVE,
    LEGACY_MASTER_SEED + 2L
  )
  task_manifest <- do.call(rbind, lapply(
    seq_len(nrow(base_scenarios)),
    function(index) {
      scenario <- base_scenarios[index, ]
      reps <- seq_len(M_ALTERNATIVE)
      offset <- (index - 1L) * M_ALTERNATIVE
      make_tasks(
        scenario, reps, unique_seeds[offset + reps],
        sprintf("smoke_%s_%03d", scenario$scenario_id, reps),
        "v13_3_1_smoke",
        if (scenario$truth_role == "alternative") {
          paste0(scenario$design_id, "::", sprintf("%04d", reps))
        } else ""
      )
    }
  ))
  for (index in seq_len(nrow(paired_scenarios))) {
    scenario <- paired_scenarios[index, ]
    partners <- task_manifest[
      task_manifest$scenario_id ==
        paste0(scenario$design_id, "_alt_rho0"),
    ]
    task_manifest <- rbind(task_manifest, make_tasks(
      scenario,
      partners$replicate,
      partners$seed,
      sprintf("smoke_%s_%03d", scenario$scenario_id, partners$replicate),
      "v13_3_1_paired_sensitivity",
      partners$pair_id
    ))
  }
} else {
  task_manifest <- legacy_tasks
  n_topup <- 2L * (M_NULL - 1000L)
  topup_candidates <- slopeaudit:::.sa_q1_make_replicate_seeds(
    n_topup + length(legacy_seeds),
    NULL_TOPUP_MASTER_SEED
  )
  topup_seeds <- topup_candidates[!topup_candidates %in% legacy_seeds]
  if (length(topup_seeds) < n_topup) {
    stop("Could not construct the nonoverlapping null top-up seed set.",
         call. = FALSE)
  }
  topup_seeds <- topup_seeds[seq_len(n_topup)]
  cursor <- 0L
  for (design_id in included_designs$design_id) {
    scenario <- scenario_manifest[
      scenario_manifest$scenario_id == paste0(design_id, "_null"),
    ]
    reps <- 1001L:M_NULL
    indices <- cursor + seq_along(reps)
    cursor <- cursor + length(reps)
    task_manifest <- rbind(task_manifest, make_tasks(
      scenario,
      reps,
      topup_seeds[indices],
      sprintf("null_topup_%s_%05d", design_id, reps),
      "v13_3_1_null_topup"
    ))
  }
  for (index in seq_len(nrow(paired_scenarios))) {
    scenario <- paired_scenarios[index, ]
    partners <- legacy_tasks[
      legacy_tasks$scenario_id == paste0(scenario$design_id, "_alt_rho0"),
    ]
    task_manifest <- rbind(task_manifest, make_tasks(
      scenario,
      partners$replicate,
      partners$seed,
      sprintf("paired_%s_%04d", scenario$design_id, partners$replicate),
      "v13_3_1_paired_sensitivity",
      partners$pair_id
    ))
  }
}

if (anyDuplicated(task_manifest$task_id)) {
  stop("Task IDs must be globally unique.", call. = FALSE)
}
seed_groups <- split(task_manifest, task_manifest$seed)
seed_reuse_ok <- vapply(seed_groups, function(group) {
  nrow(group) == 1L ||
    (nrow(group) == 2L && length(unique(group$pair_id)) == 1L &&
      nzchar(group$pair_id[[1]]) &&
      setequal(group$truth_role, c("alternative", "paired_sensitivity")))
}, logical(1))
if (!all(seed_reuse_ok)) {
  stop("Seeds may repeat only within declared rho=0/rho=.5 pairs.",
       call. = FALSE)
}
scenario_task_counts <- table(factor(
  task_manifest$scenario_id,
  levels = scenario_manifest$scenario_id
))
if (!identical(
  as.integer(scenario_task_counts),
  as.integer(scenario_manifest$requested)
)) {
  stop("Task counts do not match the scenario manifest.", call. = FALSE)
}

scientific_config <- list(
  version = "q1_table1_power_diagnostic_v13_3_1",
  mode = MODE,
  alpha = ALPHA,
  null_replicates_per_design = M_NULL,
  alternative_replicates_per_scenario = M_ALTERNATIVE,
  legacy_master_seed = LEGACY_MASTER_SEED,
  null_topup_master_seed = NULL_TOPUP_MASTER_SEED,
  empirical_critical_rule = paste0(
    "upper-tail conservative order statistic; k=floor(alpha*n); reject ",
    "strictly above the cutoff; report realized size and boundary ties"
  ),
  pairing_rule = paste0(
    "one shared null ensemble per design; calibrate each method on its own ",
    "finite statistics; pairwise comparisons use the relevant intersection"
  ),
  optimizer_rule = paste0(
    "paired nloptwrap; if pair unusable, paired bobyqa rescue; only the ",
    "constrained diagnostic may floor an otherwise-valid rescued ",
    "small-negative LR, or a narrowly adjudicated optimizer-code boundary ",
    "LR, to numerical zero"
  ),
  constrained_numerical_zero_rule = paste0(
    "initial negatives always trigger rescue; accept zero only when the ",
    "rescued constrained pair passed all validity and structure checks and ",
    "its LR is within the prespecified likelihood-scale tolerance; the ",
    "optimizer-code fallback additionally requires exactly one nonzero code, ",
    "no other convergence message, no likelihood deterioration, verified ",
    "model structure, and the fitted slope component at its boundary"
  ),
  power_estimand = paste0(
    "DGP-empirically-calibrated statistic power, not nested plug-in ",
    "parametric-bootstrap decision power"
  ),
  design_manifest = design_manifest,
  scenario_manifest = scenario_manifest,
  task_manifest = task_manifest,
  initial_control = controls$initial,
  rescue_control = controls$rescue,
  helper_md5 = unname(tools::md5sum(HELPER_PATH))
)

config_path <- file.path(OUT, "q1_table1_power_configuration.rds")
checkpoint_path <- file.path(OUT, "q1_table1_power_checkpoint.rds")
if (file.exists(config_path)) {
  prior <- readRDS(config_path)
  if (!identical(prior, scientific_config)) {
    stop("Existing output configuration differs from this invocation.",
         call. = FALSE)
  }
} else {
  saveRDS(scientific_config, config_path, version = 3)
}

utils::write.csv(
  design_manifest,
  file.path(OUT, "q1_table1_power_design_manifest.csv"),
  row.names = FALSE
)
utils::write.csv(
  scenario_manifest,
  file.path(OUT, "q1_table1_power_scenario_manifest.csv"),
  row.names = FALSE
)
utils::write.csv(
  task_manifest,
  file.path(OUT, "q1_table1_power_task_manifest.csv"),
  row.names = FALSE
)

empty_pair_fields <- function(prefix, error_message) {
  values <- list(
    status = "error",
    source = "error",
    statistic = NA_real_,
    statistic_raw = NA_real_,
    numerical_zero_floored = FALSE,
    numerical_zero_rule = "",
    boundary_optimizer_code_adjudicated = FALSE,
    numerical_zero_floor_reason = "",
    numerical_zero_floor_tolerance = NA_real_,
    rescue_triggered = NA,
    rescue_succeeded = NA,
    initial_status = "error",
    initial_reason = error_message,
    initial_LR_raw = NA_real_,
    rescue_status = "not_run",
    rescue_reason = "",
    rescue_LR_raw = NA_real_,
    chosen_logLik_full = NA_real_,
    chosen_logLik_null = NA_real_,
    chosen_tolerance = NA_real_,
    chosen_structure_ok = NA,
    chosen_full_optimizer_code = "",
    chosen_null_optimizer_code = "",
    chosen_full_convergence_messages = "",
    chosen_null_convergence_messages = "",
    chosen_full_non_singular_convergence_messages = "",
    chosen_null_non_singular_convergence_messages = "",
    chosen_full_slope_variance = NA_real_,
    chosen_full_slope_relative_sd = NA_real_,
    chosen_full_slope_at_boundary = NA,
    chosen_full_singular = NA,
    chosen_null_singular = NA,
    chosen_warnings = "",
    chosen_messages = ""
  )
  names(values) <- paste0(prefix, "_", names(values))
  values
}

failed_task_row <- function(task, error_message) {
  core <- list(
    task_id = task$task_id,
    design_id = task$design_id,
    scenario_id = task$scenario_id,
    truth_role = task$truth_role,
    replicate = task$replicate,
    seed = task$seed,
    coupling = task$coupling,
    task_origin = task$task_origin,
    pair_id = task$pair_id,
    status = "error",
    direct_statistic = NA_real_,
    direct_df = NA_integer_,
    direct_error_variance = NA_real_,
    elapsed_seconds = NA_real_
  )
  as.data.frame(
    c(
      core,
      empty_pair_fields("constrained", error_message),
      empty_pair_fields("unstructured", error_message)
    ),
    stringsAsFactors = FALSE
  )
}

run_task <- function(task_index) {
  task <- task_manifest[task_index, , drop = FALSE]
  scenario <- scenario_manifest[
    scenario_manifest$scenario_id == task$scenario_id,
    ,
    drop = FALSE
  ]
  truth <- as.list(scenario[, c(
    "random_intercept_variance",
    "random_slope_variance",
    "intercept_slope_correlation",
    "residual_variance",
    "fixed_intercept",
    "fixed_slope"
  )])

  tryCatch(
    q1p_run_dataset(
      task = task,
      schedule = schedules[[task$design_id]],
      truth = truth,
      controls = controls,
      coupling = task$coupling
    ),
    error = function(error) failed_task_row(task, conditionMessage(error))
  )
}

align_result_schema <- function(results) {
  template <- failed_task_row(task_manifest[1, , drop = FALSE], "template")
  for (column in setdiff(names(template), names(results))) {
    value <- template[[column]][[1]]
    results[[column]] <- rep(value, nrow(results))
  }
  extra <- setdiff(names(results), names(template))
  results[, c(names(template), extra), drop = FALSE]
}

validate_legacy_identity <- function(prior_results) {
  required <- c(
    "task_id", "scenario_id", "design_id", "truth_role", "replicate",
    "seed", "status", "constrained_status", "constrained_source",
    "constrained_rescue_status", "constrained_rescue_reason",
    "constrained_rescue_LR_raw", "unstructured_status"
  )
  missing <- setdiff(required, names(prior_results))
  if (length(missing)) {
    stop("Imported checkpoint lacks: ", paste(missing, collapse = ", "),
         call. = FALSE)
  }
  if (nrow(prior_results) != 5000L || anyDuplicated(prior_results$task_id)) {
    stop("Imported checkpoint must contain the 5,000 frozen v13.2 tasks.",
         call. = FALSE)
  }
  observed <- prior_results[match(legacy_tasks$task_id, prior_results$task_id), ]
  rownames(observed) <- NULL
  frozen_identity <- legacy_tasks
  rownames(frozen_identity) <- NULL
  identity_fields <- c(
    "task_id", "scenario_id", "design_id", "truth_role", "replicate", "seed"
  )
  if (anyNA(observed$task_id) || !identical(
    observed[, identity_fields, drop = FALSE],
    frozen_identity[, identity_fields, drop = FALSE]
  )) {
    stop("Imported checkpoint does not match the frozen v13.2 IDs and seeds.",
         call. = FALSE)
  }
  observed
}

import_v13_2_for_repair <- function(prior_checkpoint) {
  prior_config <- prior_checkpoint$scientific_config
  if (!is.list(prior_config) ||
      !identical(prior_config$version, "q1_table1_power_diagnostic_v13_2") ||
      !identical(prior_config$mode, "full") ||
      !identical(prior_config$alpha, ALPHA) ||
      !identical(prior_config$master_seed, LEGACY_MASTER_SEED) ||
      !identical(prior_config$null_replicates_per_design, 1000L) ||
      !identical(prior_config$alternative_replicates_per_scenario, 1000L) ||
      !identical(prior_config$initial_control, controls$initial) ||
      !identical(prior_config$rescue_control, controls$rescue)) {
    stop("Repair mode requires the frozen v13.2 full checkpoint.",
         call. = FALSE)
  }
  prior <- validate_legacy_identity(prior_checkpoint$results)
  refit <- prior$constrained_status == "unresolved_after_bobyqa" &
    prior$constrained_source == "unresolved_after_bobyqa" &
    prior$constrained_rescue_status == "failure" &
    prior$constrained_rescue_reason %in% c(
      "full:optimizer_nonzero_code", "null:optimizer_nonzero_code"
    ) &
    is.finite(prior$constrained_rescue_LR_raw) &
    abs(prior$constrained_rescue_LR_raw) <= 1e-6 &
    prior$unstructured_status == "ok"
  other_nonusable <- prior$status != "ok" & !refit
  if (sum(refit) != 4L || any(other_nonusable)) {
    stop(
      "The v13.2 import must contain exactly the four audited constrained ",
      "optimizer-code boundary rows and no other unusable rows.",
      call. = FALSE
    )
  }
  retained <- align_result_schema(prior[!refit, , drop = FALSE])
  legacy_index <- match(retained$task_id, legacy_tasks$task_id)
  retained$coupling <- legacy_tasks$coupling[legacy_index]
  retained$task_origin <- legacy_tasks$task_origin[legacy_index]
  retained$pair_id <- legacy_tasks$pair_id[legacy_index]
  retained$constrained_numerical_zero_rule[
    retained$constrained_numerical_zero_floored %in% TRUE
  ] <- "rescued_small_negative"
  retained$unstructured_numerical_zero_rule[
    retained$unstructured_numerical_zero_floored %in% TRUE
  ] <- "rescued_small_negative"
  list(
    results = retained,
    refit_reference = prior[refit, , drop = FALSE],
    refit_task_ids = prior$task_id[refit],
    imported_rows = nrow(prior),
    retained_rows = nrow(retained),
    source_version = prior_config$version
  )
}

import_v13_3_1_repair_for_extension <- function(prior_checkpoint) {
  prior_config <- prior_checkpoint$scientific_config
  if (!is.list(prior_config) ||
      !identical(
        prior_config$version,
        "q1_table1_power_diagnostic_v13_3_1"
      ) ||
      !identical(prior_config$mode, "repair") ||
      !identical(prior_config$alpha, ALPHA) ||
      !identical(prior_config$initial_control, controls$initial) ||
      !identical(prior_config$rescue_control, controls$rescue)) {
    stop("Extend mode requires a completed v13.3.1 repair checkpoint.",
         call. = FALSE)
  }
  prior <- validate_legacy_identity(prior_checkpoint$results)
  if (!all(prior$status == "ok")) {
    stop("The v13.3.1 repair checkpoint is not operationally complete.",
         call. = FALSE)
  }
  retained <- align_result_schema(prior)
  legacy_index <- match(retained$task_id, legacy_tasks$task_id)
  retained$coupling <- legacy_tasks$coupling[legacy_index]
  retained$task_origin <- legacy_tasks$task_origin[legacy_index]
  retained$pair_id <- legacy_tasks$pair_id[legacy_index]
  list(
    results = retained,
    refit_reference = NULL,
    refit_task_ids = character(),
    imported_rows = nrow(prior),
    retained_rows = nrow(prior),
    source_version = prior_config$version
  )
}

IMPORT_CHECKPOINT <- Sys.getenv(
  "SLOPEAUDIT_Q1_POWER_IMPORT_CHECKPOINT",
  ""
)
if (nzchar(IMPORT_CHECKPOINT) && !grepl("^/", IMPORT_CHECKPOINT)) {
  IMPORT_CHECKPOINT <- file.path(PACKAGE_ROOT, IMPORT_CHECKPOINT)
}
if (nzchar(IMPORT_CHECKPOINT)) {
  IMPORT_CHECKPOINT <- normalizePath(
    IMPORT_CHECKPOINT,
    winslash = "/",
    mustWork = TRUE
  )
} else if (!identical(MODE, "smoke")) {
  stop("Repair and extend modes require SLOPEAUDIT_Q1_POWER_IMPORT_CHECKPOINT.",
       call. = FALSE)
}

import_provenance <- NULL
refit_reference <- NULL
refit_reference_path <- file.path(
  OUT,
  "q1_table1_power_targeted_refit_reference.rds"
)
import_provenance_path <- file.path(
  OUT,
  "q1_table1_power_import_provenance.csv"
)
if (file.exists(checkpoint_path)) {
  checkpoint <- readRDS(checkpoint_path)
  if (!identical(checkpoint$scientific_config, scientific_config)) {
    stop("Checkpoint configuration does not match this invocation.",
         call. = FALSE)
  }
  results <- checkpoint$results
  if (file.exists(import_provenance_path)) {
    import_provenance <- utils::read.csv(
      import_provenance_path,
      stringsAsFactors = FALSE
    )
  }
  if (file.exists(refit_reference_path)) {
    refit_reference <- readRDS(refit_reference_path)
  }
} else if (nzchar(IMPORT_CHECKPOINT)) {
  imported <- if (identical(MODE, "repair")) {
    import_v13_2_for_repair(readRDS(IMPORT_CHECKPOINT))
  } else if (identical(MODE, "extend")) {
    import_v13_3_1_repair_for_extension(readRDS(IMPORT_CHECKPOINT))
  } else {
    stop("Smoke mode does not import a checkpoint.", call. = FALSE)
  }
  results <- imported$results
  refit_reference <- imported$refit_reference
  if (!is.null(refit_reference)) {
    saveRDS(refit_reference, refit_reference_path, version = 3)
  }
  import_provenance <- data.frame(
    source_checkpoint = IMPORT_CHECKPOINT,
    source_version = imported$source_version,
    imported_rows = imported$imported_rows,
    retained_rows = imported$retained_rows,
    refit_rows = length(imported$refit_task_ids),
    refit_task_ids = paste(imported$refit_task_ids, collapse = ";"),
    import_rule = paste0(
      if (identical(MODE, "repair")) {
        paste0(
          "Retain 4,996 usable v13.2 full rows; deterministically regenerate ",
          "the four audited constrained optimizer-code boundary rows"
        )
      } else {
        paste0(
          "Retain all 5,000 completed v13.3.1 repair rows; add frozen null ",
          "top-ups and paired rho sensitivities without rerunning alternatives"
        )
      }
    ),
    stringsAsFactors = FALSE
  )
  utils::write.csv(
    import_provenance,
    import_provenance_path,
    row.names = FALSE
  )
} else {
  results <- NULL
}

completed_ids <- if (is.null(results)) character() else results$task_id
pending <- which(!task_manifest$task_id %in% completed_ids)

save_checkpoint <- function() {
  saveRDS(
    list(scientific_config = scientific_config, results = results),
    checkpoint_path,
    version = 3
  )
}

if (length(pending)) {
  progress <- utils::txtProgressBar(
    min = 0,
    max = length(pending),
    initial = 0,
    style = 3
  )
  completed_now <- 0L
  batches <- split(pending, ceiling(seq_along(pending) / BATCH_SIZE))

  for (batch in batches) {
    batch_results <- if (WORKERS == 1L) {
      lapply(batch, run_task)
    } else {
      parallel::mclapply(
        batch,
        run_task,
        mc.cores = WORKERS,
        mc.preschedule = FALSE,
        mc.set.seed = FALSE
      )
    }
    batch_frame <- do.call(rbind, batch_results)
    results <- if (is.null(results)) {
      batch_frame
    } else {
      rbind(results, batch_frame)
    }
    results <- results[order(results$task_id), , drop = FALSE]
    save_checkpoint()

    completed_now <- completed_now + length(batch)
    utils::setTxtProgressBar(progress, completed_now)
  }
  close(progress)
}

results <- results[order(results$task_id), , drop = FALSE]
rownames(results) <- NULL
utils::write.csv(
  results,
  file.path(OUT, "q1_table1_power_task_results.csv"),
  row.names = FALSE
)

targeted_refit_comparison <- if (!is.null(refit_reference)) {
  regenerated <- results[
    match(refit_reference$task_id, results$task_id),
    ,
    drop = FALSE
  ]
  numeric_close <- function(a, b) {
    is.finite(a) & is.finite(b) &
      abs(a - b) <= 1e-10 * pmax(1, abs(a), abs(b))
  }
  data.frame(
    task_id = refit_reference$task_id,
    seed = refit_reference$seed,
    direct_statistic_before = refit_reference$direct_statistic,
    direct_statistic_after = regenerated$direct_statistic,
    direct_statistic_reproduced = numeric_close(
      refit_reference$direct_statistic,
      regenerated$direct_statistic
    ),
    unstructured_status_before = refit_reference$unstructured_status,
    unstructured_status_after = regenerated$unstructured_status,
    unstructured_source_before = refit_reference$unstructured_source,
    unstructured_source_after = regenerated$unstructured_source,
    unstructured_statistic_before = refit_reference$unstructured_statistic,
    unstructured_statistic_after = regenerated$unstructured_statistic,
    unstructured_science_reproduced =
      refit_reference$unstructured_status == regenerated$unstructured_status &
      refit_reference$unstructured_source == regenerated$unstructured_source &
      numeric_close(
        refit_reference$unstructured_statistic,
        regenerated$unstructured_statistic
      ),
    constrained_status_after = regenerated$constrained_status,
    constrained_source_after = regenerated$constrained_source,
    constrained_statistic_after = regenerated$constrained_statistic,
    stringsAsFactors = FALSE
  )
} else {
  data.frame(
    task_id = character(), seed = integer(),
    direct_statistic_before = numeric(), direct_statistic_after = numeric(),
    direct_statistic_reproduced = logical(),
    unstructured_status_before = character(),
    unstructured_status_after = character(),
    unstructured_source_before = character(),
    unstructured_source_after = character(),
    unstructured_statistic_before = numeric(),
    unstructured_statistic_after = numeric(),
    unstructured_science_reproduced = logical(),
    constrained_status_after = character(),
    constrained_source_after = character(),
    constrained_statistic_after = numeric(),
    stringsAsFactors = FALSE
  )
}
utils::write.csv(
  targeted_refit_comparison,
  file.path(OUT, "q1_table1_power_targeted_refit_comparison.csv"),
  row.names = FALSE
)

numerical_zero_floors <- results[
  results$constrained_numerical_zero_floored %in% TRUE,
  c(
    "task_id",
    "design_id",
    "scenario_id",
    "truth_role",
    "replicate",
    "seed",
    "constrained_source",
    "constrained_statistic",
    "constrained_statistic_raw",
    "constrained_numerical_zero_rule",
    "constrained_boundary_optimizer_code_adjudicated",
    "constrained_numerical_zero_floor_tolerance",
    "constrained_initial_status",
    "constrained_initial_LR_raw",
    "constrained_rescue_status",
    "constrained_rescue_LR_raw",
    "constrained_chosen_full_optimizer_code",
    "constrained_chosen_null_optimizer_code",
    "constrained_chosen_full_convergence_messages",
    "constrained_chosen_null_convergence_messages",
    "constrained_chosen_full_non_singular_convergence_messages",
    "constrained_chosen_null_non_singular_convergence_messages",
    "constrained_chosen_full_slope_variance",
    "constrained_chosen_full_slope_relative_sd",
    "constrained_chosen_full_slope_at_boundary",
    "constrained_chosen_logLik_full",
    "constrained_chosen_logLik_null"
  ),
  drop = FALSE
]
utils::write.csv(
  numerical_zero_floors,
  file.path(OUT, "q1_table1_power_numerical_zero_floors.csv"),
  row.names = FALSE
)

method_manifest <- data.frame(
  method_id = c("direct_known_error", "constrained_G", "unstructured_G"),
  statistic_column = c(
    "direct_statistic",
    "constrained_statistic",
    "unstructured_statistic"
  ),
  label = c(
    "Direct-channel benchmark: slope-variance statistic with known error",
    "Diagnostic bridge: random intercept and slope constrained independent",
    "Question-1 LR statistic: unstructured random-effects covariance"
  ),
  alternative_parameters_added = c(1L, 1L, 2L),
  covariance_assumption = c(
    "Not estimated",
    "Intercept-slope covariance fixed to zero",
    "Intercept-slope covariance estimated"
  ),
  stringsAsFactors = FALSE
)
utils::write.csv(
  method_manifest,
  file.path(OUT, "q1_table1_power_method_manifest.csv"),
  row.names = FALSE
)

null_rows <- list()
critical_lookup <- list()
row_index <- 0L
for (design_id in included_designs$design_id) {
  scenario_id <- paste0(design_id, "_null")
  expected_n <- scenario_manifest$requested[
    scenario_manifest$scenario_id == scenario_id
  ]
  null_design <- results[results$scenario_id == scenario_id, , drop = FALSE]
  all_three_valid <- is.finite(null_design$direct_statistic) &
    is.finite(null_design$constrained_statistic) &
    is.finite(null_design$unstructured_statistic)

  for (method_index in seq_len(nrow(method_manifest))) {
    method <- method_manifest[method_index, ]
    statistic <- null_design[[method$statistic_column]]
    statistic <- statistic[is.finite(statistic)]
    calibration <- if (length(statistic) >= ceiling(1 / ALPHA)) {
      q1p_empirical_cutoff(statistic, ALPHA)
    } else NULL

    theoretical <- if (method$method_id == "direct_known_error") {
      design_df <- included_designs$df[
        included_designs$design_id == design_id
      ]
      stats::qchisq(1 - ALPHA, df = design_df)
    } else if (method$method_id == "constrained_G") {
      stats::qchisq(1 - 2 * ALPHA, df = 1L)
    } else {
      stats::uniroot(
        function(x) {
          0.5 * stats::pchisq(x, df = 1L, lower.tail = FALSE) +
            0.5 * stats::pchisq(x, df = 2L, lower.tail = FALSE) - ALPHA
        },
        interval = c(0, 25)
      )$root
    }

    row_index <- row_index + 1L
    null_rows[[row_index]] <- data.frame(
      design_id = design_id,
      scenario_id = scenario_id,
      method_id = method$method_id,
      n_requested = expected_n,
      n_method_valid = length(statistic),
      n_all_three_valid = sum(all_three_valid),
      complete_method_ensemble = length(statistic) == expected_n,
      critical = if (is.null(calibration)) NA_real_ else
        calibration$critical,
      theoretical_boundary_comparator = theoretical,
      critical_index = if (is.null(calibration)) NA_integer_ else
        calibration$critical_index,
      target_rejections = if (is.null(calibration)) NA_integer_ else
        calibration$target_rejections,
      observed_rejections = if (is.null(calibration)) NA_integer_ else
        calibration$n_above,
      values_equal_to_critical = if (is.null(calibration)) NA_integer_ else
        calibration$n_equal,
      realized_empirical_size = if (is.null(calibration)) NA_real_ else
        calibration$realized_size,
      exact_without_randomization = if (is.null(calibration)) FALSE else
        calibration$exact_without_randomization,
      conservative_without_randomization = if (is.null(calibration)) FALSE else
        calibration$conservative_without_randomization,
      theoretical_comparator_rejections = sum(statistic > theoretical),
      realized_theoretical_comparator_size = if (length(statistic)) {
        mean(statistic > theoretical)
      } else NA_real_,
      stringsAsFactors = FALSE
    )
    critical_lookup[[paste(design_id, method$method_id, sep = "::")]] <-
      if (is.null(calibration)) NA_real_ else calibration$critical
  }
}
null_calibration <- do.call(rbind, null_rows)
utils::write.csv(
  null_calibration,
  file.path(OUT, "q1_table1_power_null_calibration.csv"),
  row.names = FALSE
)

review_comparator <- data.frame(
  scenario_id = rep(
    c(
      "three_waves_two_years_alt_rho0",
      "five_waves_five_years_alt_rho0"
    ),
    each = 3L
  ),
  method_id = rep(method_manifest$method_id, times = 2L),
  review_M = c(rep(600L, 3L), rep(1000L, 3L)),
  review_power = c(0.085, 0.083, 0.062, 0.298, 0.230, 0.178),
  role = "User-supplied independent review; contextual, not a gate",
  stringsAsFactors = FALSE
)
utils::write.csv(
  review_comparator,
  file.path(OUT, "q1_table1_power_contextual_review_comparator.csv"),
  row.names = FALSE
)

alternative_scenarios <- scenario_manifest[
  scenario_manifest$truth_role != "null",
  ,
  drop = FALSE
]
power_rows <- list()
decision_store <- list()
row_index <- 0L
for (scenario_index in seq_len(nrow(alternative_scenarios))) {
  scenario <- alternative_scenarios[scenario_index, ]
  scenario_results <- results[
    results$scenario_id == scenario$scenario_id,
    ,
    drop = FALSE
  ]
  design <- included_designs[
    included_designs$design_id == scenario$design_id,
    ,
    drop = FALSE
  ]

  for (method_index in seq_len(nrow(method_manifest))) {
    method <- method_manifest[method_index, ]
    critical <- critical_lookup[[paste(
      scenario$design_id,
      method$method_id,
      sep = "::"
    )]]
    usable <- is.finite(scenario_results[[method$statistic_column]])
    method_results <- scenario_results[usable, , drop = FALSE]
    decisions <- if (nrow(method_results) && is.finite(critical)) {
      method_results[[method$statistic_column]] > critical
    } else logical()
    key <- paste(scenario$scenario_id, method$method_id, sep = "::")
    decision_store[[key]] <- data.frame(
      task_id = method_results$task_id,
      decision = decisions,
      stringsAsFactors = FALSE
    )
    k <- sum(decisions)
    n <- length(decisions)
    interval <- if (n) q1p_exact_interval(k, n) else
      c(low = NA_real_, high = NA_real_)
    exact_critical <- if (method$method_id == "direct_known_error") {
      design$theoretical_direct_critical
    } else NA_real_
    exact_decisions <- if (method$method_id == "direct_known_error" &&
      nrow(method_results)) {
      method_results[[method$statistic_column]] > exact_critical
    } else logical()
    exact_k <- sum(exact_decisions)
    exact_n <- length(exact_decisions)
    exact_interval <- if (exact_n) q1p_exact_interval(exact_k, exact_n) else
      c(low = NA_real_, high = NA_real_)
    comparator <- review_comparator[
      review_comparator$scenario_id == scenario$scenario_id &
        review_comparator$method_id == method$method_id,
      ,
      drop = FALSE
    ]

    row_index <- row_index + 1L
    power_rows[[row_index]] <- data.frame(
      scenario_id = scenario$scenario_id,
      design_id = scenario$design_id,
      truth_role = scenario$truth_role,
      true_intercept_slope_correlation =
        scenario$intercept_slope_correlation,
      method_id = method$method_id,
      n_requested = scenario$requested,
      n_method_valid = n,
      supports_empirical_critical = k,
      empirical_power = if (n) k / n else NA_real_,
      MCSE = if (n) sqrt((k / n) * (1 - k / n) / n) else NA_real_,
      interval_low = interval[["low"]],
      interval_high = interval[["high"]],
      direct_exact_chisq_critical = exact_critical,
      direct_exact_chisq_supports = if (exact_n) exact_k else NA_integer_,
      direct_power_exact_chisq = if (exact_n) exact_k / exact_n else NA_real_,
      direct_exact_MCSE = if (exact_n) {
        sqrt((exact_k / exact_n) * (1 - exact_k / exact_n) / exact_n)
      } else NA_real_,
      direct_exact_interval_low = exact_interval[["low"]],
      direct_exact_interval_high = exact_interval[["high"]],
      analytic_direct_channel_benchmark = design$analytic_direct_power,
      review_M = if (nrow(comparator)) comparator$review_M else NA_integer_,
      review_power = if (nrow(comparator)) {
        comparator$review_power
      } else {
        NA_real_
      },
      power_estimand = paste0(
        "Statistic power at its design-specific DGP empirical null critical; ",
        "not nested-bootstrap decision power"
      ),
      MCSE_scope = paste0(
        "Alternative-ensemble Monte Carlo only; uncertainty in the empirical ",
        "null critical is not included"
      ),
      stringsAsFactors = FALSE
    )
  }
}
power_summary <- do.call(rbind, power_rows)
utils::write.csv(
  power_summary,
  file.path(OUT, "q1_table1_power_summary.csv"),
  row.names = FALSE
)

pair_definitions <- data.frame(
  method_a = c(
    "direct_known_error",
    "constrained_G",
    "direct_known_error"
  ),
  method_b = c(
    "constrained_G",
    "unstructured_G",
    "unstructured_G"
  ),
  stringsAsFactors = FALSE
)
paired_rows <- list()
row_index <- 0L
for (scenario_index in seq_len(nrow(alternative_scenarios))) {
  scenario <- alternative_scenarios[scenario_index, ]
  for (pair_index in seq_len(nrow(pair_definitions))) {
    pair <- pair_definitions[pair_index, ]
    a <- decision_store[[paste(
      scenario$scenario_id,
      pair$method_a,
      sep = "::"
    )]]
    b <- decision_store[[paste(
      scenario$scenario_id,
      pair$method_b,
      sep = "::"
    )]]
    paired <- merge(a, b, by = "task_id", suffixes = c("_a", "_b"))
    difference <- as.numeric(paired$decision_a) -
      as.numeric(paired$decision_b)
    estimate <- if (length(difference)) mean(difference) else NA_real_
    se <- if (length(difference) > 1L) {
      stats::sd(difference) / sqrt(length(difference))
    } else {
      NA_real_
    }

    row_index <- row_index + 1L
    paired_rows[[row_index]] <- data.frame(
      scenario_id = scenario$scenario_id,
      design_id = scenario$design_id,
      true_intercept_slope_correlation =
        scenario$intercept_slope_correlation,
      method_a = pair$method_a,
      method_b = pair$method_b,
      n_paired = length(difference),
      power_difference_a_minus_b = estimate,
      paired_SE = se,
      normal_interval_low = if (is.finite(se)) {
        max(-1, estimate - stats::qnorm(0.975) * se)
      } else {
        NA_real_
      },
      normal_interval_high = if (is.finite(se)) {
        min(1, estimate + stats::qnorm(0.975) * se)
      } else {
        NA_real_
      },
      a_only_supports = sum(paired$decision_a & !paired$decision_b),
      b_only_supports = sum(!paired$decision_a & paired$decision_b),
      stringsAsFactors = FALSE
    )
  }
}
paired_differences <- do.call(rbind, paired_rows)
utils::write.csv(
  paired_differences,
  file.path(OUT, "q1_table1_power_paired_differences.csv"),
  row.names = FALSE
)

rho_pair_rows <- list()
row_index <- 0L
for (design_id in included_designs$design_id) {
  rho0 <- results[
    results$scenario_id == paste0(design_id, "_alt_rho0") &
      nzchar(results$pair_id),
  ]
  rho5 <- results[
    results$scenario_id == paste0(design_id, "_alt_rho0p5_paired") &
      nzchar(results$pair_id),
  ]
  paired <- merge(rho0, rho5, by = "pair_id", suffixes = c("_rho0", "_rho0p5"))
  if (!nrow(paired)) next
  direct_difference <- paired$direct_statistic_rho0p5 -
    paired$direct_statistic_rho0
  for (method_index in seq_len(nrow(method_manifest))) {
    method <- method_manifest[method_index, ]
    statistic_0 <- paired[[paste0(method$statistic_column, "_rho0")]]
    statistic_5 <- paired[[paste0(method$statistic_column, "_rho0p5")]]
    usable <- is.finite(statistic_0) & is.finite(statistic_5)
    critical <- critical_lookup[[paste(design_id, method$method_id, sep = "::")]]
    decision_0 <- statistic_0[usable] > critical
    decision_5 <- statistic_5[usable] > critical
    difference <- as.numeric(decision_5) - as.numeric(decision_0)
    row_index <- row_index + 1L
    rho_pair_rows[[row_index]] <- data.frame(
      design_id = design_id,
      method_id = method$method_id,
      n_paired = sum(usable),
      power_rho0 = mean(decision_0),
      power_rho0p5 = mean(decision_5),
      difference_rho0p5_minus_rho0 = mean(difference),
      paired_SE = if (length(difference) > 1L) {
        stats::sd(difference) / sqrt(length(difference))
      } else NA_real_,
      rho0_only_supports = sum(decision_0 & !decision_5),
      rho0p5_only_supports = sum(!decision_0 & decision_5),
      max_abs_direct_statistic_difference = max(abs(direct_difference)),
      direct_channel_exactly_paired_within_1e_10 =
        max(abs(direct_difference)) <= 1e-10,
      stringsAsFactors = FALSE
    )
  }
}
rho_pair_comparison <- if (length(rho_pair_rows)) {
  do.call(rbind, rho_pair_rows)
} else {
  data.frame(
    design_id = character(), method_id = character(), n_paired = integer(),
    power_rho0 = numeric(), power_rho0p5 = numeric(),
    difference_rho0p5_minus_rho0 = numeric(), paired_SE = numeric(),
    rho0_only_supports = integer(), rho0p5_only_supports = integer(),
    max_abs_direct_statistic_difference = numeric(),
    direct_channel_exactly_paired_within_1e_10 = logical(),
    stringsAsFactors = FALSE
  )
}
utils::write.csv(
  rho_pair_comparison,
  file.path(OUT, "q1_table1_power_paired_rho_comparison.csv"),
  row.names = FALSE
)

fit_diagnostics <- do.call(rbind, lapply(
  c("constrained", "unstructured"),
  function(prefix) {
    data.frame(
      model = prefix,
      total_datasets = nrow(results),
      final_ok = sum(results[[paste0(prefix, "_status")]] == "ok"),
      unresolved_or_error = sum(
        results[[paste0(prefix, "_status")]] != "ok"
      ),
      rescue_triggered = sum(
        results[[paste0(prefix, "_rescue_triggered")]],
        na.rm = TRUE
      ),
      rescue_rate = mean(
        results[[paste0(prefix, "_rescue_triggered")]] %in% TRUE
      ),
      rescue_succeeded = sum(
        results[[paste0(prefix, "_rescue_succeeded")]],
        na.rm = TRUE
      ),
      numerical_zero_floored = sum(
        results[[paste0(prefix, "_numerical_zero_floored")]],
        na.rm = TRUE
      ),
      numerical_zero_floor_rate = mean(
        results[[paste0(prefix, "_numerical_zero_floored")]] %in% TRUE
      ),
      boundary_optimizer_code_adjudicated = sum(
        results[[paste0(
          prefix,
          "_boundary_optimizer_code_adjudicated"
        )]] %in% TRUE
      ),
      chosen_full_singular = sum(
        results[[paste0(prefix, "_chosen_full_singular")]],
        na.rm = TRUE
      ),
      stringsAsFactors = FALSE
    )
  }
))
utils::write.csv(
  fit_diagnostics,
  file.path(OUT, "q1_table1_power_fit_diagnostics.csv"),
  row.names = FALSE
)

primary_power <- power_summary[
  grepl("_alt_rho0$", power_summary$scenario_id),
  ,
  drop = FALSE
]
candidate_rows <- do.call(rbind, lapply(
  included_designs$design_id,
  function(design_id) {
    design <- included_designs[
      included_designs$design_id == design_id,
      ,
      drop = FALSE
    ]
    scenario_id <- paste0(design_id, "_alt_rho0")
    power <- primary_power[primary_power$scenario_id == scenario_id, ]
    get_empirical_power <- function(method) {
      power$empirical_power[power$method_id == method]
    }
    get_exact_direct_power <- function() {
      power$direct_power_exact_chisq[
        power$method_id == "direct_known_error"
      ]
    }
    sensitivity <- power_summary[
      power_summary$scenario_id ==
        paste0(design_id, "_alt_rho0p5_paired") &
        power_summary$method_id == "unstructured_G",
    ]
    data.frame(
      design_id = design_id,
      table_label = design$table_label,
      R_direct = design$R_direct,
      direct_channel_benchmark_analytic = design$analytic_direct_power,
      direct_channel_power_exact_chisq = get_exact_direct_power(),
      direct_channel_power_empirical_matched_size =
        get_empirical_power("direct_known_error"),
      unstructured_Q1_power_rho0 = get_empirical_power("unstructured_G"),
      unstructured_Q1_power_rho0p5_paired = if (nrow(sensitivity)) {
        sensitivity$empirical_power
      } else {
        NA_real_
      },
      status = if (identical(MODE, "extend")) {
        "validated statistic-level reference used by v14 nested-bootstrap validation"
      } else if (identical(MODE, "repair")) {
        "historical repair-mode evidence; extension required for v14 reference"
      } else {
        "smoke only; not manuscript evidence"
      },
      stringsAsFactors = FALSE
    )
  }
))
utils::write.csv(
  candidate_rows,
  file.path(OUT, "q1_table1_power_candidate_table_rows.csv"),
  row.names = FALSE
)

task_counts_ok <- nrow(results) == nrow(task_manifest) &&
  !anyDuplicated(results$task_id) &&
  setequal(results$task_id, task_manifest$task_id)
all_fits_resolved <- task_counts_ok && all(results$status == "ok")
all_null_counts_ok <- isTRUE(all(
  null_calibration$n_method_valid == null_calibration$n_requested
))
all_null_conservative <- isTRUE(all(
  null_calibration$conservative_without_randomization
)) && isTRUE(all(
  is.finite(null_calibration$realized_empirical_size) &
    null_calibration$realized_empirical_size <=
      ALPHA + 100 * .Machine$double.eps
))

numerical_floor_scope_ok <- q1p_numerical_zero_scope_ok(results)

direct_formula_gate <- TRUE
if (!identical(MODE, "smoke")) {
  direct_primary <- primary_power[
    primary_power$method_id == "direct_known_error",
    ,
    drop = FALSE
  ]
  direct_formula_gate <- isTRUE(all(vapply(seq_len(nrow(direct_primary)), function(i) {
    row <- direct_primary[i, ]
    target <- row$analytic_direct_channel_benchmark
    if (!is.finite(row$n_method_valid) || row$n_method_valid < 1L ||
        !is.finite(row$direct_power_exact_chisq) || !is.finite(target)) {
      return(FALSE)
    }
    tolerance <- 4 * sqrt(target * (1 - target) / row$n_method_valid)
    abs(row$direct_power_exact_chisq - target) <= tolerance
  }, logical(1))))
}

paired_direct_invariance <- if (identical(MODE, "repair")) {
  TRUE
} else {
  nrow(rho_pair_comparison) == 3L * nrow(included_designs) &&
    all(rho_pair_comparison$direct_channel_exactly_paired_within_1e_10)
}

import_gate <- if (identical(MODE, "smoke")) {
  TRUE
} else if (identical(MODE, "repair")) {
  !is.null(import_provenance) && import_provenance$imported_rows == 5000L &&
    import_provenance$retained_rows == 4996L &&
    import_provenance$refit_rows == 4L
} else {
  !is.null(import_provenance) && import_provenance$imported_rows == 5000L &&
    import_provenance$retained_rows == 5000L &&
    import_provenance$refit_rows == 0L
}
targeted_refit_reproduction_gate <- if (identical(MODE, "repair")) {
  nrow(targeted_refit_comparison) == 4L &&
    all(targeted_refit_comparison$direct_statistic_reproduced) &&
    all(targeted_refit_comparison$unstructured_science_reproduced)
} else TRUE

operational_gate_notes <- c(
  paste(nrow(results), "of", nrow(task_manifest), "tasks recorded"),
  paste(sum(results$status == "ok"), "final task rows are usable"),
  paste(
    sum(null_calibration$n_method_valid),
    "method-valid null statistics across",
    nrow(null_calibration),
    "calibrations"
  ),
  paste0(
    "Conservative upper-tail empirical criticals; realized size reported"
  ),
  paste(
    nrow(numerical_zero_floors),
    "constrained numerical-zero adjudications with raw provenance;",
    "zero unstructured floors"
  ),
  if (!identical(MODE, "smoke")) {
    "Direct statistic at exact chi-square critical versus analytic benchmark"
  } else {
    "Not applied in smoke mode"
  },
  if (identical(MODE, "repair")) {
    "Not applied in repair mode"
  } else {
    paste(nrow(rho_pair_comparison), "method-by-design paired-rho checks")
  },
  if (identical(MODE, "smoke")) {
    "Fresh smoke run"
  } else {
    paste0("Checkpoint import: ", import_provenance$imported_rows,
           " imported; ", import_provenance$refit_rows, " targeted refits")
  },
  if (identical(MODE, "repair")) {
    paste(nrow(targeted_refit_comparison),
          "targeted datasets checked against prior direct/Q1 statistics")
  } else "Not applied outside repair mode"
)

operational_gate_checks <- c(
  "task_manifest_complete_and_unique",
  "all_mixed_model_pairs_resolved",
  "all_null_ensembles_complete",
  "all_method_calibrations_are_conservative_and_available",
  "numerical_zero_adjudication_is_constrained_rescue_only",
  "direct_statistic_matches_analytic_formula_within_4_MCSE",
  "paired_rho_design_holds_direct_channel_fixed",
  "checkpoint_import_and_targeted_refit_contract",
  "targeted_refit_preserves_direct_and_unstructured_statistics"
)
operational_gate_passes <- c(
  task_counts_ok,
  all_fits_resolved,
  all_null_counts_ok,
  all_null_conservative,
  numerical_floor_scope_ok,
  direct_formula_gate,
  paired_direct_invariance,
  import_gate,
  targeted_refit_reproduction_gate
)
stopifnot(
  length(operational_gate_checks) == length(operational_gate_passes),
  length(operational_gate_checks) == length(operational_gate_notes)
)

operational_gate <- data.frame(
  check = operational_gate_checks,
  pass = operational_gate_passes,
  note = operational_gate_notes,
  stringsAsFactors = FALSE
)
utils::write.csv(
  operational_gate,
  file.path(OUT, "q1_table1_power_operational_gate.csv"),
  row.names = FALSE
)

writeLines(
  capture.output(sessionInfo()),
  file.path(OUT, "sessionInfo_q1_table1_power_v13_3_1.txt")
)

if (all(operational_gate$pass)) {
  message(
    "PASS: v13.3.1 ", MODE,
    " paired Table-1 power diagnostic completed operationally."
  )
} else {
  message(
    "DIAGNOSE: v13.3.1 ", MODE,
    " completed, but one or more operational gates failed."
  )
}
