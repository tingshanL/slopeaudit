#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(slopeaudit)
})

source(
  file.path(
    "validation",
    "extended",
    "q1_optimizer_rescue_helpers.R"
  ),
  local = TRUE
)

SCRIPT_VERSION <- "q1_optimizer_rescue_diagnostic_v11"
MODE <- tolower(Sys.getenv("SLOPEAUDIT_Q1_RESCUE_MODE", "smoke"))
if (!MODE %in% c("smoke", "full")) {
  stop("SLOPEAUDIT_Q1_RESCUE_MODE must be smoke or full.")
}

WORKERS <- q1v11_read_integer_env(
  "SLOPEAUDIT_Q1_RESCUE_WORKERS",
  1L
)
CHECKPOINT_BATCH <- q1v11_read_integer_env(
  "SLOPEAUDIT_Q1_RESCUE_CHECKPOINT_BATCH",
  max(1L, WORKERS)
)
if (WORKERS > 1L && .Platform$OS.type == "windows") {
  stop("Parallel execution currently requires macOS/Linux.")
}
if (WORKERS > 2L) {
  warning(
    "More than two outer workers are not recommended on the 8-GB M2."
  )
}

PILOT_INPUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_PILOT_INPUT",
  unset = ""
)
FIXTURE_PATH <- Sys.getenv(
  "SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE",
  unset = ""
)
OUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_OUT",
  file.path(
    "validation",
    "extended",
    "results",
    paste0("q1_optimizer_rescue_", MODE, "_v11")
  )
)

if (!nzchar(PILOT_INPUT)) {
  stop(
    "Historical v11 diagnostic requires private restricted-derived v10 input. ",
    "Set SLOPEAUDIT_Q1_PILOT_INPUT explicitly.",
    call. = FALSE
  )
}
if (!nzchar(FIXTURE_PATH)) {
  stop(
    "Historical v11 diagnostic requires the private exact irregular schedule. ",
    "Set SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE explicitly.",
    call. = FALSE
  )
}

input_paths <- c(
  configuration = file.path(
    PILOT_INPUT,
    "q1_four_design_configuration.rds"
  ),
  task_manifest = file.path(
    PILOT_INPUT,
    "q1_four_design_task_manifest.csv"
  ),
  outer_runs = file.path(
    PILOT_INPUT,
    "q1_four_design_outer_runs.csv"
  ),
  inner_problems = file.path(
    PILOT_INPUT,
    "q1_four_design_inner_problems.csv"
  ),
  irregular_schedule = FIXTURE_PATH
)
missing_inputs <- input_paths[!file.exists(input_paths)]
if (length(missing_inputs)) {
  stop(
    "Missing required v10 pilot input(s):\n",
    paste(missing_inputs, collapse = "\n")
  )
}

pilot_configuration <- readRDS(input_paths[["configuration"]])
task_manifest <- utils::read.csv(
  input_paths[["task_manifest"]],
  stringsAsFactors = FALSE,
  check.names = FALSE
)
outer_runs <- utils::read.csv(
  input_paths[["outer_runs"]],
  stringsAsFactors = FALSE,
  check.names = FALSE
)
problems <- utils::read.csv(
  input_paths[["inner_problems"]],
  stringsAsFactors = FALSE,
  check.names = FALSE
)

required_task_columns <- c(
  "task_index",
  "cell_id",
  "outer",
  "design_seed",
  "bootstrap_seed"
)
required_outer_columns <- c(
  "cell_id",
  "outer",
  "status",
  "observed_LR",
  "inner_failures",
  "small_negative",
  "material_negative"
)
required_problem_columns <- c(
  "cell_id",
  "outer",
  "design_seed",
  "bootstrap_seed",
  "replicate",
  "seed",
  "status",
  "reason",
  "LR_raw",
  "LR_zero",
  "tolerance",
  "logLik_full",
  "logLik_null",
  "full_singular",
  "null_singular",
  "warnings",
  "messages"
)
if (!all(required_task_columns %in% names(task_manifest))) {
  stop("The v10 task manifest is missing required columns.")
}
if (!all(required_outer_columns %in% names(outer_runs))) {
  stop("The v10 outer-run file is missing required columns.")
}
if (!all(required_problem_columns %in% names(problems))) {
  stop("The v10 inner-problem file is missing required columns.")
}

# This diagnostic is bound to the audited v10 pilot.
if (!identical(pilot_configuration$version,
               "q1_four_design_calibration_v10_candidate") ||
    !identical(pilot_configuration$mode, "pilot") ||
    pilot_configuration$outer_per_cell != 20L ||
    pilot_configuration$usable_B != 99L ||
    pilot_configuration$attempt_cap != 149L ||
    pilot_configuration$alpha != 0.05) {
  stop("The input configuration is not the audited v10 20x6 pilot.")
}

control_panel <- q1v11_control_panel()
current_control <- control_panel$current_nloptwrap$control
control_match <- isTRUE(all.equal(
  pilot_configuration$control,
  current_control,
  check.attributes = TRUE
))
if (!control_match) {
  stop("The v10 pilot control does not match the frozen reproduction control.")
}

fixture_md5 <- unname(tools::md5sum(FIXTURE_PATH))
if (!identical(fixture_md5, pilot_configuration$irregular_fixture_md5)) {
  stop("The irregular schedule fixture does not match the v10 pilot hash.")
}

task_key <- paste(task_manifest$cell_id, task_manifest$outer, sep = "::")
outer_key <- paste(outer_runs$cell_id, outer_runs$outer, sep = "::")
problem_outer_key <- paste(problems$cell_id, problems$outer, sep = "::")
if (anyDuplicated(task_key) || anyDuplicated(outer_key) ||
    !setequal(task_key, outer_key)) {
  stop("The v10 task and outer-run keys do not reconcile one-to-one.")
}

task_match <- match(problem_outer_key, task_key)
if (anyNA(task_match) ||
    any(problems$design_seed != task_manifest$design_seed[task_match]) ||
    any(problems$bootstrap_seed !=
      task_manifest$bootstrap_seed[task_match])) {
  stop("Problem rows do not match the v10 outer task seeds.")
}

problem_task_id <- paste(
  problems$cell_id,
  problems$outer,
  problems$replicate,
  problems$seed,
  sep = "::"
)
if (anyDuplicated(problem_task_id)) {
  stop("Duplicate inner-problem task keys were found.")
}

expected_seed <- integer(nrow(problems))
for (key in unique(problem_outer_key)) {
  index <- which(problem_outer_key == key)
  bootstrap_seed <- unique(problems$bootstrap_seed[index])
  if (length(bootstrap_seed) != 1L) {
    stop("An outer dataset has inconsistent bootstrap seeds: ", key)
  }
  seed_stream <- slopeaudit:::.sa_q1_make_replicate_seeds(
    B = pilot_configuration$attempt_cap,
    seed = bootstrap_seed
  )
  if (any(problems$replicate[index] < 1L) ||
      any(problems$replicate[index] > length(seed_stream))) {
    stop("A saved replicate index lies outside its seed stream: ", key)
  }
  expected_seed[index] <- seed_stream[problems$replicate[index]]
}
if (any(expected_seed != problems$seed)) {
  stop("At least one saved inner seed disagrees with its v10 seed stream.")
}

status_counts <- table(problems$status)
outer_diagnostic_totals <- colSums(
  outer_runs[, c(
    "inner_failures",
    "small_negative",
    "material_negative"
  )],
  na.rm = TRUE
)
input_anchors_ok <- nrow(task_manifest) == 120L &&
  nrow(outer_runs) == 120L &&
  nrow(problems) == 908L &&
  sum(problem_outer_key != "") == 908L &&
  length(unique(problem_outer_key)) == 95L &&
  unname(status_counts[["failure"]]) == 866L &&
  unname(status_counts[["small_negative"]]) == 14L &&
  unname(status_counts[["material_negative"]]) == 28L &&
  outer_diagnostic_totals[["inner_failures"]] == 866L &&
  outer_diagnostic_totals[["small_negative"]] == 14L &&
  outer_diagnostic_totals[["material_negative"]] == 28L &&
  sum(outer_runs$status == "attempt_cap_reached") == 5L
if (!input_anchors_ok) {
  stop("The input files do not match the audited 908-row v10 pilot baseline.")
}

problems$diagnostic_class <- mapply(
  q1v11_warning_class,
  problems$status,
  problems$warnings,
  problems$messages,
  USE.NAMES = FALSE
)
problems$task_id <- problem_task_id
problems$outer_key <- problem_outer_key

selected <- if (MODE == "smoke") {
  q1v11_smoke_tasks(problems)
} else {
  problems
}
selected <- selected[
  order(selected$cell_id, selected$outer, selected$replicate),
  ,
  drop = FALSE
]
rownames(selected) <- NULL
expected_selected <- if (MODE == "smoke") 20L else 908L
if (nrow(selected) != expected_selected) {
  stop(
    "Unexpected v11 ", MODE, " task count: ", nrow(selected),
    " rather than ", expected_selected, "."
  )
}

cell_manifest <- q1v11_cell_manifest()
schedules <- q1v11_load_schedules(FIXTURE_PATH)
if (!setequal(cell_manifest$cell_id, unique(task_manifest$cell_id))) {
  stop("The reconstructed DGP cell IDs do not match the pilot manifest.")
}

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)
input_manifest <- data.frame(
  input = names(input_paths),
  basename = basename(input_paths),
  bytes = as.numeric(file.info(input_paths)$size),
  md5 = unname(tools::md5sum(input_paths)),
  stringsAsFactors = FALSE
)
utils::write.csv(
  input_manifest,
  file.path(OUT, "q1_optimizer_rescue_input_manifest.csv"),
  row.names = FALSE
)

method_manifest <- do.call(
  rbind,
  lapply(names(control_panel), function(method_id) {
    spec <- control_panel[[method_id]]
    data.frame(
      method_id = method_id,
      method_role = spec$role,
      description = spec$description,
      optimizer = paste(spec$control$optimizer, collapse = ";"),
      calc_derivs = isTRUE(spec$control$calc.derivs),
      optCtrl = paste(
        paste(
          names(spec$control$optCtrl),
          unlist(spec$control$optCtrl),
          sep = "="
        ),
        collapse = ";"
      ),
      stringsAsFactors = FALSE
    )
  })
)
utils::write.csv(
  method_manifest,
  file.path(OUT, "q1_optimizer_rescue_method_manifest.csv"),
  row.names = FALSE
)

task_output_columns <- c(
  "task_id",
  "outer_key",
  "cell_id",
  "outer",
  "design_seed",
  "bootstrap_seed",
  "replicate",
  "seed",
  "status",
  "reason",
  "diagnostic_class",
  "LR_raw",
  "LR_zero",
  "tolerance",
  "logLik_full",
  "logLik_null",
  "full_singular",
  "null_singular",
  "warnings",
  "messages"
)
utils::write.csv(
  selected[, task_output_columns, drop = FALSE],
  file.path(OUT, "q1_optimizer_rescue_tasks.csv"),
  row.names = FALSE
)

scientific_config <- list(
  version = SCRIPT_VERSION,
  mode = MODE,
  workers = WORKERS,
  checkpoint_batch = CHECKPOINT_BATCH,
  pilot_input_md5 = input_manifest$md5,
  pilot_input_names = input_manifest$input,
  selected_task_ids = selected$task_id,
  reproduction_relative_tolerance = 1e-7,
  candidate_agreement_relative_tolerance = 1e-6,
  controls = lapply(control_panel, `[[`, "control"),
  control_roles = vapply(control_panel, `[[`, character(1), "role")
)
configuration_path <- file.path(
  OUT,
  "q1_optimizer_rescue_configuration.rds"
)
if (file.exists(configuration_path)) {
  prior <- readRDS(configuration_path)
  if (!identical(prior, scientific_config)) {
    stop("Existing v11 output uses a different configuration.")
  }
} else {
  saveRDS(scientific_config, configuration_path, version = 3)
}

run_outer_group <- function(key) {
  task_rows <- selected[selected$outer_key == key, , drop = FALSE]
  cell <- cell_manifest[
    cell_manifest$cell_id == task_rows$cell_id[[1]],
    ,
    drop = FALSE
  ]
  if (nrow(cell) != 1L || length(unique(task_rows$design_seed)) != 1L) {
    stop("Invalid selected outer-task group: ", key)
  }

  dat <- q1v11_generate_outcome(
    schedules[[cell$schedule]],
    cell,
    task_rows$design_seed[[1]]
  )
  full_capture <- slopeaudit:::.sa_q1_capture(
    lme4::lmer(
      y ~ time + (1 + time | id),
      data = dat,
      REML = TRUE,
      control = current_control
    )
  )
  null_capture <- slopeaudit:::.sa_q1_capture(
    lme4::lmer(
      y ~ time + (1 | id),
      data = dat,
      REML = TRUE,
      control = current_control
    )
  )
  if (!is.null(full_capture$error) || !is.null(null_capture$error)) {
    stop("Could not reconstruct the observed model pair for ", key)
  }
  full_fit <- full_capture$value
  null_fit <- null_capture$value
  local_control_panel <- control_panel
  resolved_current <- slopeaudit:::.sa_q1_resolve_refit_control(
    full_fit,
    null_fit,
    refit_control = current_control
  )
  local_control_panel$current_nloptwrap$control <-
    resolved_current$control
  local_control_panel$current_no_derivs$control$optCtrl <-
    resolved_current$optimizer_control
  observed <- slopeaudit:::.sa_q1_observed_lr(
    full_fit,
    null_fit,
    target = "time",
    diagnostic_control = current_control,
    strict_controls = TRUE
  )
  original_outer <- outer_runs[outer_key == key, , drop = FALSE]
  observed_match <- nrow(original_outer) == 1L &&
    isTRUE(observed$valid_for_inference) &&
    q1v11_numeric_match(
      observed$LR_for_inference,
      original_outer$observed_LR[[1]],
      relative_tolerance = 1e-7
    )
  observed_row <- data.frame(
    outer_key = key,
    cell_id = cell$cell_id,
    outer = task_rows$outer[[1]],
    design_seed = task_rows$design_seed[[1]],
    original_observed_LR = original_outer$observed_LR[[1]],
    reproduced_observed_LR = observed$LR_for_inference,
    original_status = original_outer$status[[1]],
    reproduced_observed_status = observed$status,
    observed_match = observed_match,
    stringsAsFactors = FALSE
  )

  pair_rows <- vector("list", nrow(task_rows))
  for (index in seq_len(nrow(task_rows))) {
    task <- task_rows[index, ]
    simulation <- slopeaudit:::.sa_q1_capture(
      slopeaudit:::.sa_q1_simulate_null(null_fit, task$seed)
    )
    if (!is.null(simulation$error)) {
      stop("Could not reproduce inner simulation for task ", task$task_id)
    }
    response <- simulation$value
    methods <- do.call(
      rbind,
      lapply(names(local_control_panel), function(method_id) {
        q1v11_refit_pair(
          full_fit,
          null_fit,
          response,
          method_id,
          local_control_panel[[method_id]]
        )
      })
    )
    identity <- data.frame(
      task_id = task$task_id,
      outer_key = task$outer_key,
      cell_id = task$cell_id,
      outer = task$outer,
      design_seed = task$design_seed,
      bootstrap_seed = task$bootstrap_seed,
      replicate = task$replicate,
      seed = task$seed,
      original_status = task$status,
      original_reason = task$reason,
      original_diagnostic_class = task$diagnostic_class,
      original_LR_raw = task$LR_raw,
      original_LR_zero = task$LR_zero,
      original_tolerance = task$tolerance,
      original_logLik_full = task$logLik_full,
      original_logLik_null = task$logLik_null,
      original_full_singular = task$full_singular,
      original_null_singular = task$null_singular,
      stringsAsFactors = FALSE
    )
    pair_rows[[index]] <- cbind(
      identity[rep(1L, nrow(methods)), , drop = FALSE],
      methods
    )
  }

  list(
    results = do.call(rbind, pair_rows),
    observed = observed_row,
    n_tasks = nrow(task_rows)
  )
}

checkpoint_path <- file.path(OUT, "q1_optimizer_rescue_checkpoint.rds")
results_csv <- file.path(OUT, "q1_optimizer_rescue_results.csv")
observed_csv <- file.path(OUT, "q1_optimizer_rescue_observed_reproduction.csv")
if (file.exists(checkpoint_path)) {
  checkpoint <- readRDS(checkpoint_path)
  results <- checkpoint$results
  observed_reproduction <- checkpoint$observed_reproduction
  completed_outer_keys <- checkpoint$completed_outer_keys
} else {
  results <- NULL
  observed_reproduction <- NULL
  completed_outer_keys <- character()
}

save_checkpoint <- function() {
  if (!is.null(results) && nrow(results)) {
    results <<- results[
      order(
        results$cell_id,
        results$outer,
        results$replicate,
        match(results$method_id, names(control_panel))
      ),
      ,
      drop = FALSE
    ]
    rownames(results) <<- NULL
  }
  if (!is.null(observed_reproduction) && nrow(observed_reproduction)) {
    observed_reproduction <<- observed_reproduction[
      order(
        observed_reproduction$cell_id,
        observed_reproduction$outer
      ),
      ,
      drop = FALSE
    ]
    rownames(observed_reproduction) <<- NULL
  }
  completed_outer_keys <<- sort(unique(completed_outer_keys))
  temporary <- paste0(checkpoint_path, ".tmp")
  saveRDS(
    list(
      results = results,
      observed_reproduction = observed_reproduction,
      completed_outer_keys = completed_outer_keys
    ),
    temporary,
    version = 3
  )
  if (!file.rename(temporary, checkpoint_path)) {
    stop("Could not publish the v11 checkpoint atomically.")
  }
  utils::write.csv(results, results_csv, row.names = FALSE)
  utils::write.csv(
    observed_reproduction,
    observed_csv,
    row.names = FALSE
  )
}

outer_keys_to_run <- unique(selected$outer_key)
missing_keys <- setdiff(outer_keys_to_run, completed_outer_keys)
if (length(missing_keys)) {
  progress_bar <- utils::txtProgressBar(
    min = 0,
    max = nrow(selected),
    initial = sum(selected$outer_key %in% completed_outer_keys),
    style = 3
  )
  completed_tasks <- sum(selected$outer_key %in% completed_outer_keys)
  batches <- split(
    missing_keys,
    ceiling(seq_along(missing_keys) / CHECKPOINT_BATCH)
  )
  for (batch in batches) {
    answers <- if (WORKERS == 1L) {
      lapply(batch, run_outer_group)
    } else {
      parallel::mclapply(
        batch,
        run_outer_group,
        mc.cores = WORKERS,
        mc.preschedule = FALSE,
        mc.set.seed = FALSE
      )
    }
    if (any(vapply(answers, inherits, logical(1), "try-error"))) {
      stop("At least one parallel outer rescue task failed.")
    }
    batch_results <- do.call(rbind, lapply(answers, `[[`, "results"))
    batch_observed <- do.call(rbind, lapply(answers, `[[`, "observed"))
    results <- rbind(results, batch_results)
    observed_reproduction <- rbind(
      observed_reproduction,
      batch_observed
    )
    completed_outer_keys <- c(completed_outer_keys, batch)
    save_checkpoint()
    completed_tasks <- completed_tasks + sum(vapply(
      answers,
      `[[`,
      integer(1),
      "n_tasks"
    ))
    utils::setTxtProgressBar(progress_bar, completed_tasks)
  }
  close(progress_bar)
}

expected_result_rows <- nrow(selected) * length(control_panel)
complete_methods <- table(results$task_id)
results_complete <- nrow(results) == expected_result_rows &&
  length(complete_methods) == nrow(selected) &&
  all(complete_methods == length(control_panel)) &&
  all(vapply(
    split(results$method_id, results$task_id),
    function(x) setequal(x, names(control_panel)),
    logical(1)
  ))

current <- results[
  results$method_id == "current_nloptwrap",
  ,
  drop = FALSE
]
current <- current[match(selected$task_id, current$task_id), , drop = FALSE]
normalize_reason <- function(x) {
  x[is.na(x)] <- ""
  x
}
status_match <- selected$status == current$status
reason_match <- normalize_reason(selected$reason) ==
  normalize_reason(current$reason)
numeric_required <- selected$status != "failure"
lr_match <- !numeric_required | mapply(
  q1v11_numeric_match,
  selected$LR_raw,
  current$LR_raw,
  MoreArgs = list(relative_tolerance = 1e-7)
)
full_logLik_match <- !numeric_required | mapply(
  q1v11_numeric_match,
  selected$logLik_full,
  current$full_logLik,
  MoreArgs = list(relative_tolerance = 1e-7)
)
null_logLik_match <- !numeric_required | mapply(
  q1v11_numeric_match,
  selected$logLik_null,
  current$null_logLik,
  MoreArgs = list(relative_tolerance = 1e-7)
)
current_class <- mapply(
  q1v11_warning_class,
  current$status,
  paste(
    current$full_captured_warnings,
    current$null_captured_warnings,
    sep = " | "
  ),
  paste(
    current$full_captured_messages,
    current$null_captured_messages,
    sep = " | "
  ),
  USE.NAMES = FALSE
)
diagnostic_class_match <- selected$diagnostic_class == current_class
baseline_reproduced <- status_match & reason_match & lr_match &
  full_logLik_match & null_logLik_match & diagnostic_class_match

reproduction <- data.frame(
  task_id = selected$task_id,
  cell_id = selected$cell_id,
  outer = selected$outer,
  replicate = selected$replicate,
  seed = selected$seed,
  original_status = selected$status,
  reproduced_status = current$status,
  original_reason = selected$reason,
  reproduced_reason = current$reason,
  original_diagnostic_class = selected$diagnostic_class,
  reproduced_diagnostic_class = current_class,
  original_LR_raw = selected$LR_raw,
  reproduced_LR_raw = current$LR_raw,
  status_match = status_match,
  reason_match = reason_match,
  diagnostic_class_match = diagnostic_class_match,
  LR_match = lr_match,
  full_logLik_match = full_logLik_match,
  null_logLik_match = null_logLik_match,
  baseline_reproduced = baseline_reproduced,
  stringsAsFactors = FALSE
)
utils::write.csv(
  reproduction,
  file.path(OUT, "q1_optimizer_rescue_reproduction.csv"),
  row.names = FALSE
)

one_method <- function(task_rows, method_id) {
  row <- task_rows[task_rows$method_id == method_id, , drop = FALSE]
  if (nrow(row) != 1L) {
    stop("Incomplete method panel for task ", task_rows$task_id[[1]])
  }
  row
}

task_summary_rows <- lapply(selected$task_id, function(task_id) {
  task_rows <- results[results$task_id == task_id, , drop = FALSE]
  current_row <- one_method(task_rows, "current_nloptwrap")
  tight <- one_method(task_rows, "tight_nloptwrap")
  bobyqa <- one_method(task_rows, "bobyqa")
  no_derivs <- one_method(task_rows, "current_no_derivs")
  both_primary <- isTRUE(tight$primary_usable) &&
    isTRUE(bobyqa$primary_usable)
  lr_agree <- both_primary && q1v11_candidate_agreement(
    tight$LR_raw,
    bobyqa$LR_raw,
    1e-6
  )
  full_agree <- both_primary && q1v11_candidate_agreement(
    tight$full_logLik,
    bobyqa$full_logLik,
    1e-6
  )
  null_agree <- both_primary && q1v11_candidate_agreement(
    tight$null_logLik,
    bobyqa$null_logLik,
    1e-6
  )
  agreement <- both_primary && lr_agree && full_agree && null_agree
  resolution_pattern <- if (both_primary && agreement) {
    "both_candidates_primary_usable_and_agree"
  } else if (both_primary) {
    "both_candidates_primary_usable_but_disagree"
  } else if (isTRUE(tight$primary_usable)) {
    "tight_nloptwrap_only"
  } else if (isTRUE(bobyqa$primary_usable)) {
    "bobyqa_only"
  } else {
    "unresolved_by_candidates"
  }

  data.frame(
    task_id = task_id,
    cell_id = current_row$cell_id,
    outer = current_row$outer,
    replicate = current_row$replicate,
    seed = current_row$seed,
    original_status = current_row$original_status,
    original_diagnostic_class = current_row$original_diagnostic_class,
    current_status = current_row$status,
    current_LR_raw_even_if_invalid = current_row$LR_raw,
    tight_status = tight$status,
    tight_reason = tight$reason,
    tight_primary_usable = tight$primary_usable,
    tight_zero_sensitivity_usable = tight$zero_sensitivity_usable,
    tight_all_zero_sensitivity_usable = tight$all_zero_sensitivity_usable,
    tight_LR_raw = tight$LR_raw,
    tight_full_logLik = tight$full_logLik,
    tight_null_logLik = tight$null_logLik,
    tight_full_logLik_gain_vs_current =
      tight$full_logLik - current_row$full_logLik,
    tight_null_logLik_gain_vs_current =
      tight$null_logLik - current_row$null_logLik,
    tight_full_singular = tight$full_singular,
    bobyqa_status = bobyqa$status,
    bobyqa_reason = bobyqa$reason,
    bobyqa_primary_usable = bobyqa$primary_usable,
    bobyqa_zero_sensitivity_usable = bobyqa$zero_sensitivity_usable,
    bobyqa_all_zero_sensitivity_usable =
      bobyqa$all_zero_sensitivity_usable,
    bobyqa_LR_raw = bobyqa$LR_raw,
    bobyqa_full_logLik = bobyqa$full_logLik,
    bobyqa_null_logLik = bobyqa$null_logLik,
    bobyqa_full_logLik_gain_vs_current =
      bobyqa$full_logLik - current_row$full_logLik,
    bobyqa_null_logLik_gain_vs_current =
      bobyqa$null_logLik - current_row$null_logLik,
    bobyqa_full_singular = bobyqa$full_singular,
    no_derivs_status = no_derivs$status,
    no_derivs_LR_raw = no_derivs$LR_raw,
    no_derivs_full_logLik = no_derivs$full_logLik,
    no_derivs_null_logLik = no_derivs$null_logLik,
    no_derivs_full_logLik_gain_vs_current =
      no_derivs$full_logLik - current_row$full_logLik,
    no_derivs_null_logLik_gain_vs_current =
      no_derivs$null_logLik - current_row$null_logLik,
    no_derivs_inferentially_eligible = FALSE,
    candidate_LR_abs_difference = if (both_primary) {
      abs(tight$LR_raw - bobyqa$LR_raw)
    } else {
      NA_real_
    },
    candidate_full_logLik_abs_difference = if (both_primary) {
      abs(tight$full_logLik - bobyqa$full_logLik)
    } else {
      NA_real_
    },
    candidate_null_logLik_abs_difference = if (both_primary) {
      abs(tight$null_logLik - bobyqa$null_logLik)
    } else {
      NA_real_
    },
    candidate_LR_agreement = lr_agree,
    candidate_full_objective_agreement = full_agree,
    candidate_null_objective_agreement = null_agree,
    candidate_pair_agreement = agreement,
    resolution_pattern = resolution_pattern,
    stringsAsFactors = FALSE
  )
})
task_summary <- do.call(rbind, task_summary_rows)
utils::write.csv(
  task_summary,
  file.path(OUT, "q1_optimizer_rescue_task_summary.csv"),
  row.names = FALSE
)

summarize_method_subset <- function(rows, class_label) {
  data.frame(
    method_id = rows$method_id[[1]],
    method_role = rows$method_role[[1]],
    original_diagnostic_class = class_label,
    n = nrow(rows),
    full_refit_errors = sum(!rows$full_refit_ok),
    null_refit_errors = sum(!rows$null_refit_ok),
    pair_diagnostic_valid = sum(
      rows$full_diag_valid & rows$null_diag_valid
    ),
    primary_usable = sum(rows$primary_usable),
    zero_sensitivity_usable = sum(rows$zero_sensitivity_usable),
    all_zero_sensitivity_usable = sum(
      rows$all_zero_sensitivity_usable
    ),
    failures = sum(rows$status == "failure"),
    small_negative = sum(rows$status == "small_negative"),
    material_negative = sum(rows$status == "material_negative"),
    full_singular = sum(rows$full_singular, na.rm = TRUE),
    null_singular = sum(rows$null_singular, na.rm = TRUE),
    full_hessian_nonpositive = sum(
      !rows$full_hessian_positive_definite,
      na.rm = TRUE
    ),
    null_hessian_nonpositive = sum(
      !rows$null_hessian_positive_definite,
      na.rm = TRUE
    ),
    median_LR_when_finite = if (any(is.finite(rows$LR_raw))) {
      stats::median(rows$LR_raw[is.finite(rows$LR_raw)])
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
}

method_summary_rows <- list()
counter <- 0L
for (method_id in names(control_panel)) {
  method_rows <- results[results$method_id == method_id, , drop = FALSE]
  counter <- counter + 1L
  method_summary_rows[[counter]] <- summarize_method_subset(
    method_rows,
    "ALL"
  )
  for (class_label in sort(unique(method_rows$original_diagnostic_class))) {
    counter <- counter + 1L
    method_summary_rows[[counter]] <- summarize_method_subset(
      method_rows[
        method_rows$original_diagnostic_class == class_label,
        ,
        drop = FALSE
      ],
      class_label
    )
  }
}
method_summary <- do.call(rbind, method_summary_rows)
utils::write.csv(
  method_summary,
  file.path(OUT, "q1_optimizer_rescue_method_summary.csv"),
  row.names = FALSE
)

outer_summary_rows <- lapply(
  sort(unique(results$outer_key)),
  function(key) {
    rows <- task_summary[
      paste(task_summary$cell_id, task_summary$outer, sep = "::") == key,
      ,
      drop = FALSE
    ]
    data.frame(
      outer_key = key,
      cell_id = rows$cell_id[[1]],
      outer = rows$outer[[1]],
      n_problem_tasks = nrow(rows),
      both_candidates_usable_and_agree = sum(
        rows$resolution_pattern ==
          "both_candidates_primary_usable_and_agree"
      ),
      both_candidates_usable_but_disagree = sum(
        rows$resolution_pattern ==
          "both_candidates_primary_usable_but_disagree"
      ),
      tight_only = sum(rows$resolution_pattern == "tight_nloptwrap_only"),
      bobyqa_only = sum(rows$resolution_pattern == "bobyqa_only"),
      unresolved = sum(
        rows$resolution_pattern == "unresolved_by_candidates"
      ),
      stringsAsFactors = FALSE
    )
  }
)
outer_summary <- do.call(rbind, outer_summary_rows)
utils::write.csv(
  outer_summary,
  file.path(OUT, "q1_optimizer_rescue_outer_summary.csv"),
  row.names = FALSE
)

operational_gate <- data.frame(
  check = c(
    "audited_v10_input_anchors",
    "saved_inner_seed_streams",
    "selected_task_method_completeness",
    "requested_optimizers_honored",
    "observed_outer_reproduction",
    "current_control_inner_reproduction",
    "calc_derivs_false_never_inferential"
  ),
  value = c(
    "120 outer; 908 problems; 866/14/28 classes; 5 caps",
    sprintf("%d/%d", sum(expected_seed == problems$seed), nrow(problems)),
    sprintf("%d/%d rows", nrow(results), expected_result_rows),
    sprintf(
      "%d/%d",
      sum(results$optimizer_control_honored),
      nrow(results)
    ),
    sprintf(
      "%d/%d",
      sum(observed_reproduction$observed_match),
      nrow(observed_reproduction)
    ),
    sprintf("%d/%d", sum(baseline_reproduced), length(baseline_reproduced)),
    as.character(all(
      !task_summary$no_derivs_inferentially_eligible
    ))
  ),
  pass = c(
    input_anchors_ok,
    all(expected_seed == problems$seed),
    results_complete,
    all(results$optimizer_control_honored),
    all(observed_reproduction$observed_match),
    all(baseline_reproduced),
    all(!task_summary$no_derivs_inferentially_eligible)
  ),
  stringsAsFactors = FALSE
)
utils::write.csv(
  operational_gate,
  file.path(OUT, "q1_optimizer_rescue_operational_gate.csv"),
  row.names = FALSE
)

writeLines(
  capture.output(sessionInfo()),
  file.path(OUT, "sessionInfo_q1_optimizer_rescue_v11.txt")
)

print(method_summary[method_summary$original_diagnostic_class == "ALL", ])
print(table(task_summary$resolution_pattern))
if (!all(operational_gate$pass)) {
  stop(
    "FAIL: v11 diagnostic did not reproduce its reference inputs completely."
  )
}

if (MODE == "smoke") {
  message(
    "PASS: v11 smoke reproduced every selected v10 row and completed all ",
    "optimizer panels. Run the full diagnostic only after smoke audit."
  )
} else {
  message(
    "PASS: v11 full optimizer-rescue diagnostic completed operationally. ",
    "The script does not select a rescue rule; evaluate candidate agreement ",
    "and unresolved rows before changing the bootstrap protocol."
  )
}
