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

MODE <- tolower(Sys.getenv("SLOPEAUDIT_Q1_CAL_MODE", "smoke"))
mode_defaults <- list(
  smoke = list(outer = 2L, usable_B = 39L, workers = 1L),
  pilot = list(outer = 20L, usable_B = 99L, workers = 1L),
  full = list(outer = 999L, usable_B = 999L, workers = 1L)
)
if (!MODE %in% names(mode_defaults)) {
  stop("SLOPEAUDIT_Q1_CAL_MODE must be smoke, pilot, or full.")
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

OUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_OUT",
  file.path(
    "validation",
    "extended",
    "results",
    paste0("q1_four_design_", MODE, "_v10")
  )
)
FIXTURE_PATH <- Sys.getenv(
  "SLOPEAUDIT_Q1_IRREGULAR_SCHEDULE",
  file.path(
    "validation",
    "reference",
    "four_design_dgp",
    "irregular_fluid_cognition_schedule.csv"
  )
)

if (!file.exists(FIXTURE_PATH)) {
  stop(
    paste0(
      "Missing exact irregular schedule fixture: ", FIXTURE_PATH, "\n",
      "Run validation/provenance/q1_freeze_irregular_schedule_fixture.py ",
      "first. Do not substitute an approximate schedule."
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

calibration_control <- lmerControl(
  optimizer = "nloptwrap",
  restart_edge = TRUE,
  boundary.tol = 1e-5,
  calc.derivs = TRUE,
  optCtrl = list(maxeval = 200000),
  check.conv.singular = "message"
)

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
  version = "q1_four_design_calibration_v10_candidate",
  mode = MODE,
  outer_per_cell = OUTER_N,
  usable_B = USABLE_B,
  attempt_cap = ATTEMPT_CAP,
  alpha = ALPHA,
  rejection_rule = "plus-one p <= alpha",
  primary_inner_convention = "fixed positive/zero usable denominator",
  master_seed = MASTER_SEED,
  irregular_fixture_md5 = unname(tools::md5sum(FIXTURE_PATH)),
  cell_manifest = cell_manifest,
  control = calibration_control
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
    elapsed_seconds) {
  data.frame(
    cell_id = cell$cell_id,
    cell_family = cell$cell_family,
    outer = outer,
    design_seed = design_seed,
    bootstrap_seed = bootstrap_seed,
    status = status,
    failure_reason = reason,
    observed_LR = NA_real_,
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
    elapsed_seconds = elapsed_seconds,
    stringsAsFactors = FALSE
  )
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

  full_capture <- slopeaudit:::.sa_q1_capture(
    lmer(
      y ~ time + (1 + time | id),
      data = dat,
      REML = TRUE,
      control = calibration_control
    )
  )
  null_capture <- slopeaudit:::.sa_q1_capture(
    lmer(
      y ~ time + (1 | id),
      data = dat,
      REML = TRUE,
      control = calibration_control
    )
  )

  if (!is.null(full_capture$error) || !is.null(null_capture$error)) {
    reasons <- c(
      if (!is.null(full_capture$error)) {
        paste0("full:", full_capture$error)
      },
      if (!is.null(null_capture$error)) {
        paste0("null:", null_capture$error)
      }
    )
    return(list(
      summary = observed_failure(
        cell,
        task$outer,
        task$design_seed,
        task$bootstrap_seed,
        "observed_fit_failure",
        paste(reasons, collapse = ";"),
        proc.time()[["elapsed"]] - started
      ),
      problems = empty_problem_rows()
    ))
  }

  full <- full_capture$value
  null <- null_capture$value
  covariance <- as.matrix(VarCorr(full)[[1]])
  geometry <- slopeaudit:::.sa_covariance_diagnostics(
    covariance,
    target_index = 2L
  )

  answer <- tryCatch(
    slopeaudit:::.sa_q1_bootstrap_fixed_usable(
      full,
      null,
      target = "time",
      usable_B = USABLE_B,
      attempt_cap = ATTEMPT_CAP,
      seed = task$bootstrap_seed,
      alpha = ALPHA,
      refit_control = calibration_control,
      retain_replicates = TRUE,
      progress = FALSE
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
      proc.time()[["elapsed"]] - started
    )
    if (
      !inherits(answer, "error") &&
        answer$status == "attempt_cap_reached"
    ) {
      partial <- answer$bootstrap
      failure_summary$observed_LR <- answer$observed$LR_for_inference
      failure_summary$full_singular <-
        answer$observed$full_diagnostics$singular
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
    }
    return(list(
      summary = failure_summary,
      problems = if (
        !inherits(answer, "error") &&
          !is.null(answer$replicates)
      ) {
        problem <- answer$replicates[
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
    full_singular = answer$observed$full_diagnostics$singular,
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
    elapsed_seconds = proc.time()[["elapsed"]] - started,
    stringsAsFactors = FALSE
  )

  stopifnot(summary$denominator_drop == USABLE_B)
  stopifnot(
    summary$attempts_evaluated ==
      USABLE_B + summary$inner_failures +
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
utils::write.csv(
  task_manifest,
  file.path(OUT, "q1_four_design_task_manifest.csv"),
  row.names = FALSE
)

checkpoint_path <- file.path(OUT, "q1_four_design_checkpoint.rds")
outer_csv <- file.path(OUT, "q1_four_design_outer_runs.csv")
problem_csv <- file.path(OUT, "q1_four_design_inner_problems.csv")

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
    observed_fit_failures = sum(
      all_cell$status == "observed_fit_failure"
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
  file.path(OUT, "sessionInfo_q1_four_design_v10.txt")
)

print(calibration_summary)
if (all(calibration_summary$provisional_gate == "PASS")) {
  message("PASS: completed calibration run passed all provisional gates.")
} else {
  message("DIAGNOSE: inspect cells marked DIAGNOSE before continuing.")
}
