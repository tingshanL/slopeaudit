#!/usr/bin/env Rscript

# Focused follow-up for q1_targeted_stress.R.
#
# The ordinary stress output is compact. This script reads
# that outer-run table, selects runs with a material negative LR, at least one
# small negative LR, or a large inner-failure cluster, and deterministically
# reruns only those cases while retaining every inner-bootstrap row.

suppressPackageStartupMessages({
  library(lme4)
  library(slopeaudit)
})

INNER_B <- as.integer(
  Sys.getenv("SLOPEAUDIT_Q1_STRESS_B", "99")
)
FAILURE_THRESHOLD <- as.integer(
  Sys.getenv("SLOPEAUDIT_Q1_STRESS_FAILURE_THRESHOLD", "10")
)
INPUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_STRESS_INPUT",
  "validation/extended/results/q1_stress/q1_targeted_stress_replicates.csv"
)
OUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_OUT",
  "validation/extended/results/q1_stress_diagnostics"
)

if (!is.finite(INNER_B) || INNER_B < 1L) {
  stop("SLOPEAUDIT_Q1_STRESS_B must be a positive integer.", call. = FALSE)
}
if (!is.finite(FAILURE_THRESHOLD) || FAILURE_THRESHOLD < 1L) {
  stop(
    "SLOPEAUDIT_Q1_STRESS_FAILURE_THRESHOLD must be a positive integer.",
    call. = FALSE
  )
}
if (!file.exists(INPUT)) {
  stop("Stress outer-run CSV not found: ", INPUT, call. = FALSE)
}

dir.create(OUT, recursive = TRUE, showWarnings = FALSE)

make_design <- function(seed, scenario) {
  set.seed(seed)

  n_id <- 32

  if (scenario == "low_information") {
    possible_times <- c(0, 0.25, 0.5, 0.75, 1)
    n_range <- 2:3
    residual_sd <- 3.5
  } else {
    possible_times <- c(0, 1, 2, 4, 7)
    n_range <- 2:5
    residual_sd <- 1.5
  }

  rows <- lapply(seq_len(n_id), function(i) {
    n_i <- sample(n_range, 1)
    ti <- sort(sample(possible_times, n_i, replace = FALSE))

    data.frame(
      id = factor(i),
      time = ti
    )
  })

  dat <- do.call(rbind, rows)
  dat$id <- factor(dat$id)

  if (scenario == "null") {
    b0 <- rnorm(n_id, 0, 2)
    b1 <- rep(0, n_id)
  } else if (scenario == "strong") {
    b0 <- rnorm(n_id, 0, 2)
    b1 <- rnorm(n_id, 0, 2.5)
  } else if (scenario == "low_information") {
    b0 <- rnorm(n_id, 0, 2)
    b1 <- rnorm(n_id, 0, 0.6)
  } else if (scenario == "near_rank_one") {
    z <- rnorm(n_id)
    b0 <- 2 * z
    b1 <- 0.9 * z + rnorm(n_id, 0, 0.015)
  } else {
    stop("Unknown stress scenario: ", scenario, call. = FALSE)
  }

  idx <- as.integer(dat$id)

  dat$y <- 5 + 0.4 * dat$time +
    b0[idx] +
    b1[idx] * dat$time +
    rnorm(nrow(dat), 0, residual_sd)

  dat
}

numeric_equal <- function(x, y) {
  isTRUE(all.equal(
    as.numeric(x),
    as.numeric(y),
    tolerance = 1e-12,
    check.attributes = FALSE
  ))
}

format_failure_counts <- function(x) {
  if (length(x) == 0L) {
    return("")
  }

  paste(
    paste0(names(x), "=", as.integer(x)),
    collapse = " | "
  )
}

min_status_lr <- function(replicates, status) {
  z <- replicates$LR_raw[replicates$status == status]
  if (length(z) == 0L) {
    return(NA_real_)
  }
  min(z, na.rm = TRUE)
}

outer <- utils::read.csv(
  INPUT,
  stringsAsFactors = FALSE,
  check.names = FALSE
)

required <- c(
  "outer",
  "scenario",
  "design_seed",
  "bootstrap_seed",
  "status",
  "observed_LR",
  "p_drop",
  "p_zero",
  "usable_rate_drop",
  "usable_rate_zero",
  "failures",
  "small_negative",
  "material_negative"
)

missing <- setdiff(required, names(outer))
if (length(missing) > 0L) {
  stop(
    "Stress outer-run CSV is missing: ",
    paste(missing, collapse = ", "),
    call. = FALSE
  )
}

selected <- outer[
  outer$status == "ok" &
    (
      outer$material_negative > 0L |
        outer$small_negative > 0L |
        outer$failures >= FAILURE_THRESHOLD
    ),
  ,
  drop = FALSE
]

selected <- selected[
  order(match(
    selected$scenario,
    c("null", "strong", "low_information", "near_rank_one")
  ), selected$outer),
  ,
  drop = FALSE
]

if (nrow(selected) == 0L) {
  stop(
    "No outer runs met the focused diagnostic selection rule.",
    call. = FALSE
  )
}

ctrl <- lmerControl(
  optimizer = "nloptwrap",
  check.conv.singular = "message"
)

run_rows <- vector("list", nrow(selected))
inner_rows <- vector("list", nrow(selected))

progress_bar <- utils::txtProgressBar(
  min = 0,
  max = nrow(selected),
  initial = 0,
  style = 3
)

for (j in seq_len(nrow(selected))) {
  old <- selected[j, , drop = FALSE]
  dat <- make_design(old$design_seed[[1]], old$scenario[[1]])

  full <- lmer(
    y ~ time + (1 + time | id),
    data = dat,
    REML = TRUE,
    control = ctrl
  )
  null <- lmer(
    y ~ time + (1 | id),
    data = dat,
    REML = TRUE,
    control = ctrl
  )

  ans <- slopeaudit:::.sa_q1_bootstrap(
    full,
    null,
    target = "time",
    refit_control = ctrl,
    B = INNER_B,
    seed = old$bootstrap_seed[[1]],
    retain_replicates = TRUE,
    progress = FALSE
  )

  if (ans$status != "ok" || is.null(ans$replicates)) {
    stop(
      "Focused diagnostic failed for ",
      old$scenario[[1]],
      " outer ",
      old$outer[[1]],
      ".",
      call. = FALSE
    )
  }

  reps <- ans$replicates
  reps <- cbind(
    data.frame(
      outer = old$outer[[1]],
      scenario = old$scenario[[1]],
      design_seed = old$design_seed[[1]],
      bootstrap_seed = old$bootstrap_seed[[1]],
      stringsAsFactors = FALSE
    )[rep(1L, nrow(reps)), , drop = FALSE],
    reps
  )
  rownames(reps) <- NULL
  inner_rows[[j]] <- reps

  reproduction_ok <- all(c(
    numeric_equal(ans$observed$LR_raw, old$observed_LR[[1]]),
    numeric_equal(ans$bootstrap$p_drop, old$p_drop[[1]]),
    numeric_equal(ans$bootstrap$p_zero, old$p_zero[[1]]),
    numeric_equal(
      ans$bootstrap$usable_rate_drop,
      old$usable_rate_drop[[1]]
    ),
    numeric_equal(
      ans$bootstrap$usable_rate_zero,
      old$usable_rate_zero[[1]]
    ),
    identical(
      as.integer(ans$bootstrap$n_failure),
      as.integer(old$failures[[1]])
    ),
    identical(
      as.integer(ans$bootstrap$n_small_negative),
      as.integer(old$small_negative[[1]])
    ),
    identical(
      as.integer(ans$bootstrap$n_material_negative),
      as.integer(old$material_negative[[1]])
    )
  ))

  run_rows[[j]] <- data.frame(
    outer = old$outer[[1]],
    scenario = old$scenario[[1]],
    design_seed = old$design_seed[[1]],
    bootstrap_seed = old$bootstrap_seed[[1]],
    reproduction_ok = reproduction_ok,
    observed_LR = ans$observed$LR_raw,
    p_drop = ans$bootstrap$p_drop,
    p_zero = ans$bootstrap$p_zero,
    usable_rate_drop = ans$bootstrap$usable_rate_drop,
    usable_rate_zero = ans$bootstrap$usable_rate_zero,
    failures = ans$bootstrap$n_failure,
    small_negative = ans$bootstrap$n_small_negative,
    material_negative = ans$bootstrap$n_material_negative,
    min_small_negative_LR = min_status_lr(
      ans$replicates,
      "small_negative"
    ),
    min_material_negative_LR = min_status_lr(
      ans$replicates,
      "material_negative"
    ),
    failure_counts = format_failure_counts(
      ans$bootstrap$failure_counts
    ),
    stringsAsFactors = FALSE
  )

  utils::setTxtProgressBar(progress_bar, j)
}

close(progress_bar)

run_table <- do.call(rbind, run_rows)
inner_table <- do.call(rbind, inner_rows)
problem_table <- inner_table[
  inner_table$status %in% c(
    "small_negative",
    "material_negative",
    "failure"
  ),
  ,
  drop = FALSE
]

utils::write.csv(
  run_table,
  file.path(OUT, "q1_targeted_stress_diagnostic_runs.csv"),
  row.names = FALSE
)
utils::write.csv(
  inner_table,
  file.path(OUT, "q1_targeted_stress_inner_replicates.csv"),
  row.names = FALSE
)
utils::write.csv(
  problem_table,
  file.path(OUT, "q1_targeted_stress_inner_problems.csv"),
  row.names = FALSE
)

print(run_table)

if (!all(run_table$reproduction_ok)) {
  stop(
    "At least one focused rerun did not reproduce its stored outer summary.",
    call. = FALSE
  )
}

cat(
  "\nPASS: focused stress diagnostics reproduced all selected outer runs.\n",
  sep = ""
)
