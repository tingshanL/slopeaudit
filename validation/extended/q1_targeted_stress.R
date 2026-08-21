#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(lme4)
  library(slopeaudit)
})

MASTER_SEED <- as.integer(
  Sys.getenv("SLOPEAUDIT_Q1_STRESS_SEED", "20260811")
)
INNER_B <- as.integer(
  Sys.getenv("SLOPEAUDIT_Q1_STRESS_B", "99")
)
OUTER_N <- as.integer(
  Sys.getenv("SLOPEAUDIT_Q1_STRESS_OUTER", "20")
)
OUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_OUT",
  "validation/extended/results/q1_stress"
)

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
    stop("Unknown stress scenario.")
  }

  idx <- as.integer(dat$id)

  dat$y <- 5 + 0.4 * dat$time +
    b0[idx] +
    b1[idx] * dat$time +
    rnorm(nrow(dat), 0, residual_sd)

  dat
}


run_one <- function(
  outer,
  scenario,
  design_seed,
  bootstrap_seed
) {
  dat <- make_design(design_seed, scenario)

  ctrl <- lmerControl(
    optimizer = "nloptwrap",
    check.conv.singular = "message"
  )

  full <- tryCatch(
    lmer(
      y ~ time + (1 + time | id),
      data = dat,
      REML = TRUE,
      control = ctrl
    ),
    error = function(e) e
  )

  null <- tryCatch(
    lmer(
      y ~ time + (1 | id),
      data = dat,
      REML = TRUE,
      control = ctrl
    ),
    error = function(e) e
  )

  if (inherits(full, "error") || inherits(null, "error")) {
    fit_failure_reason <- paste(
      c(
        if (inherits(full, "error")) {
          paste0("full:", conditionMessage(full))
        },
        if (inherits(null, "error")) {
          paste0("null:", conditionMessage(null))
        }
      ),
      collapse = "; "
    )

    return(data.frame(
      outer = outer,
      scenario = scenario,
      design_seed = design_seed,
      bootstrap_seed = bootstrap_seed,
      status = "observed_fit_failure",
      failure_reason = fit_failure_reason,
      observed_LR = NA_real_,
      full_singular = NA,
      lambda_min_over_trace = NA_real_,
      target_max_abs_correlation = NA_real_,
      p_drop = NA_real_,
      p_zero = NA_real_,
      usable_rate_drop = NA_real_,
      usable_rate_zero = NA_real_,
      failures = NA_integer_,
      small_negative = NA_integer_,
      material_negative = NA_integer_,
      stringsAsFactors = FALSE
    ))
  }

  G <- as.matrix(VarCorr(full)[[1]])
  gd <- slopeaudit:::.sa_covariance_diagnostics(
    G = G,
    target_index = 2
  )

  ans <- tryCatch(
    slopeaudit:::.sa_q1_bootstrap(
      full,
      null,
      target = "time",
      refit_control = ctrl,
      B = INNER_B,
      seed = bootstrap_seed,
      retain_replicates = FALSE,
      progress = FALSE
    ),
    error = function(e) e
  )

  if (inherits(ans, "error") || ans$status != "ok") {
    q1_failure_reason <- if (inherits(ans, "error")) {
      conditionMessage(ans)
    } else if (
      !is.null(ans$observed$reason) &&
        length(ans$observed$reason) == 1L &&
        !is.na(ans$observed$reason) &&
        nzchar(ans$observed$reason)
    ) {
      paste0(ans$status, ":", ans$observed$reason)
    } else {
      paste0("status:", ans$status)
    }

    return(data.frame(
      outer = outer,
      scenario = scenario,
      design_seed = design_seed,
      bootstrap_seed = bootstrap_seed,
      status = "q1_failure",
      failure_reason = q1_failure_reason,
      observed_LR = NA_real_,
      full_singular = lme4::isSingular(
        full,
        tol = ctrl$checkConv$check.conv.singular$tol
      ),
      lambda_min_over_trace = gd$lambda_min_over_trace,
      target_max_abs_correlation = gd$target_max_abs_correlation,
      p_drop = NA_real_,
      p_zero = NA_real_,
      usable_rate_drop = NA_real_,
      usable_rate_zero = NA_real_,
      failures = NA_integer_,
      small_negative = NA_integer_,
      material_negative = NA_integer_,
      stringsAsFactors = FALSE
    ))
  }

  data.frame(
    outer = outer,
    scenario = scenario,
    design_seed = design_seed,
    bootstrap_seed = bootstrap_seed,
    status = "ok",
    failure_reason = "",
    observed_LR = ans$observed$LR_raw,
    full_singular = ans$observed$full_diagnostics$singular,
    lambda_min_over_trace = gd$lambda_min_over_trace,
    target_max_abs_correlation = gd$target_max_abs_correlation,
    p_drop = ans$bootstrap$p_drop,
    p_zero = ans$bootstrap$p_zero,
    usable_rate_drop = ans$bootstrap$usable_rate_drop,
    usable_rate_zero = ans$bootstrap$usable_rate_zero,
    failures = ans$bootstrap$n_failure,
    small_negative = ans$bootstrap$n_small_negative,
    material_negative = ans$bootstrap$n_material_negative,
    stringsAsFactors = FALSE
  )
}


scenarios <- c(
  "null",
  "strong",
  "low_information",
  "near_rank_one"
)

N_RUNS <- length(scenarios) * OUTER_N

# Draw both seed sets directly from the validated seed generator. Deriving an
# inner seed by integer addition can overflow when a sampled design seed lies
# near .Machine$integer.max. Sampling 2 * N_RUNS unique seeds also records an
# explicit, reproducible separation between data generation and bootstrapping.
all_seeds <- slopeaudit:::.sa_q1_make_replicate_seeds(
  B = 2 * N_RUNS,
  seed = MASTER_SEED
)

design_seeds <- all_seeds[seq_len(N_RUNS)]
bootstrap_seeds <- all_seeds[N_RUNS + seq_len(N_RUNS)]

rows <- vector("list", N_RUNS)
j <- 0L
progress_bar <- utils::txtProgressBar(
  min = 0,
  max = N_RUNS,
  initial = 0,
  style = 3
)

for (scenario in scenarios) {
  for (i in seq_len(OUTER_N)) {
    j <- j + 1L
    rows[[j]] <- run_one(
      outer = i,
      scenario = scenario,
      design_seed = design_seeds[[j]],
      bootstrap_seed = bootstrap_seeds[[j]]
    )
    utils::setTxtProgressBar(progress_bar, j)
  }
}

close(progress_bar)

result <- do.call(rbind, rows)

write.csv(
  result,
  file.path(OUT, "q1_targeted_stress_replicates.csv"),
  row.names = FALSE
)

ok <- result[result$status == "ok", , drop = FALSE]

summary_rows <- lapply(scenarios, function(s) {
  all_s <- result[result$scenario == s, , drop = FALSE]
  z <- ok[ok$scenario == s, , drop = FALSE]

  data.frame(
    scenario = s,
    n_requested = nrow(all_s),
    n_ok = nrow(z),
    observed_fit_failures = sum(
      all_s$status == "observed_fit_failure"
    ),
    q1_failures = sum(all_s$status == "q1_failure"),
    support_rate_drop = if (nrow(z)) {
      mean(z$p_drop < 0.05, na.rm = TRUE)
    } else {
      NA_real_
    },
    support_rate_zero = if (nrow(z)) {
      mean(z$p_zero < 0.05, na.rm = TRUE)
    } else {
      NA_real_
    },
    min_usable_rate_drop = if (nrow(z)) {
      min(z$usable_rate_drop, na.rm = TRUE)
    } else {
      NA_real_
    },
    min_usable_rate_zero = if (nrow(z)) {
      min(z$usable_rate_zero, na.rm = TRUE)
    } else {
      NA_real_
    },
    median_usable_rate_drop = if (nrow(z)) {
      median(z$usable_rate_drop, na.rm = TRUE)
    } else {
      NA_real_
    },
    total_inner_failures = if (nrow(z)) {
      sum(z$failures, na.rm = TRUE)
    } else {
      NA_integer_
    },
    max_inner_failures = if (nrow(z)) {
      max(z$failures, na.rm = TRUE)
    } else {
      NA_integer_
    },
    total_small_negative = if (nrow(z)) {
      sum(z$small_negative, na.rm = TRUE)
    } else {
      NA_integer_
    },
    total_material_negative = if (nrow(z)) {
      sum(z$material_negative, na.rm = TRUE)
    } else {
      NA_integer_
    },
    singular_rate = if (nrow(z)) {
      mean(z$full_singular, na.rm = TRUE)
    } else {
      NA_real_
    },
    median_lambda_min_over_trace = if (nrow(z)) {
      median(z$lambda_min_over_trace, na.rm = TRUE)
    } else {
      NA_real_
    },
    median_target_max_abs_correlation = if (nrow(z)) {
      median(z$target_max_abs_correlation, na.rm = TRUE)
    } else {
      NA_real_
    },
    stringsAsFactors = FALSE
  )
})

summary <- do.call(rbind, summary_rows)

write.csv(
  summary,
  file.path(OUT, "q1_targeted_stress_summary.csv"),
  row.names = FALSE
)

print(summary)
