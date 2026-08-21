# Question-1 bootstrap bookkeeping and Monte Carlo summaries.

.sa_q1_failure_counts <- function(replicates) {
  bad <- replicates[
    replicates$status %in% c("failure", "material_negative"),
    ,
    drop = FALSE
  ]

  if (nrow(bad) == 0L) {
    return(integer())
  }

  reason <- bad$reason
  reason[is.na(reason) | !nzchar(reason)] <- "unspecified"

  sort(table(reason), decreasing = TRUE)
}


.sa_q1_escalation_advice <- function(
    summary,
    alpha = 0.05,
    target_B = 4999L) {

  if (!is.numeric(alpha) || length(alpha) != 1L ||
      !is.finite(alpha) || alpha <= 0 || alpha >= 1) {
    stop("`alpha` must lie strictly between 0 and 1.", call. = FALSE)
  }

  if (summary$denominator_drop == 0L) {
    return(list(
      evaluable = FALSE,
      recommended = FALSE,
      target_B = as.integer(target_B),
      reason = "no_usable_drop_convention_replicates"
    ))
  }

  floor_limited <- summary$exceedances_drop == 0L
  interval <- summary$mc_interval_drop

  crosses_alpha <- all(is.finite(interval)) &&
    interval[[1]] <= alpha &&
    interval[[2]] >= alpha

  reason <- if (floor_limited && crosses_alpha) {
    "floor_limited_and_MC_interval_crosses_alpha"
  } else if (floor_limited) {
    "floor_limited"
  } else if (crosses_alpha) {
    "MC_interval_crosses_alpha"
  } else {
    "none"
  }

  list(
    evaluable = TRUE,
    recommended = floor_limited || crosses_alpha,
    target_B = as.integer(target_B),
    reason = reason,
    alpha = alpha
  )
}


.sa_q1_summarize_replicates <- function(
    replicates,
    observed_LR,
    conf_level = 0.95,
    alpha = 0.05,
    escalation_target_B = 4999L) {

  required <- c(
    "replicate",
    "status",
    "LR_raw",
    "LR_zero",
    "reason"
  )

  missing <- setdiff(required, names(replicates))
  if (length(missing) > 0L) {
    stop(
      sprintf(
        "`replicates` is missing: %s",
        paste(missing, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  if (!is.numeric(observed_LR) ||
      length(observed_LR) != 1L ||
      !is.finite(observed_LR) ||
      observed_LR < 0) {
    stop(
      "`observed_LR` must be one finite nonnegative number.",
      call. = FALSE
    )
  }

  allowed <- c(
    "positive",
    "zero",
    "small_negative",
    "material_negative",
    "failure"
  )

  unknown <- setdiff(unique(replicates$status), allowed)
  if (length(unknown) > 0L) {
    stop(
      sprintf(
        "Unknown replicate status: %s",
        paste(unknown, collapse = ", ")
      ),
      call. = FALSE
    )
  }

  drop_use <- replicates$status %in% c("positive", "zero")
  zero_use <- replicates$status %in% c(
    "positive",
    "zero",
    "small_negative"
  )
  all_zero_use <- replicates$status %in% c(
    "positive",
    "zero",
    "small_negative",
    "material_negative"
  )

  LR_drop <- replicates$LR_raw[drop_use]
  LR_zero <- replicates$LR_zero[zero_use]
  LR_all_zero <- ifelse(
    replicates$status[all_zero_use] %in% c(
      "small_negative",
      "material_negative"
    ),
    0,
    replicates$LR_raw[all_zero_use]
  )

  if (any(!is.finite(LR_drop))) {
    stop(
      "A drop-convention usable replicate has nonfinite LR.",
      call. = FALSE
    )
  }

  if (any(!is.finite(LR_zero))) {
    stop(
      "A zero-convention usable replicate has nonfinite LR.",
      call. = FALSE
    )
  }

  if (any(!is.finite(LR_all_zero))) {
    stop(
      "An all-negative-zero usable replicate has nonfinite LR.",
      call. = FALSE
    )
  }

  n_drop <- length(LR_drop)
  n_zero <- length(LR_zero)
  n_all_zero <- length(LR_all_zero)

  k_drop <- sum(LR_drop >= observed_LR)
  k_zero <- sum(LR_zero >= observed_LR)
  k_all_zero <- sum(LR_all_zero >= observed_LR)

  mc_drop <- .sa_q1_exact_binomial_interval(
    k = k_drop,
    n = n_drop,
    conf_level = conf_level
  )
  mc_zero <- .sa_q1_exact_binomial_interval(
    k = k_zero,
    n = n_zero,
    conf_level = conf_level
  )
  mc_all_zero <- .sa_q1_exact_binomial_interval(
    k = k_all_zero,
    n = n_all_zero,
    conf_level = conf_level
  )

  out <- list(
    B_requested = nrow(replicates),
    n_positive = sum(replicates$status == "positive"),
    n_zero = sum(replicates$status == "zero"),
    n_small_negative = sum(
      replicates$status == "small_negative"
    ),
    n_material_negative = sum(
      replicates$status == "material_negative"
    ),
    n_failure = sum(replicates$status == "failure"),
    denominator_drop = n_drop,
    denominator_zero = n_zero,
    denominator_all_zero = n_all_zero,
    usable_rate_drop = n_drop / max(1L, nrow(replicates)),
    usable_rate_zero = n_zero / max(1L, nrow(replicates)),
    usable_rate_all_zero = n_all_zero / max(1L, nrow(replicates)),
    exceedances_drop = k_drop,
    exceedances_zero = k_zero,
    exceedances_all_zero = k_all_zero,
    p_drop = .sa_q1_plus_one_p(k_drop, n_drop),
    p_zero = .sa_q1_plus_one_p(k_zero, n_zero),
    p_all_zero = .sa_q1_plus_one_p(k_all_zero, n_all_zero),
    # These intervals target the fitted-null exceedance probabilities k/n.
    # They are not confidence intervals for the plus-one Monte Carlo p-value.
    exceedance_probability_interval_drop = mc_drop,
    exceedance_probability_interval_zero = mc_zero,
    exceedance_probability_interval_all_zero = mc_all_zero,
    # Backward-compatible aliases retained for frozen v1.1 output readers.
    mc_interval_drop = mc_drop,
    mc_interval_zero = mc_zero,
    mc_interval_all_zero = mc_all_zero,
    attainable_floor_drop = if (n_drop > 0L) {
      1 / (n_drop + 1)
    } else {
      NA_real_
    },
    attainable_floor_zero = if (n_zero > 0L) {
      1 / (n_zero + 1)
    } else {
      NA_real_
    },
    attainable_floor_all_zero = if (n_all_zero > 0L) {
      1 / (n_all_zero + 1)
    } else {
      NA_real_
    },
    conf_level = conf_level,
    primary_convention = "drop",
    dual_convention_required = any(
      replicates$status %in% c(
        "small_negative",
        "material_negative"
      )
    ),
    failure_counts = .sa_q1_failure_counts(replicates),
    bootstrap_distribution = list(
      drop = as.numeric(LR_drop),
      zero = as.numeric(LR_zero),
      all_zero = as.numeric(LR_all_zero)
    )
  )

  out$escalation <- .sa_q1_escalation_advice(
    summary = out,
    alpha = alpha,
    target_B = escalation_target_B
  )

  out
}
