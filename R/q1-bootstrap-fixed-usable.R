# Fixed-usable-replicate Question-1 bootstrap.
#
# This procedure draws until the primary convention has
# exactly `usable_B` finite, nonnegative likelihood-ratio replicates, subject
# to a declared attempts cap. The stopping rule and deterministic rescue
# hierarchy were validated in the v12--v14 Question-1 program.

.sa_q1_bootstrap_fixed_usable <- function(
    full_fit,
    null_fit,
    target,
    usable_B = 999L,
    attempt_cap = usable_B + max(50L, ceiling(0.25 * usable_B)),
    seed = 1L,
    conf_level = 0.95,
    alpha = 0.05,
    retain_replicates = FALSE,
    progress = interactive() && usable_B >= 50L,
    identity_tol = .sa_identity_tol,
    strict_controls = TRUE,
    escalation_target_B = 4999L,
    refit_control = NULL,
    simulate_fun = .sa_q1_simulate_null,
    refit_fun = .sa_q1_refit,
    one_replicate_fun = .sa_q1_one_bootstrap_replicate) {

  if (!is.numeric(usable_B) || length(usable_B) != 1L ||
      !is.finite(usable_B) || usable_B != floor(usable_B) ||
      usable_B < 1 || usable_B > .Machine$integer.max) {
    stop("`usable_B` must be one positive R integer.", call. = FALSE)
  }
  usable_B <- as.integer(usable_B)

  if (!is.numeric(attempt_cap) || length(attempt_cap) != 1L ||
      !is.finite(attempt_cap) || attempt_cap != floor(attempt_cap) ||
      attempt_cap < usable_B || attempt_cap > .Machine$integer.max) {
    stop(
      "`attempt_cap` must be an R integer at least as large as `usable_B`.",
      call. = FALSE
    )
  }
  attempt_cap <- as.integer(attempt_cap)

  seed <- .sa_q1_validate_seed(seed)
  conf_level <- .sa_q1_validate_probability(conf_level, "conf_level")
  alpha <- .sa_q1_validate_probability(alpha, "alpha")
  retain_replicates <- .sa_q1_validate_flag(
    retain_replicates,
    "retain_replicates"
  )
  progress <- .sa_q1_validate_flag(progress, "progress")
  strict_controls <- .sa_q1_validate_flag(
    strict_controls,
    "strict_controls"
  )

  if (!is.numeric(escalation_target_B) ||
      length(escalation_target_B) != 1L ||
      !is.finite(escalation_target_B) ||
      escalation_target_B != floor(escalation_target_B) ||
      escalation_target_B < 1 ||
      escalation_target_B > .Machine$integer.max) {
    stop(
      "`escalation_target_B` must be one positive R integer.",
      call. = FALSE
    )
  }
  escalation_target_B <- as.integer(escalation_target_B)

  if (!is.function(simulate_fun)) {
    stop("`simulate_fun` must be a function.", call. = FALSE)
  }
  if (!is.function(refit_fun)) {
    stop("`refit_fun` must be a function.", call. = FALSE)
  }
  if (!is.function(one_replicate_fun)) {
    stop("`one_replicate_fun` must be a function.", call. = FALSE)
  }

  resolved_control <- .sa_q1_resolve_refit_control(
    full_fit = full_fit,
    null_fit = null_fit,
    refit_control = refit_control
  )
  frozen_refit_control <- resolved_control$control

  observed <- .sa_q1_observed_lr(
    full_fit = full_fit,
    null_fit = null_fit,
    target = target,
    diagnostic_control = frozen_refit_control,
    identity_tol = identity_tol,
    strict_controls = strict_controls
  )

  if (!isTRUE(observed$valid_for_inference)) {
    return(structure(
      list(
        status = "invalid_observed",
        target = target,
        observed = observed,
        bootstrap = NULL,
        replicates = NULL,
        usable_target = usable_B,
        attempt_cap = attempt_cap,
        attempts_evaluated = 0L,
        target_reached = FALSE,
        master_seed = seed,
        simulation_source = "fitted null plug-in",
        likelihood_mode = observed$pair$likelihood_mode
      ),
      class = "slopeaudit_q1_internal"
    ))
  }

  rng_snapshot <- .sa_q1_rng_snapshot()
  on.exit(.sa_q1_rng_restore(rng_snapshot), add = TRUE)

  seeds <- .sa_q1_make_replicate_seeds(
    B = attempt_cap,
    seed = seed
  )

  rows <- vector("list", attempt_cap)
  attempts_evaluated <- 0L
  usable_drop <- 0L
  progress_bar <- NULL

  if (isTRUE(progress)) {
    progress_bar <- utils::txtProgressBar(
      min = 0,
      max = usable_B,
      initial = 0,
      style = 3
    )
    on.exit(close(progress_bar), add = TRUE)
  }

  for (attempt in seq_len(attempt_cap)) {
    row <- one_replicate_fun(
      replicate = attempt,
      seed = seeds[[attempt]],
      full_fit = full_fit,
      null_fit = null_fit,
      target = target,
      identity_tol = identity_tol,
      strict_controls = strict_controls,
      refit_control = frozen_refit_control,
      simulate_fun = simulate_fun,
      refit_fun = refit_fun
    )

    if (!is.data.frame(row) || nrow(row) != 1L ||
        !"status" %in% names(row)) {
      stop(
        "`one_replicate_fun` must return one standard replicate row.",
        call. = FALSE
      )
    }

    rows[[attempt]] <- row
    attempts_evaluated <- attempt

    if (row$status[[1]] %in% c("positive", "zero")) {
      usable_drop <- usable_drop + 1L
      if (isTRUE(progress)) {
        utils::setTxtProgressBar(progress_bar, usable_drop)
      }
    }

    if (usable_drop >= usable_B) {
      break
    }
  }

  replicates <- do.call(
    rbind,
    rows[seq_len(attempts_evaluated)]
  )
  rownames(replicates) <- NULL

  summary <- .sa_q1_summarize_replicates(
    replicates = replicates,
    observed_LR = observed$LR_for_inference,
    conf_level = conf_level,
    alpha = alpha,
    escalation_target_B = escalation_target_B
  )

  target_reached <- identical(
    as.integer(summary$denominator_drop),
    usable_B
  )
  rank_grid_value <- alpha * (usable_B + 1)
  exact_alpha_rank_grid <- abs(
    rank_grid_value - round(rank_grid_value)
  ) <= 100 * .Machine$double.eps

  structure(
    list(
      status = if (target_reached) "ok" else "attempt_cap_reached",
      target = target,
      pair = observed$pair,
      observed = observed,
      bootstrap = summary,
      replicates = if (isTRUE(retain_replicates)) {
        replicates
      } else {
        NULL
      },
      usable_target = usable_B,
      attempt_cap = attempt_cap,
      attempts_evaluated = as.integer(attempts_evaluated),
      top_up_attempts = as.integer(attempts_evaluated - usable_B),
      target_reached = target_reached,
      alpha = alpha,
      rank_grid_value = rank_grid_value,
      exact_alpha_rank_grid = exact_alpha_rank_grid,
      rejection_rule = "plus-one p <= alpha",
      master_seed = seed,
      replicate_seeds = seeds[seq_len(attempts_evaluated)],
      seed_stream = "replicate_seed_stream_v1",
      execution_mode = "fixed_drop_usable_v1",
      progress_requested = isTRUE(progress),
      simulation_source =
        "plug-in simulation from fitted null parameters",
      likelihood_mode = observed$pair$likelihood_mode,
      optimizer = observed$pair$optimizer,
      refit_control_source = resolved_control$source,
      refit_control = frozen_refit_control,
      refit_optimizer_used = resolved_control$optimizer_used,
      refit_optimizer_control = resolved_control$optimizer_control,
      full_call_control = resolved_control$full_call_control,
      null_call_control = resolved_control$null_call_control,
      validity_rules = list(
        primary = paste0(
          "draw to exactly usable_B positive/zero LRs; ordinary failures ",
          "and all negative LRs trigger top-up"
        ),
        small_negative = "p_zero sensitivity sets to zero",
        material_negative = "p_all_zero sensitivity sets to zero",
        failed_or_nonfinite = "exclude, count by reason, and top up",
        singular_fit = "retain if otherwise converged; flag separately"
      ),
      software = list(
        R = paste(R.version$major, R.version$minor, sep = "."),
        lme4 = as.character(utils::packageVersion("lme4"))
      )
    ),
    class = "slopeaudit_q1_internal"
  )
}
