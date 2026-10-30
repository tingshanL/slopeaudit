# Nuisance share kappa (manuscript Supplement S1.5 and Proposition 2).
#
# The fitted slope s_hat = w' r splits into a multiple of the target slope, a part
# driven by the nuisance random effects after removing their slope-predictable
# component (nu = w' Z_a u, u = a - g_as s / tau2), and residual error. kappa is the
# share of Var(s_hat) due to nu; sqrt(kappa) is the largest attainable absolute
# correlation between s_hat and a person-level variable uncorrelated with the
# target slope and the residuals.
#
# Values are participant-level and treat the supplied (or fitted) parameters as
# known. They are not bounds for a pooled regression over participants with
# different schedules.

.sa_nuisance_share <- function(
    Z,
    G,
    R,
    target_index,
    slope_weights,
    D_slope2,
    Delta_fit,
    identity_tol = .sa_identity_tol) {

  tau_s2 <- G[target_index, target_index]
  nuisance_index <- setdiff(seq_len(ncol(G)), target_index)
  w <- as.numeric(slope_weights)

  undefined <- list(
    kappa = NA_real_,
    sqrt_kappa = NA_real_,
    kappa_residual_share = NA_real_,
    kappa_status = "undefined_zero_determinacy",
    kappa_composition_error = NA_real_,
    kappa_ceiling_excess = NA_real_,
    kappa_composition_failed = FALSE,
    kappa_ceiling_violated = FALSE,
    kappa_materially_negative = FALSE
  )

  # kappa is a share of Var(s_hat) = tau2 * D2; it is undefined when that is zero.
  if (!is.finite(D_slope2) || D_slope2 <= identity_tol) {
    return(undefined)
  }

  fitted_variance <- tau_s2 * D_slope2
  residual_variance <- drop(crossprod(w, R %*% w))

  if (length(nuisance_index) == 0L) {
    nuisance_variance <- 0
    status <- "no_nuisance_effects"
  } else {
    g_as <- G[nuisance_index, target_index, drop = FALSE]
    S <- G[nuisance_index, nuisance_index, drop = FALSE] -
      g_as %*% t(g_as) / tau_s2
    S <- .sa_symmetrize(S)
    loading <- crossprod(Z[, nuisance_index, drop = FALSE], w)
    nuisance_variance <- drop(crossprod(loading, S %*% loading))
    status <- "defined"
  }

  kappa_raw <- nuisance_variance / fitted_variance
  residual_share <- residual_variance / fitted_variance

  # Identity: D2 + kappa + residual share = 1. Ceilings: kappa <= 1 - D2 and
  # kappa <= Delta_fit / D2 (Proposition 2(c)).
  composition_error <- D_slope2 + kappa_raw + residual_share - 1
  ceiling <- min(1 - D_slope2, Delta_fit / D_slope2)
  ceiling_excess <- kappa_raw - ceiling

  materially_negative <- kappa_raw < -identity_tol
  # Rounding-level negatives (e.g. from a numerically singular S) are set to zero
  # before the square root; material negatives are reported as invalid.
  kappa <- if (kappa_raw < 0 && !materially_negative) 0 else kappa_raw

  list(
    kappa = kappa,
    sqrt_kappa = if (kappa >= 0) sqrt(kappa) else NA_real_,
    kappa_residual_share = residual_share,
    kappa_status = status,
    kappa_composition_error = composition_error,
    kappa_ceiling_excess = ceiling_excess,
    kappa_composition_failed = abs(composition_error) > identity_tol,
    kappa_ceiling_violated = ceiling_excess > identity_tol,
    kappa_materially_negative = materially_negative
  )
}


.sa_kappa_summary <- function(participant_table) {
  defined <- !is.na(participant_table$kappa)
  k <- participant_table$kappa[defined]
  r <- participant_table$sqrt_kappa[defined]
  q <- function(x, p) if (length(x)) unname(stats::quantile(x, p)) else NA_real_

  data.frame(
    quantity = c("kappa", "sqrt_kappa"),
    mean = c(if (length(k)) mean(k) else NA_real_, if (length(r)) mean(r) else NA_real_),
    p05 = c(q(k, 0.05), q(r, 0.05)),
    p25 = c(q(k, 0.25), q(r, 0.25)),
    median = c(q(k, 0.50), q(r, 0.50)),
    p75 = c(q(k, 0.75), q(r, 0.75)),
    p95 = c(q(k, 0.95), q(r, 0.95)),
    maximum = c(if (length(k)) max(k) else NA_real_, if (length(r)) max(r) else NA_real_),
    n_defined = sum(defined),
    n_undefined = sum(!defined),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}
