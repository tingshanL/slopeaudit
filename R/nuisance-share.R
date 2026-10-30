# Nuisance share kappa (manuscript Supplement S1.5 and Proposition 2).
#
# The fitted slope s_hat = w' r splits into a multiple of the target slope, a part
# driven by the nuisance random effects after removing their slope-predictable
# component (nu = w' Z_a u, u = a - g_as s / tau2), and residual error. kappa is the
# share of Var(s_hat) due to nu; sqrt(kappa) is the largest attainable absolute
# correlation between s_hat and a person-level variable uncorrelated with the
# target slope and the residuals.
#
# Var(s_hat) is computed as the sum of its three nonnegative components rather than
# as tau2 * D2, so kappa stays accurate when determinacy is small but positive.
#
# Values are participant-level and treat the supplied (or fitted) parameters as
# known. They are not bounds for a pooled regression over participants with
# different schedules.
#
# kappa_status:
#   "defined"                        kappa computed from the components
#   "no_nuisance_effects"            target is the only random effect; kappa = 0
#   "undefined_zero_fitted_variance" Cov(y, s) = Z G[, target] is exactly zero, so
#                                    Var(s_hat) = 0 and D2 = 0; kappa is undefined
#   "unresolved_fitted_variance"     Cov(y, s) is nonzero only at rounding level
#                                    relative to the scale of Z and G; kappa is not
#                                    numerically resolved and is reported as NA
#   "invalid_negative_share"         the nuisance component is materially negative
#                                    (e.g. G not positive semidefinite); the raw
#                                    kappa is kept, sqrt_kappa is NA, and the row is
#                                    counted as invalid in summaries

.sa_kappa_valid_status <- c("defined", "no_nuisance_effects")

.sa_nuisance_share <- function(
    Z,
    G,
    R,
    target_index,
    slope_weights,
    D_slope2,
    Delta_fit,
    identity_tol = .sa_identity_tol) {

  Z <- as.matrix(Z)
  G <- as.matrix(G)
  R <- as.matrix(R)
  tau_s2 <- G[target_index, target_index]
  nuisance_index <- setdiff(seq_len(ncol(G)), target_index)
  w <- as.numeric(slope_weights)

  not_computed <- function(status) {
    list(
      kappa = NA_real_,
      sqrt_kappa = NA_real_,
      kappa_residual_share = NA_real_,
      kappa_status = status,
      kappa_composition_error = NA_real_,
      kappa_ceiling_excess = NA_real_,
      kappa_composition_failed = FALSE,
      kappa_ceiling_violated = FALSE,
      kappa_materially_negative = FALSE
    )
  }

  # Cov(y, s) = Z G[, target]. Var(s_hat) = c' V^{-1} c is zero exactly when c = 0.
  cross_cov <- drop(Z %*% G[, target_index])
  if (all(cross_cov == 0)) {
    return(not_computed("undefined_zero_fitted_variance"))
  }
  rounding_scale <- 64 * .Machine$double.eps * ncol(Z) *
    max(abs(Z)) * max(abs(G[, target_index]))
  if (!all(is.finite(cross_cov)) || !all(is.finite(w)) ||
      max(abs(cross_cov)) <= rounding_scale) {
    return(not_computed("unresolved_fitted_variance"))
  }

  # Components of Var(s_hat): target signal, slope-unrelated nuisance, residual.
  signal_variance <- drop(crossprod(w, cross_cov))^2 / tau_s2
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

  fitted_variance <- signal_variance + nuisance_variance + residual_variance
  if (!is.finite(fitted_variance) || fitted_variance <= 0) {
    return(not_computed("unresolved_fitted_variance"))
  }

  kappa_raw <- nuisance_variance / fitted_variance
  residual_share <- residual_variance / fitted_variance

  # Identity: D2 + kappa + residual share = 1, with D2 from the determinacy
  # calculation. Ceilings: kappa <= 1 - D2 and kappa <= Delta_fit / D2
  # (Proposition 2(c)).
  composition_error <- D_slope2 + kappa_raw + residual_share - 1
  ceiling <- if (is.finite(D_slope2) && D_slope2 > 0) {
    min(1 - D_slope2, Delta_fit / D_slope2)
  } else {
    NA_real_
  }
  ceiling_excess <- kappa_raw - ceiling

  materially_negative <- kappa_raw < -identity_tol
  if (materially_negative) {
    status <- "invalid_negative_share"
  }
  # Rounding-level negatives (e.g. from a numerically singular S) are set to zero
  # before the square root; material negatives are kept and reported as invalid.
  kappa <- if (kappa_raw < 0 && !materially_negative) 0 else kappa_raw

  list(
    kappa = kappa,
    sqrt_kappa = if (kappa >= 0) sqrt(kappa) else NA_real_,
    kappa_residual_share = residual_share,
    kappa_status = status,
    kappa_composition_error = composition_error,
    kappa_ceiling_excess = ceiling_excess,
    kappa_composition_failed = !is.finite(composition_error) ||
      abs(composition_error) > identity_tol,
    kappa_ceiling_violated = is.finite(ceiling_excess) &&
      ceiling_excess > identity_tol,
    kappa_materially_negative = materially_negative
  )
}


# Percentiles over participants whose kappa is valid: status "defined" or
# "no_nuisance_effects" and no failed composition or ceiling check. Rows that are
# undefined, unresolved or invalid stay in the participant table and are counted
# here; they never enter the percentiles.
.sa_kappa_summary <- function(participant_table) {
  status <- participant_table$kappa_status
  flag <- function(name) {
    v <- participant_table[[name]]
    if (is.null(v)) rep(FALSE, length(status)) else (!is.na(v) & v)
  }
  invalid <- status == "invalid_negative_share" |
    flag("kappa_composition_failed") | flag("kappa_ceiling_violated")
  valid <- status %in% .sa_kappa_valid_status & !invalid &
    is.finite(participant_table$kappa) & is.finite(participant_table$sqrt_kappa)

  k <- participant_table$kappa[valid]
  r <- participant_table$sqrt_kappa[valid]
  stat <- function(x, fun) if (length(x)) fun(x) else NA_real_
  q <- function(x, p) stat(x, function(v) unname(stats::quantile(v, p)))

  data.frame(
    quantity = c("kappa", "sqrt_kappa"),
    mean = c(stat(k, mean), stat(r, mean)),
    p05 = c(q(k, 0.05), q(r, 0.05)),
    p25 = c(q(k, 0.25), q(r, 0.25)),
    median = c(q(k, 0.50), q(r, 0.50)),
    p75 = c(q(k, 0.75), q(r, 0.75)),
    p95 = c(q(k, 0.95), q(r, 0.95)),
    maximum = c(stat(k, max), stat(r, max)),
    n_defined = sum(valid),
    n_undefined = sum(status == "undefined_zero_fitted_variance"),
    n_unresolved = sum(status == "unresolved_fitted_variance"),
    n_invalid = sum(invalid),
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}
