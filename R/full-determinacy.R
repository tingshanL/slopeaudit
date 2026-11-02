# Full fitted-slope determinacy under the plug-in Gaussian model.

.sa_full_slope_determinacy <- function(
    Z,
    G,
    R,
    target_index,
    identity_tol = .sa_identity_tol) {

  Z <- as.matrix(Z)
  G <- as.matrix(G)
  R <- as.matrix(R)

  .sa_assert_numeric_matrix(Z, "Z")
  .sa_assert_numeric_matrix(G, "G")
  .sa_assert_numeric_matrix(R, "R")

  n <- nrow(Z)
  q <- ncol(Z)

  if (n == 0L || q == 0L) {
    stop("`Z` must have at least one row and one column.", call. = FALSE)
  }
  if (nrow(G) != q || ncol(G) != q) {
    stop("`G` must conform to the columns of `Z`.", call. = FALSE)
  }
  if (nrow(R) != n || ncol(R) != n) {
    stop("`R` must conform to the rows of `Z`.", call. = FALSE)
  }

  .sa_assert_psd(G, "G", tol = identity_tol)
  .sa_assert_spd(R, "R")

  target_index <- .sa_validate_target_index(target_index, q)

  tau_s2 <- G[target_index, target_index]
  if (!is.finite(tau_s2)) {
    stop("The target random-effect variance is non-finite.", call. = FALSE)
  }
  if (tau_s2 == 0) {
    .sa_stop_target_boundary(
      target_slope_variance = 0,
      target_index = target_index
    )
  }
  if (tau_s2 < 0) {
    stop("The target random-effect variance is negative.", call. = FALSE)
  }

  V <- Z %*% G %*% t(Z) + R
  Vinv_ZG <- .sa_solve_spd(V, Z %*% G)

  C <- G - G %*% t(Z) %*% Vinv_ZG
  C <- .sa_symmetrize(C)

  posterior_slope_variance <- C[target_index, target_index]
  D_slope2 <- 1 - posterior_slope_variance / tau_s2

  list(
    conditional_random_effect_covariance = C,
    slope_weights = Vinv_ZG[, target_index],
    posterior_slope_variance = posterior_slope_variance,
    D_slope2 = D_slope2,
    target_slope_variance = tau_s2,
    determinacy_below_zero = D_slope2 < -identity_tol,
    determinacy_above_one = D_slope2 > 1 + identity_tol
  )
}
