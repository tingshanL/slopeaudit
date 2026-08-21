# Direct within-participant information for a target random slope.
#
# The nuisance participant-specific effects are treated as unknown fixed
# constants. No information is borrowed through G in this calculation.

.sa_direct_slope_information <- function(
    H,
    s,
    R,
    tau_s2,
    rank_tol = .sa_rank_tol,
    identity_tol = .sa_identity_tol) {

  s <- as.numeric(s)
  n <- length(s)

  if (n == 0L || any(!is.finite(s))) {
    stop("`s` must be a non-empty finite numeric vector.", call. = FALSE)
  }

  if (is.null(H)) {
    H <- matrix(numeric(0), nrow = n, ncol = 0L)
  } else {
    H <- as.matrix(H)
  }

  .sa_assert_numeric_matrix(H, "H", nrow_ = n)
  .sa_assert_numeric_matrix(R, "R", nrow_ = n, ncol_ = n)
  .sa_assert_spd(R, "R")

  if (!is.numeric(tau_s2) || length(tau_s2) != 1L ||
      is.na(tau_s2) || !is.finite(tau_s2) || tau_s2 <= 0) {
    stop("`tau_s2` must be one strictly positive finite number.", call. = FALSE)
  }

  U <- chol(R)
  s_w <- as.numeric(forwardsolve(t(U), s))

  if (ncol(H) == 0L) {
    H_w <- matrix(numeric(0), nrow = n, ncol = 0L)
  } else {
    H_w <- forwardsolve(t(U), H)
  }

  rank_H <- .sa_matrix_rank(H_w, tol = rank_tol)
  rank_Hs <- .sa_matrix_rank(cbind(H_w, s_w), tol = rank_tol)

  if (rank_Hs == rank_H) {
    return(list(
      has_direct_slope_information = FALSE,
      contrast = rep(NA_real_, n),
      direct_information_J = 0,
      direct_error_variance = Inf,
      R_direct = 0,
      constraint_error_nuisance = 0,
      constraint_error_target = 0,
      constraint_tolerance_nuisance = 0,
      constraint_tolerance_target = 0,
      constraint_nuisance_failed = FALSE,
      constraint_target_failed = FALSE
    ))
  }

  if (rank_H == 0L) {
    residual_w <- s_w
  } else {
    Q <- .sa_qr_column_basis(H_w, tol = rank_tol)
    residual_w <- s_w - as.numeric(Q %*% crossprod(Q, s_w))
  }

  J <- sum(residual_w * residual_w)

  if (!is.finite(J) || J <= 0) {
    stop(
      paste0(
        "The target slope increased design rank, but positive direct ",
        "information could not be represented numerically."
      ),
      call. = FALSE
    )
  }

  q <- as.numeric(backsolve(U, residual_w))
  cvec <- q / J

  v_direct <- 1 / J
  signal_to_error <- tau_s2 * J
  R_direct <- signal_to_error / (1 + signal_to_error)

  nuisance_error <- if (ncol(H) == 0L) {
    0
  } else {
    max(abs(as.numeric(crossprod(H, cvec))))
  }
  target_error <- abs(as.numeric(crossprod(s, cvec)) - 1)

  c_norm <- sqrt(sum(cvec^2))
  nuisance_scale <- if (ncol(H) == 0L) {
    1
  } else {
    max(1, max(sqrt(colSums(H^2)) * c_norm))
  }
  target_scale <- max(1, sqrt(sum(s^2)) * c_norm)

  nuisance_tol <- identity_tol * nuisance_scale
  target_tol <- identity_tol * target_scale

  list(
    has_direct_slope_information = TRUE,
    contrast = cvec,
    direct_information_J = J,
    direct_error_variance = v_direct,
    R_direct = R_direct,
    constraint_error_nuisance = nuisance_error,
    constraint_error_target = target_error,
    constraint_tolerance_nuisance = nuisance_tol,
    constraint_tolerance_target = target_tol,
    constraint_nuisance_failed = nuisance_error > nuisance_tol,
    constraint_target_failed = target_error > target_tol
  )
}
