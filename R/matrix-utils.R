# Internal matrix validation and linear-algebra helpers.

.sa_is_symmetric <- function(x, tol = .sa_identity_tol) {
  is.matrix(x) &&
    nrow(x) == ncol(x) &&
    max(abs(x - t(x))) <= tol * max(1, max(abs(x)))
}

.sa_assert_numeric_matrix <- function(x, name, nrow_ = NULL, ncol_ = NULL) {
  if (!is.matrix(x) || !is.numeric(x)) {
    stop(sprintf("`%s` must be a numeric matrix.", name), call. = FALSE)
  }
  if (any(!is.finite(x))) {
    stop(sprintf("`%s` contains non-finite values.", name), call. = FALSE)
  }
  if (!is.null(nrow_) && nrow(x) != nrow_) {
    stop(sprintf("`%s` must have %d rows.", name, nrow_), call. = FALSE)
  }
  if (!is.null(ncol_) && ncol(x) != ncol_) {
    stop(sprintf("`%s` must have %d columns.", name, ncol_), call. = FALSE)
  }
  invisible(TRUE)
}

.sa_assert_spd <- function(x, name = "matrix") {
  .sa_assert_numeric_matrix(x, name)
  if (!.sa_is_symmetric(x)) {
    stop(sprintf("`%s` must be symmetric.", name), call. = FALSE)
  }
  out <- try(chol(x), silent = TRUE)
  if (inherits(out, "try-error")) {
    stop(sprintf("`%s` must be positive definite.", name), call. = FALSE)
  }
  invisible(TRUE)
}

.sa_assert_psd <- function(x, name = "matrix", tol = .sa_identity_tol) {
  .sa_assert_numeric_matrix(x, name)
  if (!.sa_is_symmetric(x, tol = tol)) {
    stop(sprintf("`%s` must be symmetric.", name), call. = FALSE)
  }

  ev <- eigen((x + t(x)) / 2, symmetric = TRUE, only.values = TRUE)$values
  scale <- max(1, max(abs(ev)))
  if (min(ev) < -tol * scale) {
    stop(sprintf("`%s` must be positive semidefinite.", name), call. = FALSE)
  }

  invisible(ev)
}

.sa_solve_spd <- function(a, b = NULL) {
  .sa_assert_spd(a, "a")
  u <- chol(a)

  if (is.null(b)) {
    return(chol2inv(u))
  }

  b <- as.matrix(b)
  if (!is.numeric(b) || any(!is.finite(b))) {
    stop("`b` must be finite and numeric.", call. = FALSE)
  }
  if (nrow(b) != nrow(a)) {
    stop("`b` has incompatible dimensions.", call. = FALSE)
  }

  y <- forwardsolve(t(u), b)
  backsolve(u, y)
}

.sa_scale_columns_for_rank <- function(x) {
  x <- as.matrix(x)

  if (ncol(x) == 0L) {
    return(x)
  }

  out <- x

  for (j in seq_len(ncol(x))) {
    scale_j <- max(abs(x[, j]))
    if (is.finite(scale_j) && scale_j > 0) {
      out[, j] <- x[, j] / scale_j
    }
  }

  out
}

.sa_matrix_rank <- function(x, tol = .sa_rank_tol) {
  x <- as.matrix(x)

  if (length(x) == 0L || ncol(x) == 0L) {
    return(0L)
  }

  x_scaled <- .sa_scale_columns_for_rank(x)
  qr(x_scaled, tol = tol, LAPACK = FALSE)$rank
}

.sa_qr_column_basis <- function(x, tol = .sa_rank_tol) {
  x <- as.matrix(x)

  if (ncol(x) == 0L) {
    return(matrix(numeric(0), nrow = nrow(x), ncol = 0L))
  }

  x_scaled <- .sa_scale_columns_for_rank(x)
  q <- qr(x_scaled, tol = tol, LAPACK = FALSE)
  r <- q$rank

  if (r == 0L) {
    return(matrix(numeric(0), nrow = nrow(x), ncol = 0L))
  }

  qr.Q(q, complete = FALSE)[, seq_len(r), drop = FALSE]
}

.sa_same_column_space <- function(a, b, tol = .sa_rank_tol) {
  a <- as.matrix(a)
  b <- as.matrix(b)

  if (nrow(a) != nrow(b)) {
    return(FALSE)
  }

  rank_a <- .sa_matrix_rank(a, tol = tol)
  rank_b <- .sa_matrix_rank(b, tol = tol)
  rank_ab <- .sa_matrix_rank(cbind(a, b), tol = tol)

  rank_a == rank_b && rank_ab == rank_a
}

.sa_validate_target_index <- function(target_index, q) {
  ok <- is.numeric(target_index) &&
    length(target_index) == 1L &&
    !is.na(target_index) &&
    is.finite(target_index) &&
    target_index == floor(target_index) &&
    target_index >= 1L &&
    target_index <= q

  if (!ok) {
    stop("`target_index` must identify one column of `Z`/`G`.", call. = FALSE)
  }

  as.integer(target_index)
}

.sa_symmetrize <- function(x) {
  (x + t(x)) / 2
}
