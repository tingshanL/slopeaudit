# Internal numerical constants for the authoritative matrix backend.
#
# These are identity/equality tolerances only. They are NOT universal
# thresholds for declaring a fitted target-slope variance to be zero.

.sa_identity_tol <- sqrt(.Machine$double.eps)
.sa_rank_tol <- 1e-10
