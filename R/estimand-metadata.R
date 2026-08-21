# Target-estimand identification contract.
#
# The random-effect coefficient name identifies a matrix coordinate but does
# not, by itself, tell a reader what one unit of that coefficient means.
# User-facing decomposition therefore requires explicit estimand metadata.

.sa_nonempty_scalar_character <- function(x) {
  is.character(x) &&
    length(x) == 1L &&
    !is.na(x) &&
    nzchar(trimws(x))
}

.sa_validate_target_metadata <- function(
    target,
    target_metadata,
    require_identified = TRUE) {

  required <- c(
    "basis_type",
    "outcome_units",
    "time_or_exposure_scale",
    "origin_or_reference",
    "reference_window",
    "estimand"
  )

  if (is.null(target_metadata)) {
    if (require_identified) {
      .sa_stop_estimand_identification(
        paste0(
          "The target random-effect column was identified as `", target,
          "`, but its estimand is not identified. Supply `target_metadata` ",
          "with basis_type, outcome_units, time_or_exposure_scale, ",
          "origin_or_reference, reference_window, and estimand."
        ),
        missing_fields = required
      )
    }

    return(list(
      identified = FALSE,
      target_design_column = target,
      missing_fields = required
    ))
  }

  if (!is.list(target_metadata) || is.null(names(target_metadata))) {
    .sa_stop_estimand_identification(
      "`target_metadata` must be a named list.",
      missing_fields = required
    )
  }

  missing <- required[
    !vapply(required, function(nm) {
      nm %in% names(target_metadata) &&
        .sa_nonempty_scalar_character(target_metadata[[nm]])
    }, logical(1))
  ]

  if (length(missing) > 0L) {
    .sa_stop_estimand_identification(
      sprintf(
        "Target estimand metadata is incomplete. Missing/empty fields: %s.",
        paste(missing, collapse = ", ")
      ),
      missing_fields = missing
    )
  }

  allowed_basis <- c(
    "linear",
    "affine_linear",
    "nonlinear",
    "piecewise",
    "interaction",
    "other"
  )

  basis_type <- target_metadata$basis_type
  if (!basis_type %in% allowed_basis) {
    .sa_stop_estimand_identification(
      sprintf(
        "`basis_type` must be one of: %s.",
        paste(allowed_basis, collapse = ", ")
      )
    )
  }

  transformed <- basis_type %in% c(
    "nonlinear", "piecewise", "interaction", "other"
  )

  basis_expression <- target_metadata$basis_expression
  if (transformed &&
      !.sa_nonempty_scalar_character(basis_expression)) {
    .sa_stop_estimand_identification(
      paste0(
        "Transformed/piecewise/interacted targets require a non-empty ",
        "`basis_expression` describing the audited design coordinate."
      ),
      missing_fields = "basis_expression"
    )
  }

  if (!transformed &&
      !.sa_nonempty_scalar_character(basis_expression)) {
    basis_expression <- target
  }

  normalized <- target_metadata
  normalized$identified <- TRUE
  normalized$target_design_column <- target
  normalized$basis_expression <- basis_expression
  normalized$affine_invariance_claim <-
    basis_type %in% c("linear", "affine_linear")
  normalized$missing_fields <- character()

  normalized
}

.sa_require_identified_estimand <- function(estimand) {
  if (is.null(estimand) || !isTRUE(estimand$identified)) {
    missing <- if (is.null(estimand$missing_fields)) character() else
      estimand$missing_fields

    .sa_stop_estimand_identification(
      "The target slope estimand is not sufficiently identified for a user-facing decomposition.",
      missing_fields = missing
    )
  }

  invisible(TRUE)
}
