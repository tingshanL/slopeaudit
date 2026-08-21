reference_root <- function() {
  env <- Sys.getenv("SLOPEAUDIT_REFERENCE_ROOT", unset = "")
  if (nzchar(env) && dir.exists(env)) {
    return(normalizePath(env, mustWork = TRUE))
  }

  candidates <- c(
    testthat::test_path("fixtures", "reference"),
    file.path(getwd(), "validation", "reference"),
    testthat::test_path("..", "..", "validation", "reference")
  )

  for (candidate in candidates) {
    if (dir.exists(candidate)) {
      return(normalizePath(candidate, mustWork = TRUE))
    }
  }

  stop(
    paste0(
      "Could not find validation/reference. Set SLOPEAUDIT_REFERENCE_ROOT ",
      "to the repository's validation/reference directory."
    ),
    call. = FALSE
  )
}

fixture_dir <- function(id) {
  aliases <- list(
    irregular_hand_oracle =
      c("irregular_hand_oracle", "irregular_oracle"),
    zero_direct_single_observation =
      c("zero_direct_single_observation", "single_observation_zero_direct"),
    zero_direct_no_time_variation =
      c("zero_direct_no_time_variation", "no_time_variation_zero_direct")
  )

  candidates <- aliases[[id]]
  if (is.null(candidates)) {
    candidates <- id
  }

  root <- reference_root()
  for (candidate in candidates) {
    path <- file.path(root, candidate)
    if (dir.exists(path)) {
      return(path)
    }
  }

  stop(sprintf("Fixture `%s` was not found.", id), call. = FALSE)
}

read_fixture_json <- function(id, filename) {
  jsonlite::fromJSON(
    file.path(fixture_dir(id), filename),
    simplifyVector = FALSE
  )
}

json_matrix <- function(x) {
  do.call(rbind, lapply(x, function(row) as.numeric(unlist(row))))
}

json_vector <- function(x) {
  as.numeric(unlist(x))
}

expect_num_equal <- function(object, expected, tolerance = 1e-10) {
  testthat::expect_equal(
    as.numeric(object),
    as.numeric(expected),
    tolerance = tolerance
  )
}

build_random_intercept_slope_design <- function(time) {
  cbind(Intercept = 1, target_slope = as.numeric(time))
}
