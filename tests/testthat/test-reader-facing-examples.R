test_that("the installed Beat the Blues fixture is coherent", {
  path <- system.file(
    "extdata",
    "btheb_primary.csv",
    package = "slopeaudit",
    mustWork = TRUE
  )
  dat <- utils::read.csv(path, stringsAsFactors = FALSE)

  expect_identical(
    unname(tools::md5sum(path)),
    "da0c40cc87cdd4be8956c1f042614cde"
  )
  expect_identical(
    names(dat),
    c("subject", "bdi", "month", "month_c", "occasion", "treatment")
  )
  expect_equal(nrow(dat), 380L)
  expect_equal(length(unique(dat$subject)), 100L)
  expect_setequal(unique(dat$month), c(0, 2, 3, 5, 8))
  expect_equal(dat$month_c, dat$month - 4)
  expect_equal(sum(table(dat$subject) == 1L), 3L)
  expect_setequal(unique(dat$treatment), c("TAU", "BtheB"))
})


test_that("every installed copy-paste example parses", {
  example_dir <- system.file("examples", package = "slopeaudit", mustWork = TRUE)
  files <- list.files(example_dir, pattern = "[.]R$", full.names = TRUE)

  expect_setequal(
    basename(files),
    c("sleepstudy-audit.R", "btheb-audit.R", "design-sensitivity.R")
  )
  for (file in files) {
    expect_error(parse(file = file), NA)
  }
})


test_that("worked examples expose the strict target contract", {
  example_dir <- system.file("examples", package = "slopeaudit", mustWork = TRUE)
  files <- file.path(example_dir, c("sleepstudy-audit.R", "btheb-audit.R"))
  source_text <- paste(unlist(lapply(files, readLines, warn = FALSE)), collapse = "\n")
  required <- c(
    "basis_type", "outcome_units", "time_or_exposure_scale",
    "origin_or_reference", "reference_window", "estimand"
  )

  for (field in required) {
    expect_match(source_text, field, fixed = TRUE)
  }
})
