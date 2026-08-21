# Maintainer-only first-release R environment freeze.
#
# These versions are the exact direct package versions observed in the
# validated first-release environment before the project-local renv library
# was initialized. Reviewers should use renv::restore(), not rerun this file.

options(
  repos = c(
    CRAN = "https://cloud.r-project.org"
  )
)

validated_versions <- c(
  lme4      = "2.0-6",
  testthat  = "3.3.2",
  jsonlite  = "2.0.0",
  knitr     = "1.51",
  rmarkdown = "2.31",
  devtools   = "2.5.2",
  roxygen2   = "8.1.0"
)

required_renv_version <- "1.2.4"

# Bootstrap renv only if necessary.
if (!requireNamespace("renv", quietly = TRUE)) {
  install.packages("renv")
}

actual_renv_version <- utils::packageDescription(
  "renv",
  fields = "Version"
)

if (!identical(
  unname(actual_renv_version),
  required_renv_version
)) {
  stop(
    "Expected renv ",
    required_renv_version,
    " for the first-release freeze, but found ",
    actual_renv_version,
    "."
  )
}

# Initialize only if the first attempt has not already created the project
# infrastructure. This makes the script safe to rerun after a partial install.
if (!file.exists("renv/activate.R")) {
  renv::init(
    project = ".",
    bare = TRUE,
    restart = FALSE,
    repos = getOption("repos")
  )
}

exact_specs <- paste0(
  names(validated_versions),
  "@",
  unname(validated_versions)
)

cat("Installing/verifying first-release direct R packages:\n")
for (spec in exact_specs) {
  cat("  ", spec, "\n", sep = "")
}

renv::install(
  exact_specs,
  project = ".",
  repos = getOption("repos"),
  prompt = FALSE
)

# Verify the RAW DESCRIPTION versions, rather than packageVersion(), because
# CRAN version strings can legitimately contain a hyphen (e.g. lme4 2.0-6).
installed_versions <- vapply(
  names(validated_versions),
  function(pkg) {
    unname(
      utils::packageDescription(
        pkg,
        fields = "Version"
      )
    )
  },
  character(1)
)

bad <- names(validated_versions)[
  installed_versions != validated_versions
]

if (length(bad)) {
  details <- paste(
    sprintf(
      "%s: expected %s, found %s",
      bad,
      validated_versions[bad],
      installed_versions[bad]
    ),
    collapse = "; "
  )

  stop(
    "Direct-package version verification failed: ",
    details
  )
}

# Supplying packages explicitly also records their recursive dependencies.
renv::snapshot(
  project = ".",
  packages = c(
    names(validated_versions),
    "renv"
  ),
  repos = getOption("repos"),
  prompt = FALSE,
  force = TRUE
)

cat("\nFrozen validated R versions:\n")
for (pkg in names(validated_versions)) {
  cat(
    "  ",
    pkg,
    "==",
    installed_versions[[pkg]],
    "\n",
    sep = ""
  )
}

cat(
  "  renv==",
  actual_renv_version,
  "\n",
  sep = ""
)

cat("\nrenv status:\n")
renv::status(project = ".")
