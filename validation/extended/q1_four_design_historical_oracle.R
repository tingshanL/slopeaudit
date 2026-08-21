#!/usr/bin/env Rscript

# This script audits the already-frozen historical four-design calibration
# summary. It does NOT claim that the historical calibrated restricted-LR
# procedure is identical to the new Contract-v1.1 per-analysis bootstrap.

path <- Sys.getenv(
  "SLOPEAUDIT_AABC_FIG2_Q1_SUMMARY",
  unset = ""
)

if (!nzchar(path)) {
  stop(
    paste0(
      "Set SLOPEAUDIT_AABC_FIG2_Q1_SUMMARY to the local historical ",
      "figure_02_panel_A_heterogeneity_false_positive_rates.csv. ",
      "Restricted/local source files are not bundled into the package."
    )
  )
}

x <- read.csv(path, stringsAsFactors = FALSE)

k_name <- if ("k" %in% names(x)) {
  "k"
} else if ("k_false_support" %in% names(x)) {
  "k_false_support"
} else {
  stop("Could not identify the false-support count column.")
}

n_name <- if ("n" %in% names(x)) {
  "n"
} else if ("n_evaluation" %in% names(x)) {
  "n_evaluation"
} else {
  stop("Could not identify the evaluation denominator column.")
}

got <- unname(as.matrix(x[, c(k_name, n_name)]))
expected <- matrix(
  c(
    47, 999,
    48, 997,
    41, 998,
    45, 999
  ),
  ncol = 2,
  byrow = TRUE
)

if (!identical(dim(got), dim(expected)) ||
    any(got != expected)) {
  stop(
    "Historical four-design k/n oracle does not match the frozen anchors."
  )
}

print(data.frame(
  k = got[, 1],
  n = got[, 2],
  rate = got[, 1] / got[, 2]
))

cat(
  paste0(
    "\nPASS: historical four-design oracle is intact.\n",
    "This is an external historical calibration audit, not a direct ",
    "v1.1 bootstrap-equivalence test.\n"
  )
)
