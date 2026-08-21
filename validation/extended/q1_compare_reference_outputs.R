#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(jsonlite)
})

ANCHORS <- Sys.getenv(
  "SLOPEAUDIT_Q1_ANCHORS",
  "validation/reference/q1_manuscript_reference/anchors.json"
)
SLEEP <- Sys.getenv(
  "SLOPEAUDIT_Q1_SLEEP_SUMMARY",
  "validation/extended/results/q1_sleepstudy/q1_sleepstudy_summary.csv"
)
BTHEB <- Sys.getenv(
  "SLOPEAUDIT_Q1_BTHEB_SUMMARY",
  "validation/extended/results/q1_btheb/q1_btheb_summary.csv"
)
OUT <- Sys.getenv(
  "SLOPEAUDIT_Q1_COMPARE_OUT",
  "validation/extended/results/q1_reference_comparison.csv"
)

anchors <- jsonlite::fromJSON(ANCHORS, simplifyVector = TRUE)
sleep <- read.csv(SLEEP, stringsAsFactors = FALSE)
btheb <- read.csv(BTHEB, stringsAsFactors = FALSE)

rows <- rbind(
  data.frame(
    case = "sleepstudy_primary",
    quantity = "observed_LR",
    R_value = sleep$observed_LR[[1]],
    historical_Python_value =
      anchors$sleepstudy_primary$observed_LR_approx,
    direct_numeric_comparison = TRUE,
    note = "Same fitted-model target; optimizer last digits may vary.",
    stringsAsFactors = FALSE
  ),
  data.frame(
    case = "btheb_primary_homogeneous",
    quantity = "observed_LR",
    R_value = btheb$observed_LR[[1]],
    historical_Python_value =
      anchors$btheb_primary_homogeneous$observed_LR_approx,
    direct_numeric_comparison = TRUE,
    note = "Same primary estimand/model family; adjudicate material disagreement.",
    stringsAsFactors = FALSE
  ),
  data.frame(
    case = "sleepstudy_primary",
    quantity = "bootstrap_p",
    R_value = sleep$p_drop[[1]],
    historical_Python_value =
      anchors$sleepstudy_primary$historical_p_plus_one,
    direct_numeric_comparison = FALSE,
    note = "Historical and current bootstrap seed/optimizer implementations differ; contextual only until refrozen.",
    stringsAsFactors = FALSE
  ),
  data.frame(
    case = "btheb_primary_homogeneous",
    quantity = "bootstrap_p",
    R_value = btheb$p_drop[[1]],
    historical_Python_value = 0.077,
    direct_numeric_comparison = FALSE,
    note = "Historical and current bootstrap implementations differ; contextual only until refrozen.",
    stringsAsFactors = FALSE
  )
)

rows$absolute_difference <- abs(
  rows$R_value - rows$historical_Python_value
)

dir.create(dirname(OUT), recursive = TRUE, showWarnings = FALSE)
write.csv(rows, OUT, row.names = FALSE)
print(rows)
