"""Expected-output files under results/pm2026/checks/expected, used by
check_reported_values.py (which compares against them) and prepare_release_files.py
(which requires them when building results/pm2026)."""

S3_TABLES = ["S3_boundary_rows.tex", "S3_c0_rows.tex", "S3_dose_rows.tex", "S3_fit_rows.tex",
             "S3_known_rows.tex", "S3_paired_rows.tex", "S3_settings_rows.tex"]
S4_TABLES = ["S4_calibration_rows.tex", "S4_cells_rows.tex", "S4_changes_rows.tex",
             "S4_classes_rows.tex", "S4_repairs_rows.tex"]
OTHER = ["table1_rows.tex", "s5_attenuation_audit_numbers.json", "s7_schedules_rows.tex"]

# Paths relative to checks/expected.
EXPECTED_FILES = ([f"s3_tables/{f}" for f in S3_TABLES] + [f"s4_tables/{f}" for f in S4_TABLES] + OTHER)

# Regenerated check outputs kept beside the expected files (relative to checks/).
CHECK_OUTPUTS = ["s2/s2_numerical_checks.json", "s7/S7_checks.json", "s7/S7_schedules_rows.tex",
                 "s7/figure_S7_1.pdf", "s7/figure_S7_1.png"]

# 7 S3 tables + Table 1 + 5 S4 tables + Figure 3 + S5.4 numbers + Table S7.1 + Figure 4.
N_REPORTED_GROUPS = len(S3_TABLES) + 1 + len(S4_TABLES) + 1 + 1 + 1 + 1
