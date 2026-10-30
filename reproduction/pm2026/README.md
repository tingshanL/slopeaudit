# Reproduction materials for the 2026 manuscript

Code for "What Do Person-Specific Slopes Measure? Direct Information, Model Assistance, and
Downstream Inference." Folders follow the online supplement. Run every command from the
repository root after restoring the environments (`environment/README.md`; Python 3.9.20 with
`environment/requirements.txt`, R with `renv.lock`).

Small result files are in `results/pm2026/`; full run outputs are in a new version of the
Zenodo output dataset (10.5281/zenodo.22092591). The code of each tagged release is archived
separately as a Zenodo software record. The v0.1.0 reproduction folders elsewhere in
`reproduction/` are unchanged; see "Relation to v0.1.0" below.

## Results, code and inputs (Table S8.2)

| Result | Code | Inputs |
|---|---|---|
| Figure 1 | `figures/fig1_values.py`, `figures/make_figure_01.py` | none (known parameters) |
| S2 identities | `s2_identities/s2_numerical_checks.py` | none (random configurations) |
| Downstream simulation: Table 1, Figure 2, Tables S3.1–S3.7 | `s3_downstream/downstream_simulation.py` (with `reml_tools.py`); `make_figure_02.py`, `make_S3_tables.py` | none (synthetic) |
| Operating-study audit: Table 2, Figure 3, Tables S4.1–S4.6, S5.3 | `s4_operating_audit/s4_step1`–`s4_step7`, `make_S4_tables.py`, `make_figure_03.py`; generator and summarizer in `reproduction/s4_operating_study/` | original S4 outputs (Zenodo v0.1.0); restricted schedules for empirical cells |
| Attenuation study and its audit: S3.4, S5.4 | generator in `reproduction/s5_downstream_simulation/`; audit in `provenance/attenuation_audit/`; `s5_attenuation_audit/s5_partB_numbers.py` | original attenuation-study outputs (Zenodo v0.1.0); restricted source anchors |
| sleepstudy κ: S6.2 | `s6_empirical/sleepstudy_kappa.R` | public (`lme4`) |
| AABC2 κ: Figure 4, Table S6.2 | `s6_empirical/aabc2_kappa_aggregate.py`, `make_figure_04.py` | restricted participant-level file (data holder only) |
| Design sensitivity: Figure S7.1, Table S7.1 | `s7_design/s7_core.py`, `make_S7.py` | `results/pm2026/empirical/s7_aabc2_aggregate/` |
| AABC2 tests and decompositions, Beat the Blues: S6 | `reproduction/s7_s8_aabc2_restricted/`, `reproduction/s6_public_audits/` (v0.1.0) | as documented there |

## Checking the reported values without refitting

```bash
python3 reproduction/pm2026/release/check_reported_values.py
```

This regenerates Table 1, Figures 2–4, Tables S3.1–S3.7, S4.1–S4.6 and S7.1 and the S5.4 audit
numbers from `results/pm2026/` and compares them with `results/pm2026/checks/expected/`, which
were checked against the manuscript. The S2 and S7 identity checks run without inputs:

```bash
python3 reproduction/pm2026/s2_identities/s2_numerical_checks.py s2_numerical_checks.json
python3 reproduction/pm2026/s7_design/make_S7.py --aabc2 results/pm2026/empirical/s7_aabc2_aggregate --out-dir s7_out
```

## Downstream simulation (Supplement S3)

```bash
S=reproduction/pm2026/s3_downstream/downstream_simulation.py
python3 $S --stage fixtures
python3 $S --stage common    --target-d2 0.8 --reps 200 --n 200 --seed 20261015 --workers 8 --output-dir runs/pilot_v2
python3 $S --stage common    --target-d2 0.5 --reps 200 --n 200 --seed 20261015 --workers 8 --output-dir runs/pilot_v2
python3 $S --stage irregular --target-d2 0.8 --reps 200 --n 200 --seed 20261015 --workers 8 --output-dir runs/contam_irregular_full
python3 $S --stage irregular --target-d2 0.5 --reps 200 --n 200 --seed 20261015 --workers 8 --output-dir runs/contam_irregular_full
python3 $S --stage common --target-d2 0.5 --constructions A --c-values 0 0.25 0.5 0.75 \
    --reps 500 --n 200 --seed 30261018 --workers 8 --output-dir runs/pilot_v2_A_leakage
python3 $S --stage common --target-d2 0.5 --constructions A --c-values 0.5 \
    --reps 500 --n 600 --seed 40261018 --workers 8 --output-dir runs/pilot_v2_A_leakage_N600
```

Each run writes `<stage>_D2_<target>_likelihood_selection/` with a manifest, replicate and
candidate-solution files and summaries. The settings above are those recorded in the run
manifests. Tables and Figure 2:

```bash
R=runs; D=reproduction/pm2026/s3_downstream
ARGS="--run common_80=$R/pilot_v2/common_D2_0.80_likelihood_selection \
      --run common_50=$R/pilot_v2/common_D2_0.50_likelihood_selection \
      --run irregular_80=$R/contam_irregular_full/irregular_D2_0.80_likelihood_selection \
      --run irregular_50=$R/contam_irregular_full/irregular_D2_0.50_likelihood_selection \
      --dose $R/pilot_v2_A_leakage/common_D2_0.50_likelihood_selection \
      --n600 $R/pilot_v2_A_leakage_N600/common_D2_0.50_likelihood_selection"
python3 $D/make_S3_tables.py --out-dir s3_tables $ARGS
python3 $D/make_figure_02.py --out-dir figure_02 $ARGS
```

## Operating-study audit (Supplements S4 and S5.3)

`S4=<original S4 output dir>` is the folder of the original operating-study run
(`s4_completion_validation_v1` in the v0.1.0 Zenodo archive). Empirical-schedule cells also
need the restricted schedule JSON produced by `s4_operating_audit/restricted/`; balanced and
true-zero balanced cells do not.

```bash
A=reproduction/pm2026/s4_operating_audit; T=reproduction/pm2026/s3_downstream/reml_tools.py
python3 $A/s4_step1_integrity.py $S4 --out runs/s4_step1/s4_step1_report.csv
python3 $A/s4_step2_stratify.py  $S4 --out-dir runs/s4_step2
python3 $A/s4_step3_fit_audit.py --repo-root . --s4-dir $S4 --reml-tools $T \
    --per-class 10 --sample-seed 1 --out runs/s4_step3_full/s4_step3_audit.csv
# first screen: refit the 245 fits that stayed at their starting values
python3 $A/s4_step4_startpoint_refit.py --repo-root . --reml-tools $T --s4-dir $S4 \
    --schedule-json <schedules v1 JSON> --workers 2 --out-dir runs/s4_step4_full_v2
# second screen, by schedule group (balanced; ATT/CBF/cognition; structural)
for spec in "balanced_core critical_near_boundary true_zero|balanced_three_wave|<schedules v1 JSON>|balanced_full_quiet2" \
            "empirical_AABC true_zero|empirical__fluid_cognition empirical__global_cortical_att empirical__global_cortical_cbf|<schedules v1 JSON>|empirical_full" \
            "empirical_AABC|empirical__global_mean_thickness empirical__hippocampus empirical__lateral_ventricles|<schedules all_v2 JSON>|structural_full"; do
  IFS='|' read fam sch js name <<< "$spec"
  python3 $A/s4_step5_null_screen.py --repo-root . --reml-tools $T --step4-script $A/s4_step4_startpoint_refit.py \
      --s4-dir $S4 --replicates-csv runs/s4_step4_full_v2/operating_study_replicate_results_startpoint_patched.csv \
      --step4-audit runs/s4_step4_full_v2/startpoint_refit_audit.csv --schedule-json "$js" \
      --families $fam --schedules $sch --workers 2 --out-dir runs/s4_step5_$name
done
# union of the flagged keys (6,661), then the consolidated refit of 245 + 6,661 data sets
python3 - <<'EOF'
import pandas as pd
k = pd.concat([pd.read_csv(f"runs/s4_step5_{n}/step5_flagged_keys.csv")
               for n in ("balanced_full_quiet2", "empirical_full", "structural_full")]).drop_duplicates()
k.to_csv("runs/s4_step5_all_flagged_keys.csv", index=False); print(len(k))
EOF
python3 $A/s4_step4_startpoint_refit.py --repo-root . --reml-tools $T --s4-dir $S4 \
    --schedule-json <schedules all_v2 JSON> --extra-keys runs/s4_step5_all_flagged_keys.csv \
    --workers 2 --out-dir runs/s4_step6_consolidated_refit
python3 reproduction/s4_operating_study/summarize_operating_study.py \
    --replicates-csv runs/s4_step6_consolidated_refit/operating_study_replicate_results_startpoint_patched.csv \
    --manifest-csv $S4/operating_study_design.csv --output-dir runs/s4_step6_consolidated_refit/summary --mode full
python3 reproduction/s4_operating_study/summarize_operating_study.py \
    --replicates-csv $S4/operating_study_replicate_results.csv \
    --manifest-csv $S4/operating_study_design.csv --output-dir runs/s4_original_summary --mode full
python3 $A/s4_step7_reporting.py --orig-summary runs/s4_original_summary \
    --new-summary runs/s4_step6_consolidated_refit/summary --out-dir runs/s4_step7_reporting
python3 $A/make_S4_tables.py --summary runs/s4_step6_consolidated_refit/summary --step7 runs/s4_step7_reporting \
    --design $S4/operating_study_design.csv --out-dir s4_tables
python3 $A/make_figure_03.py --operating-csv runs/s4_step6_consolidated_refit/summary/operating_study_replicate_results.csv \
    --out-dir figure_03
```

The step manifests record the hashes of every input and script; the executed script versions
are in `provenance/downstream_and_s4_audit/`.

**Restricted workflow.** `s4_operating_audit/restricted/` exports the empirical schedules
from authorized AABC2 inputs and the extracts of the original analysis notebook. The extracts
contain participant identifiers and are not distributed; the scripts check their hashes, so
these steps run only on an authorized user's machine, and their output (schedule JSON) stays
there.

## Attenuation-study audit (Supplement S5.4)

Rerunning the audit and summarizing it are separate steps.

**Rerun the audit.** Use the executed snapshot, which loads the earlier likelihood helper it was
run with (see `provenance/README.md`). It needs the original attenuation-study outputs
(`s5_downstream_simulation_frozen_outputs.zip` in the v0.1.0 Zenodo archive) and, for Part B,
the restricted source-anchor manifest.

```bash
python3 reproduction/pm2026/provenance/attenuation_audit/reproduction/s5_downstream_simulation/audit_s5_fit_selection.py \
    --s5-output <original attenuation-study output dir> --anchor-manifest <restricted anchor manifest> \
    --part AB --per-stratum 10 --include-flagged 40 --sample-seed 1 --workers 4 --output-dir runs/s5_fit_audit_AB
```

**Summarize the reported audit.** No refitting:

```bash
python3 reproduction/pm2026/s5_attenuation_audit/s5_partB_numbers.py results/pm2026/attenuation_audit/partB_by_task.csv
```

## Empirical analyses and design sensitivity (Supplements S6 and S7)

```bash
Rscript reproduction/pm2026/s6_empirical/sleepstudy_kappa.R s6_sleepstudy
# data holder only:
python3 reproduction/pm2026/s6_empirical/aabc2_kappa_aggregate.py --input <restricted reliability CSV> \
    --outcomes lateral_ventricles fluid_cognition --out-dir s6_aabc2
python3 reproduction/pm2026/s6_empirical/make_figure_04.py --sleep s6_sleepstudy \
    --aabc2 results/pm2026/empirical/s7_aabc2_aggregate --out-dir figure_04
python3 reproduction/pm2026/s7_design/make_S7.py --aabc2 results/pm2026/empirical/s7_aabc2_aggregate --out-dir s7_out
```

The sleepstudy values depend on the `lme4` optimizer only at about the 10th significant digit.
Output file names keep the `s5_` and `s7_` prefixes of the original runs.

## Release tools

`release/prepare_release_files.py` builds `results/pm2026/` and the Zenodo output archives
from the local project folder using an explicit file list, rewrites absolute paths in
manifests (both hashes are listed in `path_rewrites.csv`), and stops on restricted files,
participant identifiers or remaining home-directory paths. It locates and screens every input
before writing, requires a new or empty output folder, and moves the output into place only
when it is complete.

```bash
# complete results/pm2026 tree (summaries, compact replicate table, checks/, FILES.csv, path_rewrites.csv)
python3 reproduction/pm2026/release/prepare_release_files.py --root <project folder> \
    --github-out <new folder> --checks results/pm2026/checks
diff -r <new folder> results/pm2026        # no output: the local runs match the committed files

# Zenodo output archives (four zips, archive_entry_manifest.csv, path_rewrites.csv, SHA256SUMS.txt)
python3 reproduction/pm2026/release/prepare_release_files.py --root <project folder> \
    --zenodo-out <new folder>
```

`checks/` holds the regenerated S2 and S7 check outputs and the expected values that
`release/check_reported_values.py` compares against; they are copied as given, so the
`diff -r` comparison tests the files derived from the project folder. The expected files are
listed in `release/expected_outputs.py`; `check_reported_values.py` stops if any is missing
and requires all 17 groups to be evaluated.

## Relation to v0.1.0

- **Operating study.** The v0.1.0 summaries in `results/frozen_summaries/s4/` and the v0.1.0
  Zenodo outputs predate the likelihood audit. They are superseded for reporting by the summaries
  after the consolidated refit in `results/pm2026/operating_audit/`.
- **Attenuation study.** The reported results are retained. In the audited sample
  (Supplement S5.4), refitting from the best fits changed downstream coefficients negligibly
  (at most 7 × 10⁻⁶) and changed none of the significance decisions in the reported regimes;
  coverage was not reassessed, because the audit did not record standard errors or the
  generating coefficient. Its auxiliary likelihood-ratio diagnostics failed nesting checks and
  are retired; heterogeneity testing is evaluated by the true-zero simulation instead.
- **Package.** v0.2.0 adds the nuisance share κ (`NEWS.md`); the other numerical routines are
  unchanged.
