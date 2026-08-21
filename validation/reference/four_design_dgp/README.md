# Four-design DGP schedule fixture

The calibration uses the exact four true-zero cells from
`AABC_parametric_RLRT_anchored_profile_adequacy_operating_simulation.ipynb`.
The balanced schedule is generated directly in R. The irregular schedule must
be frozen once from the original AABC source files because the notebook stores
the reconstruction code, not the 1,805 schedule rows.

The v11 audit bundle includes the already-frozen deidentified schedule and its
provenance record from the audited v10 pilot. The extraction commands below are
for provenance-preserving regeneration, not a prerequisite when those two
files are already present and their hashes match.

From the package root, set:

```bash
export AABC_Q1_SUBJECTS_CSV="/absolute/path/to/AABC2_subjects.csv"
export AABC_Q1_ASL_AUDIT="/absolute/path/to/AABC_ASL_metabolic_perfusion_audit.xlsx"

python validation/provenance/q1_freeze_irregular_schedule_fixture.py
```

This creates:

- `irregular_fluid_cognition_schedule.csv`
- `irregular_fluid_cognition_schedule.provenance.json`

The fixture contains only deidentified schedule IDs, visit order, and
`t5_mid`. It must contain 691 schedules and 1,805 rows. Do not substitute an
approximate schedule if these assertions fail.
