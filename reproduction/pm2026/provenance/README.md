# Provenance snapshots

The files in this folder are unchanged copies of the scripts that produced the results
reported in the 2026 manuscript. The manifests of operating-study audit steps 3–6 in
`results/pm2026/` record the hashes of the audit scripts and likelihood helper they ran, and
`RELEASE_HASHES.csv` links each snapshot that has a maintained copy to that copy. Use the maintained copies in the sibling
folders to rerun analyses, and these files to check what was run.

## `downstream_and_s4_audit/`

The final likelihood helper (`reml_tools.py`), the downstream simulation
(`contamination_pilot.py`) and the operating-study audit scripts, as executed.

## `attenuation_audit/`

The attenuation-study audit (Supplement S5.4) was run with an **earlier version** of the
likelihood helper, whose candidates are the three `statsmodels` optimizers and the
independent multistart maximizer, without the random-intercept solution embedded at zero
slope variance. The audit script loads its helper from `../s5b_contamination_pilot/`
relative to itself, so the original folder layout is kept: running
`attenuation_audit/reproduction/s5_downstream_simulation/audit_s5_fit_selection.py` uses the
earlier helper, as in the reported run. `simulation_core.py` and
`generate_downstream_simulation.py` here are identical to the released files in
`reproduction/s5_downstream_simulation/`.

## `RELEASE_HASHES.csv`

Links each maintained script to its original file and, where one exists, to the executed
snapshot here, and states what changed. The maintained copies differ from the originals only
in docstrings and comments, except for three recorded changes: the repository-path default in
`downstream_simulation.py`, an optional `--replicates` argument in `make_S4_tables.py` (so the
S4 tables can be regenerated from the compact public replicate table), and exact
(round-trip) number parsing in `s5_partB_numbers.py`. For the other copies, equivalence was
checked by comparing Python syntax trees with docstrings removed.

## Not distributed

- The extracts of the original AABC2 analysis notebook (`s4_source_code_1.txt`,
  `s4_source_code_2.txt`), which contain participant identifiers. The two schedule-export
  scripts in `../s4_operating_audit/restricted/` need them and therefore run only on an
  authorized user's machine.
- The restricted-derived schedule files (`s4_refit_schedules.json`) and the attenuation
  study's source-anchor manifest.
