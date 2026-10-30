# Provenance snapshots

The files in this folder are byte-for-byte copies of the scripts that produced the
results reported in the 2026 manuscript. They are kept unchanged so that their SHA-256
hashes match the hashes recorded in the run manifests. Use the maintained copies in the
sibling folders to rerun analyses; use these files to verify provenance.

## `downstream_and_s4_audit/`

The final likelihood helper (`reml_tools.py`), the downstream simulation
(`contamination_pilot.py`) and the operating-study audit scripts, as executed. The S4 audit
manifests record these hashes, for example `reml_tools.py`
`94f761cff1b7129793e63554097d020d95960f1b51eab1392bd792d30438f746`.

## `attenuation_audit/`

The attenuation-study audit (Supplement S5.4) was run with an **earlier version** of the
likelihood helper, whose candidates are the three `statsmodels` optimizers and the
independent multistart maximizer, without the random-intercept solution embedded at zero
slope variance. That audit script locates its helper at
`../s5b_contamination_pilot/reml_tools.py` relative to itself, so the original folder layout
is kept here: running
`attenuation_audit/reproduction/s5_downstream_simulation/audit_s5_fit_selection.py` loads the
earlier helper in `attenuation_audit/reproduction/s5b_contamination_pilot/`, as in the
reported run. `simulation_core.py` and `generate_downstream_simulation.py` in this folder are
identical to the released files in `reproduction/s5_downstream_simulation/`.

| File | SHA-256 (first 12) |
|---|---|
| `attenuation_audit/.../audit_s5_fit_selection.py` | `364edd88383a` |
| `attenuation_audit/.../reml_tools.py` (earlier version) | `217818efbb01` |

## `RELEASE_HASHES.csv`

For every maintained script: its release path and hash, the original file name and hash,
the executed snapshot it corresponds to, and the change made. Maintained copies differ from
the originals only in docstrings and comments, apart from two recorded executable changes:
the repository-path default in `downstream_simulation.py` (`DEFAULT_REPO_ROOT`), and an
optional `--replicates` argument added to `make_S4_tables.py` so that the S4 tables can be
regenerated from the compact public replicate table (without the argument the script behaves
as before). For all other maintained copies, equivalence was checked by comparing the Python
syntax trees with docstrings removed.

## Not distributed

- The extracts of the original AABC2 analysis notebook (`s4_source_code_1.txt`,
  `s4_source_code_2.txt`). They contain participant identifiers. The two schedule-export
  scripts in `../s4_operating_audit/restricted/` need them and check their hashes; those
  scripts therefore run only on an authorized user's machine.
- The restricted-derived schedule files (`s4_refit_schedules.json`) and the attenuation
  study's source-anchor manifest.
