# S4 operating-study reproduction

This directory contains reproduction code for the
operating-characteristic study reported in Supplement S4.

## Scope

The original notebook workflow is organized into four components:

- `prepare_operating_study_inputs_restricted.py`
  reconstructs the exact empirical visit schedules from authorized AABC2
  inputs. It uses explicit input paths and performs no local file discovery.

- `simulation_core.py`
  contains the numerical mixed-model and covariance helpers used by the
  operating-study generator and targeted refit. It does not read AABC2 data
  or search the filesystem.

- `generate_operating_study.py`
  consumes an explicit ordered design manifest and schedule input, generates
  replicate-level simulation results, supports resumable execution, and
  applies the historical targeted-refit procedure.

- `summarize_operating_study.py`
  performs deterministic post-processing of the final repaired replicate
  results. It performs no model fitting and generates no simulated data.

The manuscript Figure 2 post-processing is implemented separately in:

- `../figures_tables/make_figure_02.py`

## Input requirements

The public code contains no AABC2 participant rows, participant identifiers,
individual trajectories, local-machine search roots, or embedded AABC2 file
paths.

Two inputs required for exact reproduction are derived from restricted AABC2
analyses and are not distributed in this repository:

1. the exact empirical participant visit schedules used by the six
   empirical-schedule simulation cells;
2. the exact empirical generating anchors used by those cells.

`prepare_operating_study_inputs_restricted.py` documents how an authorized
researcher can reconstruct the schedule input from explicitly supplied AABC2
files.

The generator accepts these restricted-derived inputs explicitly rather than
discovering or embedding them.

## Historical design

The completed operating study contains 46 simulation cells and 62,000 requested
replicates:

- 24 balanced-core cells;
- 12 critical near-boundary cells;
- 6 empirical-AABC cells;
- 4 true-zero cells.

The historical cell order is part of the reproducibility contract because
the deterministic random-number seed for each replicate depends on the
cell's original manifest position.

## Historical repair trail

The original operating-study run was not a pristine one-pass analysis.

After the initial 62,000 replicate fits:

- 758 Hessian-nonpositive fits were targeted for multistart refitting;
- all 758 targeted rows were retained in the repair audit;
- the final repaired export still contains 82 rows that fail the strict
  numerical-validity criterion.

The final repaired 62,000-row result is the replicate-level
input to deterministic S4 summarization and Figure 2 post-processing.

## Generator modes

### Smoke mode

Smoke mode is intended to validate the code path without running the full
operating study. It caps the number of replicates per supplied cell and does
not execute the historical 758-fit targeted-refit step.

Example:

```bash
python3 reproduction/s4_operating_study/generate_operating_study.py \
  --design-manifest reproduction/smoke_inputs/s4_synthetic_design.csv \
  --output-dir reproduction_smoke_output/s4/generator \
  --mode smoke \
  --n-jobs 1 \
  --smoke-reps 2
```

The included design is fully synthetic and contains one balanced true-zero cell.
Smoke generation writes a summarizer-compatible replicate file for software
testing; it is not suitable for scientific reporting.
Run the deterministic smoke summarizer with:

```bash
python3 reproduction/s4_operating_study/summarize_operating_study.py \
  --replicates-csv reproduction_smoke_output/s4/generator/operating_study_replicate_results.csv \
  --manifest-csv reproduction_smoke_output/s4/generator/operating_study_design.csv \
  --output-dir reproduction_smoke_output/s4/summary \
  --mode smoke
```

Exact historical full mode additionally requires the prespecified 46-cell design and
restricted-derived empirical schedule/anchor inputs described above. Those inputs
are not replaced by the synthetic smoke design.
