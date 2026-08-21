# Step 4 Batch 3 — advanced diagnostics validation entry points

Adds notebook-independent public validation entry points for the already
validated advanced profile, influence, adequacy, and operating-simulation
branches.

Scientific design:

- imports the 53-function advanced computational core frozen in Batch 2;
- preserves the validated computation/comparison tails from the private smoke
  validators;
- removes notebook parsing/execution from the public workflow;
- keeps participant-linked influence/deletion outputs restricted;
- provides a zero-refit operating-simulation preflight for the complete
  46-cell / 62,000-row final state;
- retains the five-case operating repair smoke as an optional targeted refit,
  not something required by the Batch-3 release validation;
- removes internal repair labels from public comments/messages.

Files:

- `run_aabc2_profiles_influence_adequacy.py`
- `validate_aabc2_advanced_profiles.py`
- `validate_aabc2_advanced_influence.py`
- `validate_aabc2_advanced_adequacy.py`
- `run_aabc2_operating_simulation.py`
- `preflight_aabc2_operating_simulation.py`
- `validate_aabc2_operating_simulation_repairs.py`
- `advanced_diagnostics_validation_origin.csv`
- private release-prep validator:
  `private_staging/s7s8_refactor/validate_step4_batch3_20260819.py`

The Batch-3 validator deliberately performs zero profile, influence/deletion,
adequacy, or operating-repair refits. Those computations were already
validated in the private refactor; Batch 3 proves that the public
notebook-independent execution tails are source-equivalent and reruns the
zero-refit 62,000-row operating preflight.
