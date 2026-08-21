# Installed examples

- `sleepstudy-audit.R`: balanced 18-participant worked audit;
- `btheb-audit.R`: irregular 100-participant worked audit;
- `design-sensitivity.R`: four prospective schedule calculations.

The two fitted-model scripts run the fast decomposition by default. Set
`SLOPEAUDIT_RUN_BOOTSTRAP=true` before sourcing either script to run Question
1. `SLOPEAUDIT_EXAMPLE_B` defaults to the validated `999`; use a smaller value
only for a smoke test.
