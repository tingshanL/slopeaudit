# S9 validation and planning

`make_planning_table.py` is the script refactor of the frozen practical-design notebook. It is fully analytic/synthetic: it reads no participant data and fits no empirical model.

The default design manifest contains four illustrative schedules (three waves over two years, five waves over five years, ten daily observations, and a 30-day EMA burst) crossed with three residual-precision scenarios. The script evaluates direct-change reliability from the nuisance-orthogonal GLS contrast and reports Question 1 planning power for the stated equal-schedule Gaussian direct-contrast variance test with known residual covariance. Those power values are planning references, not exact power for the fitted-model parametric bootstrap used after fitting.

Run:

```bash
python3 reproduction/s9_validation_planning/make_planning_table.py \
  --output-dir reproduction_smoke_output/s9
```

The script writes exactly five small deterministic outputs:

- `practical_design_grid_full_S9.csv`
- `practical_design_assumptions.csv`
- `practical_design_table_main.csv`
- `practical_design_table_main.tex`
- `practical_design_contract.json`

The JSON contract records relative output filenames rather than machine-specific paths.
