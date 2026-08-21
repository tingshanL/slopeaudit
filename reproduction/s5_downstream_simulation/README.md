# S5 downstream-simulation reproduction

The public generator and summarizer use explicit source-anchor and output
paths. Exact manuscript reproduction uses AABC2-derived source anchors that
remain restricted. The repository therefore includes a fully synthetic
four-regime smoke anchor bundle under `reproduction/smoke_inputs/s5_synthetic/`.
Those smoke values are nonreportable and test only the public code path.

## Public smoke run

```bash
python3 reproduction/s5_downstream_simulation/generate_downstream_simulation.py \
  --anchor-manifest reproduction/smoke_inputs/s5_synthetic/synthetic_anchor_manifest.json \
  --output-dir reproduction_smoke_output/s5/generator \
  --mode smoke \
  --workers 1 \
  --reps-per-cell 1

python3 reproduction/s5_downstream_simulation/summarize_downstream_simulation.py \
  --anchor-manifest reproduction/smoke_inputs/s5_synthetic/synthetic_anchor_manifest.json \
  --task-manifest reproduction_smoke_output/s5/generator/task_manifest.csv \
  --simulation-replicates-raw reproduction_smoke_output/s5/generator/simulation_replicate_results_raw.csv \
  --model-fit-diagnostics reproduction_smoke_output/s5/generator/model_fit_diagnostics.csv \
  --optimizer-attempts reproduction_smoke_output/s5/generator/optimizer_attempts.csv \
  --output-dir reproduction_smoke_output/s5/summary \
  --mode smoke
```

Full mode requires the authorized empirical source-anchor manifest and its
relative schedule JSON files. `prepare_source_anchors_restricted.py` reconstructs
those inputs from explicitly supplied authorized AABC2 files.
