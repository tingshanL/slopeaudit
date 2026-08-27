# Reproduction workflows

Run commands from the repository root. Restore/install the environments first
as described in `environment/README.md`.

## One-command public smoke check

```bash
python3 reproduction/run_public_smoke.py \
  --output-dir reproduction_smoke_output
```

The smoke runner uses only public or fully synthetic inputs. Its outputs test
software execution and are not suitable for scientific reporting. It exercises
S4 generation+summarization, S5 generation+summarization,
all four S6 public audit/sensitivity scripts, and the deterministic S9 planning
table.

| Workflow | Public smoke | Exact/full inputs | Public status |
|---|---:|---|---|
| S4 operating study | yes | prespecified design plus restricted-derived empirical schedule/anchors | exact full run requires authorization |
| S5 downstream simulation | yes | restricted AABC2-derived source-anchor manifest/schedules | exact full run requires authorization |
| S6 Beat the Blues | yes | pinned public HSAUR3 artifact | public |
| S6 sleepstudy | yes | pinned public source or explicit `SLEEPSTUDY_CSV` | public |
| S7/S8 AABC2 | contract/manifest check only | authorized AABC2 source files | restricted-input workflow |
| S9 planning | yes | none; analytic/synthetic | public |

The smoke runner never treats synthetic smoke outputs as manuscript results.
Historical full-run commands and restricted-input reconstruction contracts are
documented in the corresponding workflow directories.
