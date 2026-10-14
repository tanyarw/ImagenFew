# guidance

Carries storms across v14's 5h20 pieces by steering each new piece, while it is generated, to agree with the
end of the previous one (reconstruction guidance, Ho et al. 2022). **No training:** v14's weights never change;
only the strength `w` is chosen, by the pilot's written rules. Plan: [`code_plan/GUIDANCE_PLAN.md`](../code_plan/GUIDANCE_PLAN.md).

| File | Role |
| :--- | :--- |
| `sampler.py` | the guided EDM sampler; with `w = 0` it is exactly `generate_context.sample_block` (tested) |
| `pilot.py` | settings table, chained generation over 32 chains (8 start dates x 4), scoring, pass rules, choice |
| `__main__.py` | `python -m guidance pilot-task --index i`, `pilot-summary`, `pilot-context2h`, `tasks` |
| `tests/` | `python -m unittest discover -s guidance/tests -t .` (needs torch: run inside a job on the cluster) |
| `slurm/` | `submit_pilot.sh`: 20-task GPU array, then summary (CPU), then the 2-hour context check (GPU) |

Each command writes a run record to `results/guidance/records/` (commit, parameters, hashes).
Results: `results/guidance/pilot.json` (all settings, rules, chosen setting), `context2h.json`.

```bash
# cluster, repository root
git pull && GPU_PARTITION=gpu-long guidance/slurm/submit_pilot.sh --dry-run && GPU_PARTITION=gpu-long guidance/slurm/submit_pilot.sh
```
