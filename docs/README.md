# Documentation

| File | Read it for |
|---|---|
| [timeline.md](timeline.md) | What was done when: commit, step, key number |
| [decisions.md](decisions.md) | Every design decision with its evidence, including decisions that were changed and why |
| [sagemaker.md](sagemaker.md) | How training jobs are launched, monitored and downloaded |
| [results/](results/) | One note per result: setup, table, curves, observations |

Results so far:

- [m2_learning_check.md](results/m2_learning_check.md): MAPPO solves an easy Pursuit variant (capture rate 0.39 → 1.00)
- [m2_baseline.md](results/m2_baseline.md): **M2 baseline** (No-Comm MAPPO, SageMaker): capture rate 0.54–0.55 vs 0.029 random on 50 held-out episodes
- [m2_local_baseline_partial.md](results/m2_local_baseline_partial.md): first attempt on the PC, interrupted at 3.84M steps (superseded; same plateau)

Rules: numbers are computed from saved raw run records (`metrics.jsonl`, `episodes.jsonl`), not typed
in by hand. Each note states the commit, config, seed and hardware. Superseded results are kept and
marked as such.
