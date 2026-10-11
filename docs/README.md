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
- [m3_tarmac.md](results/m3_tarmac.md): **M3 TarMAC**: same plateau as No-Comm (0.561 vs 0.537, n.s.), reached ~2× faster, unaffected by 0–30 % packet loss
- [m4_point_diabl.md](results/m4_point_diabl.md): **M4 point DIABL + InfER**: 0.974 vs 0.562 for the matched control (paired +0.41); the gain disappears without messages
- [m5_d_diabl.md](results/m5_d_diabl.md): **M5 D-DIABL** (occupancy-map belief): 0.623 at coef 10 (paired +0.06), 0.523 at coef 1; calibrated maps, but far below point DIABL

Rules: numbers are computed from saved raw run records (`metrics.jsonl`, `episodes.jsonl`), not typed
in by hand. Each note states the commit, config, seed and hardware. Superseded results are kept and
marked as such.
