# Final runs (M7): arms, seeds, stacks, deviations

Written 2026-11-10 before the final runs were launched. Hypotheses:
[`configs/analysis/hypotheses.yaml`](../../configs/analysis/hypotheses.yaml).

## Arms

| Arm | Config | What changes |
|---|---|---|
| ctrl | `tarmac_pos` | TarMAC + own position, no belief loss |
| A | `point_diabl` | point belief (nearest team-visible evader) through the channel + uniform InfER |
| D | `d_diabl_coef10` | occupancy-map belief, target = true occupancy of all evaders, coef 10 |
| C′ | `d_diabl_posterior` | D with only the target changed: the team posterior of a training-only Bayes filter |
| F | `point_diabl_stopgrad` | A with the belief gradient stopped at the messages |
| FD (stretch) | `fd_heatmap` | single-target heatmap, KL to the filter posterior of the nearest tracked evader |

## Seeds and training stacks

| Seed | Runs | Stack |
|---|---|---|
| 0 | ctrl, A, D (pilots, Oct 2026), C′ | pilots: SageMaker T4; C′: Kaggle T4 |
| 1 | ctrl, A, D, C′ (+ FD if compute allows) | Kaggle T4 |
| 2 | A, F | Apple M5 CPU (this machine) |

All 5M env steps, frozen hyperparameters. Every run records its stack in `metadata.yaml`.
Evaluations: all on one CPU (Apple M5), 1 torch thread, sampled actions, `final.pt`.

## Deviations from the pre-registration draft

| Item | Draft | Final | Why |
|---|---|---|---|
| Pre-registration | sent before runs | not sent; hypotheses committed here before runs | 2 days left |
| Seeds | 1, 2, 3 per arm | table above | a 5M-step run takes ~14 h on the Mac CPU (PPO update 35–41 s); Kaggle quota ~30 GPU-h/week |
| Training stack | one Kaggle image | Kaggle T4 + Mac CPU; all arms of one seed share a stack, except C′ s0 vs the D/ctrl/A pilots | compute |
| Episodes per condition | 200 | 100 | evaluation time |
| Delay, 8-bit messages, B, E | planned / reserve | cut | time |
| C′/FD code | Sarthak | Claude Code under Varun's direction (disclosed in the AI appendix) | team decision |
