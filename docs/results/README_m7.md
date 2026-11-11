# Final runs (M7): arms, seeds, stacks, deviations

Hypotheses: [`configs/analysis/hypotheses.yaml`](../../configs/analysis/hypotheses.yaml).
Aggregated outputs: [`final/`](final/) (`verdicts.json`, `tables.md`, `tables.tex`, figures).

## Arms

| Arm | Config | What changes |
|---|---|---|
| ctrl | `tarmac_pos` | TarMAC + own position, no belief loss |
| A | `point_diabl` | point belief (nearest team-visible evader) through the channel + uniform InfER |
| D | `d_diabl_coef10` | occupancy-map belief, target = true occupancy of all evaders, coef 10 |
| C′ | `d_diabl_posterior` | D with only the target changed: the team posterior of a training-only Bayes filter |
| F | `point_diabl_stopgrad` | A with the belief gradient stopped at the messages |
| FD (stretch) | `fd_heatmap` | cut: no compute |

## Seeds and training stacks (completed)

| Seed | Runs | Stack |
|---|---|---|
| 0 | ctrl, A, D (pilots); C′ | pilots: SageMaker T4; C′: Kaggle T4 |
| 1 | ctrl, A, D, C′, F | Kaggle T4 |
| 2 | A | Apple M5 CPU (this machine). F seed 2 not run: weekly Kaggle GPU quota exhausted; Mac F s2 was killed early |

All 5M env steps, frozen hyperparameters. Evaluations: Apple M5 CPU, 1 torch thread, sampled actions, `final.pt`, shared test stream.

## Verdicts (test stream)

| ID | Verdict | Note |
|---|---|---|
| R1 | confirmed | A / ctrl = 1.734 at iid-p0.2 |
| C1a | not detected | remote BCE explained ≈ 0.041 < 0.15 |
| C1b | not detected | C′ − D ≈ −0.079 |
| M1 | incomplete | only F seed 1; A − F = 0 on that seed |
| C2a | not comparable | trigger send rate left ±10 % band around b* |
| C2b | confirmed | iid − bursty ≈ 0.181 at mean loss 0.8 |

## Deviations from the pre-registration draft

| Item | Draft | Final | Why |
|---|---|---|---|
| Pre-registration | sent before runs | not sent; hypotheses committed here before runs | deadline |
| Seeds | 1, 2, 3 per arm | table above | Mac ≈14–20 h/run; Kaggle 2 jobs, ≈30 GPU-h/week |
| Training stack | one Kaggle image | Kaggle T4 + Mac CPU; one stack per seed except C′ s0 vs pilots | compute |
| Episodes per condition | 200 | 100 | evaluation time |
| Delay, 8-bit, B, E, FD | planned / reserve | cut | time / quota |
| C′/FD code | Sarthak | Claude Code under Varun's direction (AI appendix) | team decision 2026-11-10 |
