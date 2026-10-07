# Timeline

What happened, in order, with the commit and the key number. Decisions are in
[decisions.md](decisions.md), results in [results/](results/).

## 2026-10-07

| Commit | Step | Outcome |
|---|---|---|
| `30050d8` | Repo created | — |
| `aac9663` | **M0 + M1**: package, strict config, seeding, Pursuit wrapper, packet-loss channel, random smoke test | 107 tests; an 8-agent review found 12 issues (hangs, final-step termination, vacuous tests), all fixed. Simulator ~360 env steps/s per process |
| `6b8a7da` | M2.1 subprocess env workers | Bit-identical to in-process |
| `3cf257c` | M2.2 recurrent actor + centralized critic | ~0.3M parameters each |
| `d04c4ef` | M2.3/2.4 buffer, GAE, PPO | GAE matches a brute-force reference |
| `309051a` | M2.5 checkpoints | Update after save + restore equals the uninterrupted update bit-for-bit |
| `abe17da` | M2.6 trainer, CLIs, configs | Found and fixed: SDL swallowing SIGTERM (resume hung), RNG state on GPU |
| `2259b17` | M2.8a synthetic learning tests | Cue and GRU-memory tasks solved; chunk length bounds memory (D15) |
| `7cad1e3` | M2.8b Pursuit learning check | Easy task: capture rate 0.39 → **1.00** on held-out episodes in 2 min ([result](results/m2_learning_check.md)) |
| `43754b6` `d040366` `b056080` | M2 review (8 agents, 13 confirmed findings) | Value-clip centre (D16), contiguous obs, IPC errors, deterministic GPU mode (D10), torch-free workers (−2 GB RAM), 10 new tests, reward explanation corrected (D19) |
| — | Local 5M-step baseline | Capture rate plateaued at **~0.55** from 0.7M steps (random: 0.029); **killed at 3.84M** when the session ended ([result](results/m2_local_baseline_partial.md)) |
| `b53e9d9` | History rewritten | Co-author trailers removed at the owner's request; trees unchanged, hashes changed |
| `6cf6daa` | Training moved to SageMaker (D22, D23) | `cloud/launch.py`, `cloud/sagemaker_entry.py`; GPU instances because CPU updates are ~10× slower |
| `428376d` `d46390a` | Cloud smoke jobs | Two launch bugs found and fixed (hyperparameter types, empty overrides). Third smoke job **completed** on a Tesla T4; commit, metrics, evals and curves synced to S3 |
| `d46390a` | **M2 baseline on SageMaker** (`dynabelief-no-comm-s0-20261007-173719`, `ml.g4dn.8xlarge`, 16 workers) | Running, ~750–900 env steps/s |
