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
| `d46390a` | **M2 baseline on SageMaker** (`dynabelief-no-comm-s0-20261007-173719`, `ml.g4dn.8xlarge`, 16 workers) | 5M steps in 1.5 h ($4.6). Held-out capture rate **0.537 / 0.553** (sampled / greedy) vs 0.029 random; plateau from ~1M steps, as on the PC ([result](results/m2_baseline.md)). **M2 complete** |

## 2026-10-10

| Commit | Step | Outcome |
|---|---|---|
| `abaf135` | **M3.1** TarMAC module in the actor | Undelivered messages: no effect, no gradient; receiver loss reaches sender's encoder; no phantom messages at episode starts |
| `8fada31` | **M3.2** comm through trainer, buffer, checkpoints, eval | Recomputed log-probs equal stored ones **with** messages and differ without them, so comm is really used; channel RNG saved on resume |
| `8ce5b58` | **M3.3** configs + cloud eval sweep + AWS test jobs | `tarmac.yaml` = `no_comm.yaml` + comm; full suite on AWS: **209 passed** (ml.c5.2xlarge, 2.6 min) |
| `8035b23` | Cloud fix | Toolkit passes hyperparameters unquoted to a shell: `;` in the loss sweep ran as commands. Now base64/comma encoded; smoke job on AWS completed with all 5 evals |
| `8035b23` | **M3 TarMAC run** (`dynabelief-tarmac-s0-20261010-163342`, `ml.g4dn.8xlarge`) | Running: 5M steps, evaluated at 0 / 10 / 20 / 30 % packet loss |
