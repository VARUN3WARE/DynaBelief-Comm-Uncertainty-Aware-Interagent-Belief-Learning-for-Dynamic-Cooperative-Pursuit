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
| `8035b23` | **M3 TarMAC run** (`dynabelief-tarmac-s0-20261010-163342`, `ml.g4dn.8xlarge`, $5.2) | Capture rate **0.561** vs 0.537 No-Comm (paired Δ +0.024 ± 0.032, n.s.); same plateau reached ~2× faster; flat under 0–30 % packet loss ([result](results/m3_tarmac.md)). **M3 complete** |
| `a9762d1` | **M4.1** optional own-position input (D31) | Needed for absolute-coordinate beliefs; off by default, M2/M3 unchanged |
| `80e0183` | **M4.2** point-belief labels | Nearest team-visible evader; aligned with the simulator on 50+ real targets |
| `88ab496` | **M4.3** point-belief DIABL head + loss | A receiver's belief trains the sender only through delivered messages; logged sender-side gradient |
| `bd04fae` | **M4.4** uniform InfER + configs `tarmac_pos`, `point_diabl` | Windows never cross episode ends; replay halves the replayed belief loss; 232 fast tests pass |
| `bd04fae` | M4 smoke job on AWS (`dynabelief-smoke-diabl-20261010-164844`) | Completed: belief loss falls, sender gradient non-zero, events/windows/replay work |
| `64def42` `d58a27f` `07ab675` `29744fc` | Belief eval metrics + plots, eval-only jobs, `compare_runs.py`; fixes for old-checkpoint compatibility and live metric sync | — |
| `3628b49` | **M4 runs**: `tarmac_pos` (control) and `point_diabl`, 5M steps each ($20 incl. evals) | **point DIABL 0.974 vs 0.562** capture rate (paired Δ **+0.41 ± 0.03**), 0.966 at 50 % loss; with no messages it falls to 0.561, so the gain is communication. `best.pt` 1.000. Late-training collapse/recovery and greedy collapse noted ([result](results/m4_point_diabl.md)). **M4 complete** |
| `7c3844f` | **M5.1** map targets + calibration metrics | (x, y) → target[y, x] verified on real resets; ECE ≈ 0 when calibrated |
| `b9e5610` | **M5.2** D-DIABL map head + BCE through the channel | Prior-initialized; receiver's map trains senders only via delivered messages; PPO pushes BCE below the best constant |
| `22f8d44` | **M5.3** map evaluation (BCE / Brier / ECE vs prior and own-view maps), map plots, `d_diabl.yaml` | AWS test job: **251 passed**; D-DIABL smoke job completed |
| `22f8d44` | **M5 runs**: `d_diabl` (`ml.g4dn.16xlarge`, as point DIABL) and `d_diabl_coef10` (`ml.g5.8xlarge`) | Running, 5M steps each |
