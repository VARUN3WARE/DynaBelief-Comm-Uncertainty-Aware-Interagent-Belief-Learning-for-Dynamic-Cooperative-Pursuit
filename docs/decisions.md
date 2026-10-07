# Decision log

One entry per decision: what was chosen, the evidence, and what it replaced. Superseded
entries stay, so the change in reasoning is visible. Newest last within each area.
Commit hashes refer to the current (rewritten) history.

## Environment and data

| ID | Date | Decision | Evidence / reason |
|---|---|---|---|
| D1 | 10-07 | PettingZoo `pursuit_v4` (Parallel API), 16×16, 8 pursuers, **30 evaders** (PettingZoo default), 7×7 views, 500 steps | Project plan; all values configurable |
| D2 | 10-07 | Strict split `ActorObs` (local view + mask) vs `PrivilegedInfo` (global state, occupancy, positions) | CTDE boundary; a test checks every actor view equals the local crop of the global state, including on capture steps |
| D3 | 10-07 | One grid layout everywhere: `row = y, col = x` | Simulator maps are `[x, y]`, PettingZoo views are `[y, x]`; a transpose bug is caught by a test |
| D4 | 10-07 | Catching the last evader on the `max_cycles` step counts as **terminated** | PettingZoo flags only truncation there, so MAPPO would bootstrap from an absorbing state |
| D5 | 10-07 | Reject layouts where spawning could hang: `5·(n−1) < free cells` in the smallest spawn window | PettingZoo's rejection sampler has no retry limit (8×8 grid, 100 evaders and `constraint_window=0.5` all hung). Conservative: ≤ 39 agents per group on 16×16 |
| D6 | 10-07 | Depend on `pettingzoo` + `pygame` + `scipy`, not `pettingzoo[sisl]` | The extra forces a Box2D C++ build that Pursuit never imports |
| D7 | 10-07 | Set `SDL_NO_SIGNAL_HANDLERS=1` before pygame starts | `pygame.init()` swallowed SIGTERM: workers ignored `terminate()` and a crashed run hung at exit |

## Reproducibility

| ID | Date | Decision | Evidence / reason |
|---|---|---|---|
| D8 | 10-07 | Named RNG streams from one seed; episode `k` of env `i` uses `derive_seed(seed, stream, i, k)` | Each episode reproducible on its own; train and eval streams separate |
| D8b | 10-07 | **Changed:** derived seeds 32 → **64-bit**; eval packet-loss stream keyed per episode | Review found real train/eval seed collisions possible at 32 bits; per-episode keying gives every method identical drop patterns |
| D9 | 10-07 | Subprocess env workers must give **bit-identical** trajectories to in-process | Tested for 0/1/2/5 workers, and training weights are identical with 0 vs 2 workers |
| D10 | 10-07 | `experiment.deterministic: true` for runs that enter comparisons | Measured: two CUDA runs of the same seed differ by default and are bit-identical with deterministic kernels |
| D11 | 10-07 | Tests pin PyTorch to 2 CPU threads; learning tests use the median of 3 seeds | Results changed with thread count (reduction order); parallel jobs thrashed at 770 % CPU each; single seeds stall on 4 of 5 cues |
| D12 | 10-07 | SageMaker jobs run a `git archive` of a clean, committed `HEAD`; the commit is recorded in the run | Every result maps to an exact commit; the launcher refuses a dirty tree |

## Learning algorithm (M2)

| ID | Date | Decision | Evidence / reason |
|---|---|---|---|
| D13 | 10-07 | MAPPO: shared CNN → GRU actor on local views; critic V(s, i) = global state + one-hot own cell + episode time fraction | CTDE; the time fraction lets the critic value the remaining time before truncation |
| D14 | 10-07 | Minibatches over (env, chunk) units, all agents of an env together | Needed once TarMAC mixes agents (M3) |
| D15 | 10-07 | `ppo.chunk_length = 16` | **It bounds the memory horizon the GRU can learn** (truncated BPTT): memory task solved at chunk 4, stalls at 0.6 with chunk 2. Revisit if agents must remember for > 16 steps |
| D16 | 10-07 | **Changed:** value-clip centre = critic's own collection-time output (pre-update ValueNorm stats) | Review: with the old centre, 88 % of value gradients were zero at update 1 on the study task and the critic stayed worse than a constant for 8+ updates |
| D17 | 10-07 | Keep the entropy bonus (0.01) | lr 3e-3 without it collapsed on the memory task for most seeds |
| D18 | 10-07 | **Changed:** batch 16 envs / 8 workers → **32 envs × 128 steps, 12 workers** (4,096 env steps per update), fixed for all methods | Collection 690 → 1,200 env steps/s: each vector step waits for the slowest worker |
| D19 | 10-07 | Lead every comparison with **capture rate and time-to-capture**, not return | `catch_reward` is credited to every pursuer on the evader's cell, so joint captures pay double. Measured catch reward per capture rose 2.56 → 3.51 during training. (An earlier note wrongly blamed tag-reward farming; corrected) |
| D20 | 10-07 | Open: learning-rate decay or `target_kl` before M3 | Local baseline KL rose 0.004 → 0.036 while capture rate was flat |

## Compute

| ID | Date | Decision | Evidence / reason |
|---|---|---|---|
| D21 | 10-07 | Plan: develop on the PC, use SageMaker for the M7 sweeps | PC: RTX 3050 4 GB, 16 threads; simulator ~360 env steps/s per process |
| D22 | 10-07 | **Changed (user decision): all training runs on SageMaker** | The PC is shared with other heavy jobs, and the local 5M-step baseline was killed at 3.84M steps when the session ended |
| D23 | 10-07 | **Changed:** SageMaker **GPU** instances (`ml.g4dn.8xlarge`), not CPU-only ones | One PPO update at the baseline batch: 1.5 s on GPU vs 14.4 s on CPU (4 or 8 threads), i.e. ~5 h of updates per 5M-step run on CPU. The vCPUs run the env workers |
