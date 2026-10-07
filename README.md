# DynaBelief-Comm-Uncertainty-Aware-Interagent-Belief-Learning-for-Dynamic-Cooperative-Pursuit

Cooperative multi-agent RL on PettingZoo [`pursuit_v4`](https://pettingzoo.farama.org/environments/sisl/pursuit/):
pursuers with local `7x7` views learn to communicate about moving evaders through a
probability-map belief (**D-DIABL**) supervised through a differentiable channel, plus
surprise-prioritized informative-event replay (**SP-InfER**). This extends DIABL/InfER from
Puc et al., *Interagent Beliefs for Learning to Communicate in Large-Scale Multirobot Visual
Object Search*, IEEE T-RO 2026 ([code](https://github.com/JernejPuc/centurymaze)).

## Status

| Milestone | Scope | State |
|---|---|---|
| M0 | packaging, strict config, seeding, logging, CLIs, tests | done |
| M1 | Pursuit wrapper, information boundary, packet-loss channel, random-policy smoke test | done |
| M2 | No-Communication MAPPO | done: capture rate 0.54–0.55 vs 0.029 random ([result](docs/results/m2_baseline.md)) |
| M3 | TarMAC-MAPPO | next |
| M4 | point-belief DIABL + uniform InfER | |
| M5 | D-DIABL | |
| M6 | SP-InfER | |
| M7 | three-seed evaluation | |

`train.policy: random` runs a no-learning smoke rollout (summaries record `"learning": false`).
`train.policy: mappo` trains the No-Communication MAPPO baseline. The first learning result is
in [docs/results/m2_learning_check.md](docs/results/m2_learning_check.md).

Documentation: [docs/](docs/README.md) has the [timeline](docs/timeline.md), the
[decision log](docs/decisions.md) (including how and why decisions changed) and
[results](docs/results/).

**Training runs on SageMaker** ([docs/sagemaker.md](docs/sagemaker.md)); the local machine only
launches jobs and runs tests:

```bash
.venv/bin/python cloud/launch.py train --config configs/no_comm.yaml   # committed HEAD only
.venv/bin/python cloud/launch.py status
```

## Setup (Linux)

Requires Python 3.11. The commands below use a conda-provided interpreter to build a local `.venv`.

```bash
conda create -y -n dynabelief python=3.11
$(conda run -n dynabelief which python) -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cu121   # or the CPU wheel
.venv/bin/python -m pip install -e ".[dev,viz,aws]"
```

Verify:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -m pytest -q
```

## Usage

```bash
# Random-policy smoke rollout (data path only, no learning)
.venv/bin/python scripts/train.py --config configs/smoke.yaml
.venv/bin/python scripts/train.py --config configs/smoke.yaml --set experiment.seed=3

# Evaluate on the fixed evaluation seed list, with 20% packet loss
.venv/bin/python scripts/evaluate.py --config configs/smoke.yaml --episodes 5 --packet-loss 0.2

# Simulator throughput (for sizing training runs)
.venv/bin/python scripts/benchmark_env.py --config configs/default.yaml --steps 5000

# No-Communication MAPPO
.venv/bin/python scripts/train.py --config configs/smoke_mappo.yaml       # 1-minute pipeline check
.venv/bin/python scripts/train.py --config configs/debug_learning.yaml    # easy task, learns in ~2 min
.venv/bin/python scripts/train.py --config configs/no_comm.yaml           # study baseline
.venv/bin/python scripts/train.py --config configs/no_comm.yaml --resume runs/<run>/checkpoints/latest.pt

# Evaluate a checkpoint (uses the config saved inside it); generalization via --set
.venv/bin/python scripts/evaluate.py --checkpoint runs/<run>/checkpoints/final.pt --episodes 50
.venv/bin/python scripts/evaluate.py --checkpoint runs/<run>/checkpoints/final.pt --set env.n_pursuers=6

# Learning curves from the raw metrics
.venv/bin/python scripts/plot_metrics.py runs/<run>
```

MAPPO runs also write `checkpoints/{latest,best,final}.pt`. `best.pt` is selected by the
training-window capture rate and is meant for picking a checkpoint only; reported numbers come
from `evaluate.py` on the held-out evaluation seeds.

Each run writes `runs/<name>_s<seed>_<utc>/` with `config.yaml` (resolved), `metadata.yaml`
(git commit + dirty flag, package versions, Python, device), `episodes.jsonl` (raw per-episode
records), `metrics.jsonl`, and `summary.json`.

## Design contracts

**Information boundary (CTDE).** Every env step returns two objects:
`ActorObs` (local views + agent mask, the only actor inputs) and `PrivilegedInfo`
(global state, evader occupancy, positions: critic, auxiliary labels, and metrics only).
`tests/test_information_boundary.py` checks that each actor observation is exactly the local
crop of the global state, including on capture steps.

**Grid layout.** The simulator indexes maps `[x, y]`. Everything this package emits uses image
layout `row = y, col = x`: local obs `[N, 7, 7, 3]`, global state `[3, Y, X]`, occupancy `[Y, X]`.
Positions stay `(x, y)`. Actions: `0: x-1, 1: x+1, 2: y+1, 3: y-1, 4: stay`.

**Communication timing.** One-step delay: messages produced at step `t` are dropped
independently per directed link with probability `comm.packet_loss` (seeded generator) and
readable at `t+1`. Masks are receiver-major `[receiver, sender]` with no self-messages.

**Seeding.** Each stochastic component draws from its own named stream derived from
`experiment.seed`. Episode `k` of sub-env `i` uses the 64-bit `derive_seed(seed, "train", i, k)`.
Evaluation uses the separate `"eval"` stream, with a per-episode packet-loss stream, so
every method is evaluated on identical episodes with identical drop patterns.

**MAPPO (M2).** Parameter-shared CNN → GRU actor on local views only; centralized critic
V(s, i) on the global state plus a one-hot of agent i's cell plus the episode time fraction.
GAE stops bootstrapping on termination but bootstraps V(terminal obs) on truncation. Updates use
recurrent chunks of `ppo.chunk_length` steps over (env, chunk) units, keeping all agents of an
env together for M3's communication. Gradients do not cross chunk boundaries, so
`chunk_length` bounds the memory horizon the GRU can be trained for. Value targets are
normalized (ValueNorm); actor and critic use separate Adam optimizers.

**Resume.** A checkpoint restores weights, both optimizers, ValueNorm, every RNG stream, and
counters exactly. PettingZoo's simulator state cannot be serialized, so a resumed run starts each
environment on its next episode seed (in-progress episodes are dropped).

**Config.** YAML is mapped onto frozen dataclasses. Unknown or duplicate keys, wrong types,
non-finite floats, and out-of-range values raise `ConfigError`.

### PettingZoo Pursuit quirks the wrapper handles

| Quirk in `pursuit_v4` | Handling |
|---|---|
| `reset()` without a seed draws OS entropy | `PursuitEnv.reset` requires an integer seed |
| Catching the last evader on step `max_cycles` is flagged only as truncation | reported as `terminated` (absorbing state, no value bootstrap) |
| Spawning rejection-samples without a retry limit, so crowded layouts or a small `constraint_window` hang forever | `EnvConfig.validate` rejects layouts where spawning might fail (conservative: at most 39 agents per group on the default 16x16 map) |
| Caught evaders are popped from a list, so indices shift | positions are reported by original evader id, with `-1` once caught |
| `pettingzoo[sisl]` pulls `box2d-py` (SWIG/C++ build), which Pursuit never imports | depend on `pettingzoo` + `pygame` + `scipy` only |
| `pygame.init()` lets SDL install a C-level SIGTERM handler that swallows the signal, so workers ignore `terminate()` and crashed runs hang at exit | `SDL_NO_SIGNAL_HANDLERS=1` is set before pygame initializes |

## Measured simulator throughput

PettingZoo Pursuit is pure Python. Measured on an i5-12500H with the study config
(16x16, 8 pursuers, 30 evaders) and a random policy, no network:

| Processes | env steps/s |
|---|---|
| 1 | ~360 |
| 8 | ~1,900 |
| 14 | ~2,300 |

A single process needs ~7.6 h of simulator time per 10M env steps, so training uses
subprocess workers (`train.num_workers`). These produce trajectories bit-identical to the
in-process backend. Measured MAPPO training on the study config: ~690 env steps/s collection
with 16 envs/8 workers and ~1,200 with 32 envs/12 workers. Each vector step waits for the
slowest worker. Peak VRAM is ~1 GB and RAM ~1.4 GB.

## Repository layout

```
cloud/              launch.py (SageMaker jobs from this machine), sagemaker_entry.py (in-job)
configs/            smoke.yaml, default.yaml, smoke_mappo.yaml, debug_learning.yaml, no_comm.yaml
scripts/            train.py, evaluate.py, benchmark_env.py, plot_metrics.py
docs/               timeline, decision log, SageMaker guide, results/ (tables + plots from raw data)
src/dynabelief/
  config.py         strict dataclass config
  envs/             pursuit.py (wrapper + boundary types), vector.py (auto-reset vector env, workers)
  comm/             channel.py (packet loss + message accounting)
  models/           encoder.py (CNNs), actor_critic.py (recurrent actor, central critic)
  algorithms/       gae.py, mappo.py (PPO losses + update)
  policies/         random_policy.py
  training/         rollout.py (smoke rollout), buffer.py (recurrent minibatches), trainer.py
  evaluation/       evaluate.py (shared eval loop, checkpoint evaluation)
  utils/            seeding.py, logging.py, device.py, value_norm.py, checkpointing.py
tests/
```

## Team

Varun Rao, Sarthak Gopal Sharma (IIT Bhilai)
