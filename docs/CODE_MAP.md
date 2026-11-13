# Code map: the whole project on one page

Read this first. The repo looks big, but nine files hold the ideas; the rest is plumbing and tests.

## The project in four sentences

Eight pursuers with small views chase thirty randomly moving evaders and can message each other.
Trained on reward alone they catch 57 %. If, during training, each pursuer must also *predict where
the nearest evader a teammate sees is* ("point belief"), they catch 99 %. We tested why that works,
what else could be predicted, where it breaks, and how to fix the breakage.

## Plain names ↔ config files

| Name in the report | Config (`configs/…yaml`) | One-line meaning |
|---|---|---|
| TarMAC baseline | `tarmac_pos` | messages trained by reward only |
| Point belief | `point_diabl` | predict the nearest evader the team sees; replay discovery moments |
| Map belief (true positions) | `d_diabl_coef10` | predict a probability map of all evaders |
| Map belief (team knowledge) | `d_diabl_posterior` | same map, target = what the team could know |
| Point belief, no channel gradient | `point_diabl_stopgrad` | prediction error may not train senders through messages |
| Point belief, no replay | `point_diabl_noreplay` | no replay of discovery moments |
| Point belief, robust training | `point_diabl_robust` | trained on a randomly unreliable channel |

## The nine files that matter

| File (under `src/dynabelief/`) | What it does |
|---|---|
| `envs/pursuit.py` | Wraps the PettingZoo game. Splits what an agent may see (`ActorObs`) from the full truth used only for training (`PrivilegedInfo`). |
| `models/actor_critic.py` | The network: CNN → GRU memory → messages → action head and prediction head. Critic sees the whole grid (training only). |
| `models/comm.py` | TarMAC messages: each agent sends a key and a value, receivers pay attention to the useful ones. |
| `algorithms/mappo.py` | The PPO update, plus the prediction ("belief") loss. The stop-gradient switch lives here. |
| `beliefs/targets.py`, `beliefs/maps.py`, `beliefs/losses.py` | The training labels: the point target, the map target, and their losses. |
| `beliefs/filter.py` | The "teacher": tracks where each evader could be from what the team has seen (move step, then look step). Makes the team-knowledge map. |
| `comm/channel.py` | Message loss. `PacketLossChannel` = fixed loss; `RandomizedTrainChannel` = robust training (clean / bursty / low send rate per episode). |
| `evaluation/conditions.py` | Test-time channels: independent loss, bursty loss, send budgets, team size. Same random numbers for every method. |
| `evaluation/aggregate.py` | Statistics: paired differences per seed, confidence intervals, confirmed / not confirmed. |

Supporting: `training/trainer.py` (the training loop), `training/buffer.py` (stores a rollout),
`replay/infer.py` (replay of discovery moments), `config.py` (strict settings), `tests/` (330 tests).

## How a result is produced

```
scripts/train.py --config configs/<name>.yaml --set experiment.seed=<s>     # one trained model (final.pt)
scripts/eval_grid.py --checkpoint <name>=<final.pt> --plan core|channel      # 100 shared test episodes per condition
scripts/aggregate_results.py runs/evals/final --out docs/results/final       # tables, confidence intervals, verdicts
scripts/make_report_figures.py runs/evals/final --out <figures folder>       # the report's figures
```

Thresholds for each claim were written in `configs/analysis/hypotheses.yaml` before the runs. The
short IDs there map to plain questions:

| ID in the file | Plain question | Result |
|---|---|---|
| R1 | Does the point belief beat the baseline? | Yes, 1.73× |
| C1a, C1b | Is the team-knowledge map informative, and better than the true-position map? | No |
| M1 | Is the gradient through the messages needed? | No (1.00 without it) |
| M2 | Is the replay needed? | Mostly not |
| C2a | Is a "send when my belief changed" rule better than random sending? | Inconclusive |
| C2b | Does bursty loss hurt more than spread-out loss? | Yes (0.81 vs 0.99) |
| AR1, AR2, AR3 | Does robust training fix bursts and low budgets at no clean-channel cost? | Yes: both gains confirmed; clean-channel change −0.002 (allowed −0.02) |
| FD, FD-2 | A single-evader heatmap belief | Not run |

## Likely viva questions

**Why not report reward?** The simulator pays the catch reward to every pursuer on the evader's cell, so
reward mixes speed and crowding. Capture rate is clean.

**Do agents see hidden information?** No. Agents get only their 7×7 view, own position and messages.
Labels and the critic use the truth during training only; tests check this boundary.

**Why did the map fail?** Evaders move randomly, so a sighting goes stale in a few steps. Even our teacher,
which uses everything the team saw, removes only about 15 % of the uncertainty about unseen cells.

**If stop-gradient changes nothing, what is the method doing?** The prediction task acts as an extra
training signal for each agent's own network: predicting where a visible evader is teaches searching
and tracking. It is not improving the messages.

**Is robust training "better communication"?** No, and we say so. The robustly trained team catches
99.9 % with no messages at all: it learned not to depend on the channel.

**How do you know results aren't luck?** Same 100 test episodes for every method, 2–3 training seeds
each, thresholds fixed before the runs, confidence intervals from resampling paired episodes.

**What was AI-written?** Most code and report drafts (Claude Code), under our direction; disclosed in the
report's appendix with our prompts.
