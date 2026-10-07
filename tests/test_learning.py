"""The learner must actually learn: synthetic tasks with known optimal behaviour.

``cue``: each step shows a cue in the centre cell's evader channel; reward 1 for
the action equal to the cue. Random play scores 0.2.

``memory``: the cue is shown only on the first step of a 4-step episode and
must be repeated on every step, so steps 2-4 are solvable only by remembering
it in the GRU. A memoryless policy scores ~0.2 on those steps.
"""

from __future__ import annotations

import pytest
import torch

from dynabelief.algorithms.mappo import MAPPO
from dynabelief.config import PPOConfig
from dynabelief.models.actor_critic import MAPPOPolicy
from dynabelief.training.buffer import RolloutBuffer

OBS, STATE = (7, 7, 3), (3, 16, 16)
E, N, T = 16, 2, 16


def run_task(
    memory: bool,
    iterations: int,
    seed: int = 0,
    chunk_length: int = 4,
    lr: float = 3e-3,
    entropy_coef: float = 0.0,
) -> tuple[float, float]:
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    episode_len = 4 if memory else 1
    policy = MAPPOPolicy(OBS, STATE, n_actions=5, max_cycles=episode_len, hidden_dim=32,
                         conv_channels=8, critic_hidden_dim=32)  # fmt: skip
    cfg = PPOConfig(lr_actor=lr, lr_critic=lr, epochs=4, num_minibatches=2,
                    chunk_length=chunk_length, entropy_coef=entropy_coef, gamma=0.9)  # fmt: skip
    algo = MAPPO(policy, cfg, "cpu", torch.Generator().manual_seed(seed))
    buf = RolloutBuffer(T, E, N, OBS, STATE, hidden_dim=32)
    hidden = policy.actor.initial_state(E, N)
    cue = torch.zeros(E, N, dtype=torch.long)
    accuracy_first = accuracy_later = 0.0
    for _ in range(iterations):
        buf.reset()
        hits_first, hits_later, n_first, n_later = 0, 0, 0, 0
        for t in range(T):
            step_in_episode = t % episode_len
            start = torch.full((E,), step_in_episode == 0)
            if step_in_episode == 0:
                cue = torch.randint(0, 5, (E, N), generator=g)
            obs = torch.zeros(E, N, *OBS)
            state = torch.zeros(E, *STATE)
            if step_in_episode == 0 or not memory:
                obs[:, :, 3, 3, 2] = cue.float()
            state[:, 2, 0, :N] = cue.float()  # critic sees the cue (privileged)
            pos = torch.zeros(E, N, 2, dtype=torch.long)
            pos[:, :, 0] = torch.arange(N)
            step = torch.full((E,), step_in_episode)

            actions, logp, new_hidden = policy.act(obs, hidden, start)
            values = algo.values(state, pos, step)
            correct = actions == cue
            last = step_in_episode == episode_len - 1
            buf.add(
                obs=obs, episode_start=start, actor_hidden=hidden, actions=actions,
                log_probs=logp, values=values, rewards=correct.float(),
                agent_mask=torch.ones(E, N, dtype=torch.bool),
                terminated=torch.full((E,), last), truncated=torch.zeros(E, dtype=torch.bool),
                final_values=torch.zeros(E, N), global_state=state, pursuer_pos=pos, step=step,
            )  # fmt: skip
            hidden = new_hidden
            if step_in_episode == 0:
                hits_first, n_first = hits_first + int(correct.sum()), n_first + correct.numel()
            else:
                hits_later, n_later = hits_later + int(correct.sum()), n_later + correct.numel()
        algo.compute_returns(buf, torch.zeros(E, N))
        algo.update(buf)
        accuracy_first = hits_first / n_first
        accuracy_later = hits_later / n_later if n_later else accuracy_first
    return accuracy_first, accuracy_later


def test_learns_reactive_cue_task():
    # Median over seeds: single seeds can stall on 4 of 5 cues (measured 0.76-0.82),
    # which says nothing about correctness of the learner.
    accuracies = sorted(run_task(memory=False, iterations=40, seed=s)[0] for s in range(3))
    assert accuracies[1] > 0.9, f"median accuracy {accuracies[1]:.2f} (random = 0.2)"


@pytest.mark.slow
def test_learns_memory_task_through_gru():
    # chunk_length must cover the memory horizon: gradients do not cross chunk
    # boundaries (truncated BPTT), so with chunk_length=2 nothing trains the GRU to
    # carry the cue from step 1 into step 2 (measured: accuracy stalls at ~0.6). Exact
    # hidden-state alignment for mid-episode chunk starts is tested separately in
    # test_ppo_math.test_ratio_is_one_before_first_gradient_step.
    # Moderate settings: lr 3e-3 without an entropy bonus collapses on this task for
    # most seeds (measured 0.35-0.78), with or without the value-clip fix.
    runs = [
        run_task(memory=True, iterations=200, seed=s, lr=1e-3, entropy_coef=0.01) for s in range(3)
    ]
    later = sorted(r[1] for r in runs)[1]
    assert later > 0.8, f"median memory accuracy {later:.2f}; memoryless policy ~0.2"
