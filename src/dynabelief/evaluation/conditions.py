"""Evaluation conditions: channel models and send budgets (evaluation only, never training).

Label grammar (each part optional except the loss model)::

    iid-p{loss}            independent loss per directed message
    ge-p{loss}-b{burst}    team-wide Gilbert-Elliott: in the bad state every link drops;
                           mean loss = ``loss``, mean bad-run length = ``burst`` steps
    -random-r{rate}        each sender transmits with probability ``rate`` per step
    -periodic-r{rate}      each sender transmits every ``1/rate`` steps (staggered)
    -trigger-t{tau}        a sender transmits when its point belief moved > ``tau`` grid
                           cells since its last transmission (or it never sent)
    -n{pursuers}           zero-shot team size (handled by the caller's env config)
    -greedy                greedy actions (handled by the caller)

Common random numbers: every step draws the same random numbers in the same order
(``[N, N]`` link uniforms, one state uniform, ``[N]`` send uniforms) whatever the condition
or the policy's beliefs, so two policies see identical drop patterns for the same seed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from dynabelief.comm.channel import MessageStats
from dynabelief.utils.seeding import derive_seed

STREAMS = ("eval", "calib", "test")
EVAL_BASE_SEED = 0  # calib/test episodes are the same for every arm and training seed

_LABEL = re.compile(
    r"^(?P<model>iid|ge)-p(?P<loss>[0-9.]+)(?:-b(?P<burst>\d+))?"
    r"(?:-(?P<budget>random|periodic)-r(?P<rate>[0-9.]+)|-trigger-t(?P<tau>[0-9.]+))?"
    r"(?:-n(?P<n>\d+))?(?P<greedy>-greedy)?$"
)


def _num(x: float) -> str:
    return f"{x:g}"


@dataclass(frozen=True)
class Condition:
    loss: float = 0.0
    burst: int | None = None  # None = i.i.d. per link; int = team-wide Gilbert-Elliott
    budget: str | None = None  # None | "random" | "periodic" | "trigger"
    rate: float = 1.0
    tau: float = 0.0
    n_pursuers: int | None = None
    greedy: bool = False

    def __post_init__(self) -> None:
        if not 0.0 <= self.loss <= 1.0:
            raise ValueError(f"loss must be in [0, 1], got {self.loss}")
        if self.burst is not None:
            if self.burst < 1:
                raise ValueError("burst must be >= 1")
            if 0.0 < self.loss < 1.0 and self.loss / (self.burst * (1 - self.loss)) > 1.0:
                raise ValueError(f"no Gilbert-Elliott chain: loss {self.loss}, burst {self.burst}")
        if self.budget not in (None, "random", "periodic", "trigger"):
            raise ValueError(f"unknown budget {self.budget!r}")
        if self.budget in ("random", "periodic") and not 0.0 < self.rate <= 1.0:
            raise ValueError("rate must be in (0, 1]")

    @property
    def label(self) -> str:
        out = f"iid-p{_num(self.loss)}" if self.burst is None else (
            f"ge-p{_num(self.loss)}-b{self.burst}")  # fmt: skip
        if self.budget in ("random", "periodic"):
            out += f"-{self.budget}-r{_num(self.rate)}"
        elif self.budget == "trigger":
            out += f"-trigger-t{_num(self.tau)}"
        if self.n_pursuers is not None:
            out += f"-n{self.n_pursuers}"
        if self.greedy:
            out += "-greedy"
        return out

    @classmethod
    def parse(cls, label: str) -> Condition:
        m = _LABEL.match(label)
        if m is None:
            raise ValueError(f"bad condition label {label!r}")
        if (m["model"] == "ge") != (m["burst"] is not None):
            raise ValueError(f"{label!r}: 'ge' needs -b<burst>, 'iid' must not have it")
        budget = m["budget"] or ("trigger" if m["tau"] is not None else None)
        return cls(
            loss=float(m["loss"]),
            burst=int(m["burst"]) if m["burst"] else None,
            budget=budget,
            rate=float(m["rate"]) if m["rate"] else 1.0,
            tau=float(m["tau"]) if m["tau"] else 0.0,
            n_pursuers=int(m["n"]) if m["n"] else None,
            greedy=bool(m["greedy"]),
        )


def stream_episode_seeds(stream: str, episodes: int) -> list[int]:
    """Episode seeds of a named evaluation stream (shared by every arm and training seed)."""
    if stream not in STREAMS:
        raise ValueError(f"stream must be one of {STREAMS}")
    return [derive_seed(EVAL_BASE_SEED, stream, 0, k) for k in range(episodes)]


class EvalChannel:
    """Delivery masks ``[N_recv, N_send]`` under a ``Condition``; one instance per episode."""

    def __init__(self, n_agents: int, condition: Condition, rng: np.random.Generator,
                 grid_scale: float = 15.0) -> None:  # fmt: skip
        self.n, self.c, self.rng = n_agents, condition, rng
        self.grid_scale = grid_scale  # normalized point belief -> grid cells
        self.t = 0
        self.bad: bool | None = None
        self.last_sent: np.ndarray | None = None
        self.sends = 0
        self.opportunities = 0
        self.stats = MessageStats()
        self._off_diagonal = ~np.eye(n_agents, dtype=bool)
        c = condition
        self._p_gb = 0.0 if c.burst is None or c.loss in (0.0, 1.0) else (
            c.loss / (c.burst * (1 - c.loss)))  # fmt: skip

    @property
    def send_rate(self) -> float:
        return self.sends / self.opportunities if self.opportunities else 0.0

    def _channel_ok(self, u_link: np.ndarray, u_state: float) -> np.ndarray:
        c = self.c
        if c.burst is None:
            return u_link >= c.loss
        if c.loss >= 1.0:
            self.bad = True
        elif c.loss <= 0.0:
            self.bad = False
        elif self.bad is None:
            self.bad = bool(u_state < c.loss)  # start from the stationary distribution
        elif self.bad:
            self.bad = not (u_state < 1.0 / c.burst)
        else:
            self.bad = bool(u_state < self._p_gb)
        return np.full((self.n, self.n), not self.bad)

    def _send(self, alive: np.ndarray, belief: np.ndarray | None, u_send: np.ndarray):
        c = self.c
        if c.budget is None:
            return np.ones(self.n, dtype=bool)
        if c.budget == "random":
            return u_send < c.rate
        if c.budget == "periodic":
            period = max(int(round(1.0 / c.rate)), 1)
            return (self.t + np.arange(self.n)) % period == 0
        # trigger
        if not alive.any():
            return np.zeros(self.n, dtype=bool)
        if belief is None or np.asarray(belief).shape != (self.n, 2):
            raise ValueError("the trigger budget needs the senders' point beliefs [N, 2]")
        belief = np.asarray(belief, dtype=np.float64)
        if self.last_sent is None:
            self.last_sent = np.full((self.n, 2), np.nan)
        moved = np.linalg.norm((belief - self.last_sent) * self.grid_scale, axis=-1)
        send = np.isnan(moved) | (moved > c.tau)
        update = send & alive
        self.last_sent[update] = belief[update]
        return send

    def sample_delivery(self, alive: np.ndarray, belief: np.ndarray | None = None) -> np.ndarray:
        alive = np.asarray(alive, dtype=bool)
        # Fixed draws per step, whatever the condition: common random numbers across arms.
        u_link = self.rng.random((self.n, self.n))
        u_state = float(self.rng.random())
        u_send = self.rng.random(self.n)
        ok = self._channel_ok(u_link, u_state)
        send = self._send(alive, belief, u_send) & alive
        attempted = self._off_diagonal & alive[:, None] & send[None, :]
        delivered = attempted & ok
        self.sends += int(send.sum())
        self.opportunities += int(alive.sum())
        n_sent, n_delivered = int(attempted.sum()), int(delivered.sum())
        self.stats.sent += n_sent
        self.stats.delivered += n_delivered
        self.stats.dropped += n_sent - n_delivered
        self.stats.steps += 1
        self.t += 1
        return delivered
