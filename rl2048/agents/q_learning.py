"""Tabular Q-learning: every distinct board has its own four action values."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

State = tuple[int, ...]


def state_key(board: np.ndarray) -> State:
    return tuple(int(tile) for tile in board.flat)


@dataclass(frozen=True)
class UpdateTrace:
    """Numbers to inspect in the lesson, all from the actual learning update."""

    old_value: float
    reward: float
    next_value: float
    target: float
    td_error: float
    new_value: float


class QLearningAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.99, seed: int = 0):
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Require 0 < alpha <= 1 and 0 <= gamma <= 1.")
        self.alpha = alpha
        self.gamma = gamma
        self.rng = np.random.default_rng(seed)
        self.q: dict[State, np.ndarray] = {}
        self.visits: dict[State, int] = {}
        self.updates = 0

    def values(self, board: np.ndarray) -> np.ndarray:
        """Reading an unseen board returns zeros without growing the table."""
        return self.q.get(state_key(board), np.zeros(4, dtype=np.float64)).copy()

    def act(self, board: np.ndarray, action_mask: np.ndarray, epsilon: float = 0) -> int:
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1.")
        legal = np.flatnonzero(action_mask)
        if len(legal) == 0:
            raise ValueError("Cannot select an action in a terminal state.")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(legal))
        values = self.values(board)
        best = legal[values[legal] == values[legal].max()]
        # Random ties avoid teaching an accidental preference for 'up'.
        return int(self.rng.choice(best))

    def update(self, board: np.ndarray, action: int, reward: float,
               next_board: np.ndarray, terminated: bool,
               next_action_mask: np.ndarray) -> UpdateTrace:
        """One-step Q-learning. A time-limit truncation still bootstraps.

        target = reward + gamma * max_legal Q(next_board, next_action)
        Q(board, action) += alpha * (target - Q(board, action))

        For natural termination, the target is just the final reward.
        """
        key = state_key(board)
        if key not in self.q:
            self.q[key] = np.zeros(4, dtype=np.float64)
        old_value = float(self.q[key][action])
        if terminated:
            next_value = 0.0
        else:
            legal = np.flatnonzero(next_action_mask)
            if len(legal) == 0:
                raise ValueError("A nonterminal next state must have a legal action.")
            next_value = float(self.values(next_board)[legal].max())
        target = float(reward) + self.gamma * next_value
        td_error = target - old_value
        new_value = old_value + self.alpha * td_error
        self.q[key][action] = new_value
        self.visits[key] = self.visits.get(key, 0) + 1
        self.updates += 1
        return UpdateTrace(old_value, float(reward), next_value, target, td_error, new_value)

    def coverage(self) -> dict:
        unique = len(self.q)
        return {
            "updates": self.updates,
            "unique_states": unique,
            "repeated_updates": self.updates - unique,
            "states_visited_more_than_once": sum(n > 1 for n in self.visits.values()),
            "repeated_update_fraction": (self.updates - unique) / max(1, self.updates),
        }

    def save(self, path: str | Path) -> Path:
        """Portable compressed arrays + JSON metadata, without pickle."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.q)
        metadata = {"format_version": 1, "alpha": self.alpha, "gamma": self.gamma,
                    "updates": self.updates, "rng_state": self.rng.bit_generator.state}
        with path.open("wb") as file:
            np.savez_compressed(
                file,
                states=np.array(keys, dtype=np.int64).reshape(-1, 16),
                q_values=np.array([self.q[k] for k in keys], dtype=np.float64).reshape(-1, 4),
                visits=np.array([self.visits.get(k, 0) for k in keys], dtype=np.int64),
                metadata=np.array(json.dumps(metadata)),
            )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "QLearningAgent":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata["format_version"] != 1:
                raise ValueError("Unsupported checkpoint version.")
            agent = cls(alpha=metadata["alpha"], gamma=metadata["gamma"])
            for board, values, visits in zip(data["states"], data["q_values"], data["visits"], strict=True):
                key = state_key(board)
                agent.q[key] = values.copy()
                agent.visits[key] = int(visits)
            agent.updates = metadata["updates"]
            agent.rng.bit_generator.state = metadata["rng_state"]
        return agent
