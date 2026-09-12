"""A baseline with no learning and no preference among legal actions."""

import numpy as np


class RandomAgent:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, board: np.ndarray, action_mask: np.ndarray) -> int:
        legal = np.flatnonzero(action_mask)
        if len(legal) == 0:
            raise ValueError("Cannot select an action in a terminal state.")
        return int(self.rng.choice(legal))
