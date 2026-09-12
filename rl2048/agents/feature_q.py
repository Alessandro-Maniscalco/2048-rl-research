"""Linear Q-learning: share ten learned weights across all 2048 boards.

Unlike the exact-board table, this representation generalizes using engineered
features. It is semi-gradient Q-learning, not a convergence-guaranteed tabular
method. Moving a board here computes features; it does not search future turns.
"""

import json
from pathlib import Path

import numpy as np

from rl2048.agents.q_learning import UpdateTrace
from rl2048.game import move

FEATURE_NAMES = (
    "bias", "empty_fraction", "merge_reward_over_128", "largest_exponent_over_16",
    "largest_tile_in_corner", "horizontal_order", "vertical_order",
    "neighbor_similarity", "equal_neighbor_fraction", "corner_weighted_tiles",
)


def features(board: np.ndarray, action: int) -> np.ndarray:
    """Describe the deterministic board AFTER the action, before random spawn.

    Order is high when rows/columns rise or fall consistently. Similarity is
    high when occupied neighbors have close exponents. No feature is a reward
    bonus: training and evaluation still use the original merge reward.
    """
    after, reward, _ = move(board, action)
    logs = np.log2(np.maximum(after, 1))
    horizontal = np.diff(logs, axis=1)
    vertical = np.diff(logs, axis=0)
    horizontal_cost = np.minimum(np.maximum(horizontal, 0).sum(axis=1),
                                 np.maximum(-horizontal, 0).sum(axis=1)).sum()
    vertical_cost = np.minimum(np.maximum(vertical, 0).sum(axis=0),
                               np.maximum(-vertical, 0).sum(axis=0)).sum()
    h_occupied = (after[:, 1:] > 0) & (after[:, :-1] > 0)
    v_occupied = (after[1:, :] > 0) & (after[:-1, :] > 0)
    differences = np.concatenate((horizontal[h_occupied], vertical[v_occupied]))
    smoothness = np.abs(differences).sum() / (24 * 16)
    equals = np.count_nonzero(differences == 0) / 24
    maximum = int(after.max())
    # Choose the best corner orientation of a fixed decreasing spatial map.
    # This supplies an interpretable feature, not a hardcoded action preference.
    corner_map = np.array([[6, 5, 4, 3], [5, 4, 3, 2], [4, 3, 2, 1], [3, 2, 1, 0]]) / 6
    corner_value = max(float((after * np.rot90(corner_map, k)).sum()) for k in range(4))
    return np.array([
        1., np.count_nonzero(after == 0) / 16, reward / 128,
        logs.max() / 16, float(maximum in after[[0, 0, 3, 3], [0, 3, 0, 3]]),
        1 - horizontal_cost / 192, 1 - vertical_cost / 192,
        1 - smoothness, equals, corner_value / max(1, int(after.sum())),
    ], dtype=np.float64)


class FeatureQAgent:
    name = "linear_feature_q_learning"
    display_name = "Q-learning · shared features"

    def __init__(self, alpha: float = 0.01, gamma: float = 0.95, seed: int = 0):
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Invalid learning rate or discount.")
        self.alpha, self.gamma = alpha, gamma
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
        self.updates = 0
        self._cached_board = None
        self._cached_features = None

    def action_features(self, board):
        if self._cached_board is None or not np.array_equal(board, self._cached_board):
            self._cached_board = board.copy()
            self._cached_features = np.array([features(board, a) for a in range(4)])
        return self._cached_features

    def values(self, board):
        return self.action_features(board) @ self.weights

    def act(self, board, action_mask, epsilon=0):
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0, 1].")
        legal = np.flatnonzero(action_mask)
        if not len(legal):
            raise ValueError("No legal action.")
        if self.rng.random() < epsilon:
            return int(self.rng.choice(legal))
        values = self.values(board)
        return int(self.rng.choice(legal[values[legal] == values[legal].max()]))

    def update(self, board, action, reward, next_board, terminated, next_action_mask):
        phi = self.action_features(board)[action].copy()
        old = float(self.weights @ phi)
        legal = np.flatnonzero(next_action_mask)
        if not terminated and not len(legal):
            raise ValueError("Nonterminal next board has no legal action.")
        next_value = 0. if terminated else float(self.values(next_board)[legal].max())
        # Constant reward scaling changes value units, not which policy is best.
        scaled_reward = float(reward) / 128
        target = scaled_reward + self.gamma * next_value
        error = target - old
        # Semi-gradient of 1/2 (target - weights @ phi)^2, target held fixed.
        self.weights += self.alpha * error * phi
        if not np.isfinite(self.weights).all() or np.abs(self.weights).max() > 1e8:
            raise FloatingPointError("Linear Q values diverged; lower alpha or gamma.")
        self.updates += 1
        return UpdateTrace(old, scaled_reward, next_value, target, error, float(self.weights @ phi))

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {"agent": self.name, "format_version": 1, "alpha": self.alpha,
                    "gamma": self.gamma, "updates": self.updates,
                    "feature_names": FEATURE_NAMES, "rng_state": self.rng.bit_generator.state}
        with path.open("wb") as file:
            np.savez_compressed(file, weights=self.weights, metadata=np.array(json.dumps(metadata)))
        return path

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["agent"] != cls.name or meta["format_version"] != 1 or tuple(meta["feature_names"]) != FEATURE_NAMES:
                raise ValueError("Incompatible feature checkpoint.")
            agent = cls(meta["alpha"], meta["gamma"])
            agent.weights = data["weights"].copy()
            agent.updates = meta["updates"]
            agent.rng.bit_generator.state = meta["rng_state"]
        return agent


def load_agent(path):
    """Read the recorded representation; original table checkpoints still work."""
    from rl2048.agents.q_learning import QLearningAgent
    if Path(path).is_dir():
        metadata = json.loads((Path(path) / "metadata.json").read_text())
        if metadata.get("agent") == "neural_current_state":
            from rl2048.agents.neural import NeuralAgent
            return NeuralAgent.load(path)
        if metadata.get("agent") == "q_network_search":
            from rl2048.agents.q_planning import PlanningQAgent
            return PlanningQAgent.load(path)
        if metadata.get("agent") == "deep_q_network_search":
            from rl2048.agents.deep_q_planning import DeepQPlanner
            return DeepQPlanner.load(path)
        from rl2048.agents.ntuple import NTupleAgent
        return NTupleAgent.load(path, mmap_mode="r")
    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["metadata"]))
    return FeatureQAgent.load(path) if meta.get("agent") == FeatureQAgent.name else QLearningAgent.load(path)


class GreedyMergeAgent:
    """Non-learning control: choose the largest immediate merge reward.

    This checks whether learned features help beyond just knowing game rules.
    """

    name = "greedy_immediate_merge"
    display_name = "Greedy merge · no learning"

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def act(self, board, action_mask):
        legal = np.flatnonzero(action_mask)
        if not len(legal):
            raise ValueError("No legal action.")
        rewards = np.array([move(board, int(action))[1] for action in legal])
        return int(self.rng.choice(legal[rewards == rewards.max()]))
