"""Paper-based afterstate value learning, with a small readable Python API.

Pattern layouts: Hung Guei's TDL2048+, MIT license, and cited 2048 papers.
https://github.com/moporgic/TDL2048 (4x6patt and 8x6patt definitions).
Implementation of the kernels in this project is independent Python/Numba code.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

from rl2048.fast2048 import make_row_tables, search, value, downgrade_root

LAYOUTS = {
    "4x6": ("012345", "456789", "012456", "45689a"),
    "8x6": ("012456", "456789", "012345", "234569", "01259a", "345678", "134567", "01489a"),
    "4x4": ("0123", "4567", "0145", "1256"),
}


def make_patterns(layout="4x6"):
    """Apply all eight square symmetries while sharing one table per pattern."""
    cells = np.arange(16).reshape(4, 4)
    transforms = [np.rot90(cells, k).ravel() for k in range(4)]
    transforms += [np.rot90(np.fliplr(cells), k).ravel() for k in range(4)]
    return np.array([[transform[[int(c, 16) for c in pattern]] for transform in transforms]
                     for pattern in LAYOUTS[layout]], dtype=np.int64)


@lru_cache(maxsize=1)
def row_tables():
    return make_row_tables()


def encode(board):
    board = np.asarray(board)
    if board.shape != (4, 4) or np.any(board < 0) or np.any((board > 0) & ((board & (board - 1)) != 0)):
        raise ValueError("Expected a 4x4 board of zero or positive powers of two.")
    return np.log2(np.maximum(board, 1)).astype(np.uint8).ravel()


def decode(board):
    return np.where(board > 0, np.left_shift(np.int64(1), board.astype(np.int64)), 0).reshape(4, 4)


class NTupleAgent:
    name = "ntuple_afterstate_td"
    display_name = "Afterstate TD · n-tuples + search"

    def __init__(self, layout="4x6", initial_value=0.0, seed=0, depth=1, cutoff=0.0001, downgrade_threshold=0):
        if layout not in LAYOUTS or not 1 <= depth <= 4:
            raise ValueError("Unknown layout or search depth outside 1..4.")
        self.layout = layout
        self.patterns = make_patterns(layout)
        self.weights = np.full((len(self.patterns), 16 ** self.patterns.shape[-1]),
                               initial_value / (len(self.patterns) * 8), dtype=np.float32)
        self.initial_value = initial_value
        self.depth, self.cutoff = depth, cutoff
        self.downgrade_threshold = downgrade_threshold
        self.rng = np.random.default_rng(seed)
        self.training_games = 0
        self.training_transitions = 0
        self.metadata = {}

    def act(self, board, action_mask):
        root = encode(board)
        if self.downgrade_threshold:
            root = downgrade_root(root, self.downgrade_threshold)
        action = int(search(root, self.weights, self.patterns, *row_tables(), self.depth, self.cutoff))
        if action < 0 or not action_mask[action]:
            raise ValueError("No legal action, or compiled rules disagree with the reference game.")
        return action

    def afterstate_value(self, board):
        return float(value(encode(board), self.weights, self.patterns))

    def save(self, path):
        """Directory checkpoint; weights.npy supports mmap and avoids huge ZIP copies."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "weights.npy", self.weights)
        metadata = {"agent": self.name, "format_version": 1, "layout": self.layout,
                    "downgrade_threshold": self.downgrade_threshold,
                    "initial_value": self.initial_value, "depth": self.depth, "cutoff": self.cutoff,
                    "training_games": self.training_games, "training_transitions": self.training_transitions,
                    "rng_state": self.rng.bit_generator.state, "experiment": self.metadata}
        (path / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        return path

    @classmethod
    def load(cls, path, *, mmap_mode=None):
        path = Path(path)
        meta = json.loads((path / "metadata.json").read_text())
        if meta["agent"] != cls.name or meta["format_version"] != 1:
            raise ValueError("Incompatible n-tuple checkpoint.")
        agent = cls.__new__(cls)
        agent.layout = meta["layout"]
        agent.patterns = make_patterns(agent.layout)
        agent.weights = np.load(path / "weights.npy", mmap_mode=mmap_mode, allow_pickle=False)
        agent.initial_value = meta["initial_value"]
        agent.depth, agent.cutoff = meta["depth"], meta["cutoff"]
        agent.downgrade_threshold = meta.get("downgrade_threshold", 0)
        agent.training_games, agent.training_transitions = meta["training_games"], meta["training_transitions"]
        agent.rng = np.random.default_rng()
        agent.rng.bit_generator.state = meta["rng_state"]
        agent.metadata = meta["experiment"]
        return agent
