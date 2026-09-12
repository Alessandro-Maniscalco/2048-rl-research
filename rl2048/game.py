"""The game knows the rules; it knows nothing about learning."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

ACTION_NAMES = ("up", "right", "down", "left")
Board = np.ndarray


def merge_left(row: np.ndarray) -> tuple[np.ndarray, int]:
    """Slide one row left, merging each original tile at most once.

    [2, 2, 4, 0] -> [4, 4, 0, 0], reward 4 (not [8, 0, 0, 0]).
    Python ints avoid narrow-integer overflow during arithmetic.
    """
    tiles = [int(tile) for tile in row if tile != 0]
    merged = []
    reward = 0
    index = 0
    while index < len(tiles):
        if index + 1 < len(tiles) and tiles[index] == tiles[index + 1]:
            value = 2 * tiles[index]
            merged.append(value)
            reward += value
            index += 2
        else:
            merged.append(tiles[index])
            index += 1
    merged.extend([0] * (4 - len(merged)))
    return np.array(merged, dtype=np.int64), reward


def move(board: Board, action: int) -> tuple[Board, int, bool]:
    """Pure deterministic move, without spawning. Never modifies the input.

    Rotate so the requested direction points left, merge, then rotate back.
    Separating this from spawning lets masks and tests inspect the same rules.
    """
    if action not in range(4):
        raise ValueError("Action must be 0=up, 1=right, 2=down, or 3=left.")
    rotations = (1, 2, 3, 0)[action]
    oriented = np.rot90(board, rotations)
    shifted = np.zeros((4, 4), dtype=np.int64)
    reward = 0
    for index, row in enumerate(oriented):
        shifted[index], row_reward = merge_left(row)
        reward += row_reward
    result = np.rot90(shifted, -rotations).copy()
    return result, reward, not np.array_equal(board, result)


def legal_actions(board: Board) -> np.ndarray:
    """Four booleans in ACTION_NAMES order; True means the board changes."""
    return np.array([move(board, action)[2] for action in range(4)], dtype=bool)


class Game2048(gym.Env):
    """Standard 2048; reaching 2048 does not end an episode.

    Observations are independent int64 copies containing actual tile values.
    The base game never truncates. Experiments may wrap it in TimeLimit.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(self, render_mode: str | None = None):
        super().__init__()
        if render_mode not in (None, "ansi"):
            raise ValueError("Use render_mode='ansi' or the notebook viewer.")
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0, high=np.iinfo(np.int64).max, shape=(4, 4), dtype=np.int64
        )
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.steps = 0
        self._terminated = False
        self._has_reset = False

    def _spawn_tile(self) -> None:
        empty = np.argwhere(self.board == 0)
        row, column = empty[self.np_random.integers(len(empty))]
        self.board[row, column] = 2 if self.np_random.random() < 0.9 else 4

    def _info(self, mask: np.ndarray, moved: bool = False) -> dict:
        return {
            "action_mask": mask,
            "score": self.score,
            "max_tile": int(self.board.max()),
            "steps": self.steps,
            "moved": moved,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.board = np.zeros((4, 4), dtype=np.int64)
        self.score = 0
        self.steps = 0
        self._terminated = False
        self._has_reset = True
        self._spawn_tile()
        self._spawn_tile()
        return self.board.copy(), self._info(legal_actions(self.board))

    def step(self, action: int):
        if not self._has_reset or self._terminated:
            raise RuntimeError("Call reset() before stepping a new episode.")
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")
        board, reward, moved = move(self.board, int(action))
        self.steps += 1
        if moved:
            self.board = board
            self.score += reward
            self._spawn_tile()
        mask = legal_actions(self.board)
        self._terminated = not bool(mask.any())
        return (
            self.board.copy(), float(reward), self._terminated, False,
            self._info(mask, moved),
        )

    def render(self) -> str:
        rows = [" ".join(f"{int(tile):6d}" if tile else "     ." for tile in row)
                for row in self.board]
        return f"Score: {self.score}\n" + "\n".join(rows)
