"""A batch of independent games; compiled rules, explicit reset boundaries.

The learner receives final next states BEFORE auto-reset, so time limits can
bootstrap correctly. Observations for the next interaction come from boards.
"""
import numpy as np
from numba import njit
from rl2048.fast2048 import slide, spawn
from rl2048.agents.ntuple import row_tables


@njit(cache=True)
def legal_masks(boards, rows, rewards):
    masks = np.empty((len(boards), 4), np.bool_)
    for i in range(len(boards)):
        for action in range(4):
            masks[i, action] = slide(boards[i], action, rows, rewards)[2]
    return masks


@njit(cache=True)
def reset_boards(count, seed):
    np.random.seed(seed)
    boards = np.zeros((count, 16), np.uint8)
    for board in boards:
        spawn(board)
        spawn(board)
    return boards


@njit(cache=True)
def step_boards(boards, actions, scores, lengths, rows, row_rewards, max_steps):
    rewards = np.zeros(len(boards), np.float32)
    final_states = np.empty_like(boards)
    terminated = np.zeros(len(boards), np.bool_)
    truncated = np.zeros(len(boards), np.bool_)
    # Completed episodes: score, length, max tile; zero otherwise.
    episodes = np.zeros((len(boards), 3), np.int64)
    for i in range(len(boards)):
        board, gain, changed = slide(boards[i], actions[i], rows, row_rewards)
        if changed:
            spawn(board)
        rewards[i] = gain
        scores[i] += gain
        lengths[i] += 1
        final_states[i] = board
        any_legal = False
        for action in range(4):
            any_legal = any_legal or slide(board, action, rows, row_rewards)[2]
        terminated[i] = not any_legal
        truncated[i] = lengths[i] >= max_steps and any_legal
        if terminated[i] or truncated[i]:
            episodes[i, 0] = scores[i]
            episodes[i, 1] = lengths[i]
            episodes[i, 2] = 1 << int(board.max())
            board[:] = 0
            spawn(board)
            spawn(board)
            scores[i] = 0
            lengths[i] = 0
        boards[i] = board
    return rewards, final_states, terminated, truncated, episodes


class VectorGame:
    def __init__(self, count=256, seed=0, max_steps=40_000):
        self.rows, self.row_rewards = row_tables()
        self.boards = reset_boards(count, seed)
        self.scores = np.zeros(count, np.int64)
        self.lengths = np.zeros(count, np.int64)
        self.max_steps = max_steps

    def masks(self):
        return legal_masks(self.boards, self.rows, self.row_rewards)

    def step(self, actions):
        return step_boards(self.boards, np.asarray(actions, dtype=np.int64), self.scores,
                           self.lengths, self.rows, self.row_rewards, self.max_steps)
