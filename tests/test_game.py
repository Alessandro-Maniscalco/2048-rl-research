import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl2048.game import Game2048, legal_actions, merge_left, move


@pytest.mark.parametrize("row,expected,reward", [
    ([0, 2, 0, 2], [4, 0, 0, 0], 4),
    ([2, 2, 4, 0], [4, 4, 0, 0], 4),
    ([2, 2, 2, 2], [4, 4, 0, 0], 8),
    ([4, 4, 8, 8], [8, 16, 0, 0], 24),
    ([0, 0, 0, 0], [0, 0, 0, 0], 0),
    ([64, 64, 8192, 8192], [128, 16384, 0, 0], 16512),
])
def test_merge_once_and_score(row, expected, reward):
    actual, actual_reward = merge_left(np.array(row))
    np.testing.assert_array_equal(actual, expected)
    assert actual_reward == reward


@pytest.mark.parametrize("action,positions", [
    (0, [(0, 1), (0, 2)]), (1, [(1, 2), (1, 3)]),
    (2, [(3, 1), (3, 2)]), (3, [(1, 0), (1, 1)]),
])
def test_direction_labels(action, positions):
    board = np.zeros((4, 4), dtype=np.int64)
    board[1, 1:3] = [2, 4]
    original = board.copy()
    actual, reward, moved = move(board, action)
    expected = np.zeros_like(board)
    for position, value in zip(positions, [2, 4], strict=True):
        expected[position] = value
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(board, original)
    assert reward == 0 and moved


def test_invalid_move_does_not_spawn_or_consume_randomness():
    first, second = Game2048(), Game2048()
    for env in (first, second):
        env.reset(seed=5)
        env.board[:] = 0
        env.board[0, 0] = 2
    observation, reward, terminated, truncated, info = first.step(0)
    assert reward == 0 and not terminated and not truncated and not info["moved"]
    np.testing.assert_array_equal(observation, second.board)
    assert info["score"] == 0
    np.testing.assert_array_equal(first.step(1)[0], second.step(1)[0])


def test_reset_seed_and_observation_copy():
    env = Game2048()
    board, _ = env.reset(seed=42)
    assert board.dtype == np.int64 and board.shape == (4, 4)
    assert np.count_nonzero(board) == 2
    assert set(board.flat) <= {0, 2, 4}
    expected = board.copy()
    board[:] = 128
    np.testing.assert_array_equal(env.board, expected)
    np.testing.assert_array_equal(env.reset(seed=42)[0], expected)


def test_spawn_distribution_and_empty_cell_selection():
    env = Game2048()
    env.reset(seed=123)
    positions = np.zeros(16, dtype=int)
    fours = 0
    for _ in range(8000):
        env.board[:] = 0
        env._spawn_tile()
        position = np.flatnonzero(env.board)[0]
        positions[position] += 1
        fours += int(env.board.flat[position] == 4)
    assert 0.08 < fours / 8000 < 0.12
    assert np.all((positions > 380) & (positions < 620))
    env.board[:] = 8
    env.board[2, 1] = 0
    env._spawn_tile()
    assert env.board[2, 1] in (2, 4)
    assert np.count_nonzero(env.board == 8) == 15


def test_terminal_mask_and_time_limit_are_distinct():
    blocked = np.array([[2, 4, 2, 4], [4, 2, 4, 2], [2, 4, 2, 4], [4, 2, 4, 2]])
    assert not legal_actions(blocked).any()
    env = Game2048()
    env.reset(seed=0)
    env.board = blocked.copy()
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and reward == 0
    assert not info["action_mask"].any()
    with pytest.raises(RuntimeError):
        env.step(0)
    limited = gym.wrappers.TimeLimit(Game2048(), max_episode_steps=1)
    _, info = limited.reset(seed=0)
    _, _, terminated, truncated, _ = limited.step(int(np.flatnonzero(info["action_mask"])[0]))
    assert truncated and not terminated


def test_2048_is_not_terminal_and_merge_reward_is_raw_score():
    env = Game2048()
    env.reset(seed=0)
    env.board[:] = 0
    env.board[0, :2] = [1024, 1024]
    board, reward, terminated, _, info = env.step(3)
    assert reward == info["score"] == 2048
    assert board[0, 0] == 2048 and not terminated
    assert np.count_nonzero(board) == 2


def test_seeded_trajectories_and_contract():
    first, second = Game2048(), Game2048()
    _, info = first.reset(seed=10)
    second.reset(seed=10)
    for _ in range(50):
        action = int(np.flatnonzero(info["action_mask"])[0])
        one, two = first.step(action), second.step(action)
        np.testing.assert_array_equal(one[0], two[0])
        assert one[1:4] == two[1:4]
        info = one[4]
        if one[2]:
            break
    check_env(Game2048(), skip_render_check=True)


def test_step_requires_reset_and_valid_action():
    env = Game2048()
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=0)
    with pytest.raises(ValueError):
        env.step(4)
