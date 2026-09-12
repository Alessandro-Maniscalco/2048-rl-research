import numpy as np
import pytest

from rl2048.agents.q_learning import QLearningAgent, state_key
from rl2048.agents.random_agent import RandomAgent
from rl2048.train import TrainConfig, epsilon_at


def boards():
    board = np.zeros((4, 4), dtype=np.int64)
    board[0, :2] = 2
    next_board = np.zeros_like(board)
    next_board[0, 0] = 4
    return board, next_board


def test_hand_calculated_update_masks_illegal_max():
    board, next_board = boards()
    agent = QLearningAgent(alpha=0.1, gamma=0.99)
    agent.q[state_key(board)] = np.array([0., 2., 0., 0.])
    agent.q[state_key(next_board)] = np.array([1000., 5., 10., 2000.])
    trace = agent.update(board, 1, 4, next_board, False, np.array([False, True, True, False]))
    assert trace.old_value == 2
    assert trace.next_value == 10
    assert trace.target == pytest.approx(13.9)
    assert trace.td_error == pytest.approx(11.9)
    assert trace.new_value == pytest.approx(3.19)
    assert agent.q[state_key(board)][1] == pytest.approx(3.19)


def test_terminal_target_and_truncation_bootstrap():
    board, next_board = boards()
    agent = QLearningAgent(alpha=0.1, gamma=0.99)
    agent.q[state_key(next_board)] = np.ones(4) * 10
    terminal = agent.update(board, 0, 4, next_board, True, np.zeros(4, dtype=bool))
    assert terminal.target == 4 and terminal.new_value == pytest.approx(0.4)
    # TimeLimit sets truncated=True, terminated=False; update receives False.
    truncated = agent.update(board, 1, 4, next_board, False, np.ones(4, dtype=bool))
    assert truncated.target == pytest.approx(13.9)


@pytest.mark.parametrize("epsilon", [0., 0.5, 1.])
def test_action_selection_is_legal_and_randomizes_ties(epsilon):
    board, _ = boards()
    agent = QLearningAgent(seed=7)
    agent.q[state_key(board)] = np.array([999., 0., 0., 999.])
    mask = np.array([False, True, True, False])
    actions = [agent.act(board, mask, epsilon) for _ in range(200)]
    assert set(actions) == {1, 2}
    assert 65 < actions.count(1) < 135
    baseline = RandomAgent(seed=7)
    assert {baseline.act(board, mask) for _ in range(100)} == {1, 2}


def test_unseen_reads_are_frozen_and_state_keys_include_entire_board():
    board, next_board = boards()
    agent = QLearningAgent()
    assert state_key(board) != state_key(next_board)
    assert len(state_key(board)) == 16
    agent.values(board)
    agent.act(board, np.ones(4, dtype=bool))
    assert agent.q == {}


def test_coverage_and_checkpoint_roundtrip(tmp_path):
    board, next_board = boards()
    agent = QLearningAgent(seed=23)
    mask = np.ones(4, dtype=bool)
    for _ in range(3):
        agent.update(board, 0, 4, next_board, False, mask)
    assert agent.coverage() == {
        "updates": 3, "unique_states": 1, "repeated_updates": 2,
        "states_visited_more_than_once": 1, "repeated_update_fraction": 2 / 3,
    }
    checkpoint = agent.save(tmp_path / "agent.npz")
    loaded = QLearningAgent.load(checkpoint)
    assert loaded.coverage() == agent.coverage()
    np.testing.assert_array_equal(loaded.values(board), agent.values(board))
    assert loaded.alpha == agent.alpha and loaded.gamma == agent.gamma
    assert [loaded.act(board, mask, 0.5) for _ in range(100)] == [agent.act(board, mask, 0.5) for _ in range(100)]
    empty = QLearningAgent.load(QLearningAgent().save(tmp_path / "empty.npz"))
    assert empty.q == {}


def test_exploration_schedule():
    config = TrainConfig()
    assert epsilon_at(0, config) == 1
    assert epsilon_at(99_999, config) == pytest.approx(0.1)
    assert epsilon_at(200_000, config) == pytest.approx(0.1)
