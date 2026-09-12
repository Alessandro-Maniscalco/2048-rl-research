import numpy as np
import pytest
import torch
from rl2048.agents.dqn import QNetwork
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.ntuple import encode, decode, row_tables
from rl2048.agents.q_planning import outcomes, PlanningQAgent
from rl2048.game import move, legal_actions


def test_spawn_expectation_matches_reference_rules():
    board = np.array([[2,2,4,0],[8,0,0,0],[0,0,0,0],[0,0,0,0]])
    states, actions, probabilities, rewards = outcomes(encode(board), *row_tables())
    for action in np.flatnonzero(legal_actions(board)):
        after, reward, _ = move(board, int(action))
        select = actions == action
        assert probabilities[select].sum() == pytest.approx(1.)
        assert np.all(rewards[select] == reward)
        for state, probability in zip(states[select], probabilities[select]):
            difference = decode(state) - after
            assert np.count_nonzero(difference) == 1
            tile = difference.max()
            assert tile in (2, 4)
            assert probability == pytest.approx((.9 if tile == 2 else .1) / (after == 0).sum())


def test_zero_leaf_values_recover_immediate_merge_reward():
    model = QNetwork()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    planner = PlanningQAgent(NeuralAgent(model, 'double_dqn'))
    board = np.array([[2,2,4,0],[2,0,0,0],[0,0,0,0],[0,0,0,0]])
    scores = planner.action_values(board)
    for action in np.flatnonzero(legal_actions(board)):
        assert scores[action] == pytest.approx(move(board, int(action))[1] / 128)
    assert legal_actions(board)[planner.act(board, legal_actions(board))]


def test_terminal_board_and_non_q_network_rejected():
    agent = NeuralAgent(QNetwork(), 'double_dqn')
    planner = PlanningQAgent(agent)
    board = np.array([[2,4,2,4],[4,2,4,2],[2,4,2,4],[4,2,4,2]])
    assert np.all(np.isneginf(planner.action_values(board)))
    with pytest.raises(ValueError):
        planner.act(board, legal_actions(board))
    with pytest.raises(ValueError):
        PlanningQAgent(NeuralAgent(QNetwork(), 'ppo'))


def test_two_moves_match_independent_reference_expectation():
    model = QNetwork()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    planner = PlanningQAgent(NeuralAgent(model, 'double_dqn'), gamma=.9, depth=2)
    board = np.array([[2,2,4,8],[4,8,16,32],[8,16,32,64],[16,32,64,128]])
    scores = planner.action_values(board)
    for action in np.flatnonzero(legal_actions(board)):
        after, reward, _ = move(board, int(action))
        continuation = 0.
        for cell in np.argwhere(after == 0):
            for tile, probability in ((2,.9),(4,.1)):
                next_board = after.copy()
                next_board[tuple(cell)] = tile
                best_reward = max(move(next_board, a)[1] for a in range(4))
                continuation += probability * best_reward / (after == 0).sum()
        assert scores[action] == pytest.approx((reward + .9*continuation)/128)


def test_checkpoint_retains_search_and_reward_settings(tmp_path):
    from rl2048.agents.feature_q import load_agent
    torch.manual_seed(42)
    planner = PlanningQAgent(NeuralAgent(QNetwork(),'double_dqn'),gamma=.97,
                            reward_mode='corner_snake',shaping_scale=3.,depth=2)
    planner.save(tmp_path)
    loaded = load_agent(tmp_path)
    board = np.array([[2,2,4,8],[4,8,16,32],[8,16,32,64],[16,32,64,128]])
    np.testing.assert_allclose(planner.action_values(board),loaded.action_values(board))
    assert loaded.depth == 2 and loaded.gamma == .97
