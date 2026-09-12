import numpy as np
import pytest
import torch

from rl2048.agents.dqn import QNetwork
from rl2048.agents.neural import NeuralAgent
from rl2048.agents.q_planning import PlanningQAgent, outcomes
from rl2048.agents.deep_q_planning import DeepQPlanner, SearchBudgetExceeded, fast_legal_masks
from rl2048.agents.ntuple import row_tables
from rl2048.vector_game import legal_masks


def test_qr_search_uses_mean_quantiles_and_roundtrips(tmp_path):
    from rl2048.agents.qr_dqn import QuantileTransformer
    torch.manual_seed(42)
    model = QuantileTransformer(16, 'exponents', 2, 4, 5).eval()
    qr = NeuralAgent(model, 'qr_dqn', width=16)
    board = np.array([[1,1,2,3,2,3,4,5,3,4,5,6,4,5,6,7]], np.uint8)
    # Independent one-move enumeration: average quantiles before legal max,
    # then average spawn outcomes (not max over a lucky spawn).
    states, actions, probabilities, raw = outcomes(board[0], *row_tables())
    with torch.no_grad():
        q = model.quantile_values(torch.tensor(states)).mean(-1).numpy()
    masks = legal_masks(states, *row_tables())
    future = np.where(masks.any(1), np.where(masks, q, -np.inf).max(1), 0.)
    expected = np.zeros(4)
    np.add.at(expected, actions, probabilities * (raw / 128 + .99 * future))
    expected[~legal_masks(board, *row_tables())[0]] = -np.inf
    planner = DeepQPlanner(qr, depth=1, reward_mode='score')
    np.testing.assert_allclose(planner.planned_values(board)[0], expected, rtol=1e-5)
    planner.save(tmp_path)
    restored = DeepQPlanner.load(tmp_path)
    assert restored.agent.algorithm == 'qr_dqn'
    np.testing.assert_allclose(restored.planned_values(board), planner.planned_values(board))


def test_neighbor_legality_matches_full_move_simulation():
    rng = np.random.default_rng(92)
    boards = rng.integers(0, 18, size=(1000, 16), dtype=np.uint8)
    boards[:300] = rng.integers(0, 3, size=(300,16), dtype=np.uint8)
    boards[0] = 0
    np.testing.assert_array_equal(fast_legal_masks(boards), legal_masks(boards, *row_tables()))


def test_pruning_uses_q_fallback_and_preserves_its_settings(tmp_path):
    torch.manual_seed(23)
    agent = NeuralAgent(QNetwork(), 'double_dqn')
    b = np.array([[1,1,2,3,2,3,4,5,3,4,5,6,4,5,6,7]], np.uint8)
    pruned = DeepQPlanner(agent, depth=5, probability_cutoff=1.)
    # Every chance outcome has probability < 1, so this stops at one move.
    np.testing.assert_allclose(pruned.planned_values(b), DeepQPlanner(agent, depth=1).planned_values(b))
    pruned.save(tmp_path)
    restored = DeepQPlanner.load(tmp_path)
    assert restored.probability_cutoff == 1. and 'pruned' in restored.algorithm
    np.testing.assert_allclose(restored.planned_values(b), pruned.planned_values(b))


def test_merged_search_matches_original_with_shaping_and_duplicate_roots():
    torch.set_num_threads(1)
    torch.manual_seed(11)
    agent = NeuralAgent(QNetwork(), 'double_dqn')
    boards = np.array([[1,1,2,3,2,3,4,5,3,4,5,6,4,5,6,7]] * 2, np.uint8)
    for depth in [1, 2]:
        old = PlanningQAgent(agent, .97, 'corner_snake', 2., depth)
        new = DeepQPlanner(agent, .97, 'corner_snake', 2., depth)
        np.testing.assert_allclose(new.planned_values(boards), old.planned_values(boards, depth), rtol=1e-5, atol=1e-6)
        hashed = DeepQPlanner(agent, .97, 'corner_snake', 2., depth, dedup='hash')
        np.testing.assert_allclose(hashed.planned_values(boards), new.planned_values(boards), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize('depth', [3, 4, 5])
def test_deep_moves_match_independent_scalar_recursion(depth):
    agent = NeuralAgent(QNetwork(), 'double_dqn')
    with torch.no_grad():
        for p in agent.policy.parameters(): p.zero_()
    board = np.array([1,1,2,3,2,3,4,5,3,4,5,6,4,5,6,7], np.uint8)
    def reference(b, depth):
        mask = legal_masks(b[None], *row_tables())[0]
        q = np.zeros(4)
        if depth:
            states, actions, probs, raw = outcomes(b, *row_tables())
            for nxt, a, prob, reward in zip(states, actions, probs, raw):
                future = reference(nxt, depth-1)
                continuation = max(future) if np.isfinite(future).any() else 0.
                q[a] += prob * (reward / 128 + .9 * continuation)
        return np.where(mask, q, -np.inf)
    expected = reference(board, depth)
    actual = DeepQPlanner(agent, gamma=.9, depth=depth).planned_values(board[None])[0]
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_terminal_depth5_and_resource_limit():
    agent = NeuralAgent(QNetwork(), 'double_dqn')
    terminal = np.array([[1,2,1,2,2,1,2,1,1,2,1,2,2,1,2,1]], np.uint8)
    assert np.isneginf(DeepQPlanner(agent, depth=5).planned_values(terminal)).all()
    with pytest.raises(SearchBudgetExceeded):
        DeepQPlanner(agent, depth=3, max_edges=1).planned_values(np.array([[1,1]+[0]*14], np.uint8))


def test_deep_checkpoint_load_preserves_resource_and_reward_settings(tmp_path):
    from rl2048.agents.feature_q import load_agent
    p = DeepQPlanner(NeuralAgent(QNetwork(), 'double_dqn'), gamma=.97, reward_mode='corner_snake',
                     depth=3, max_edges=500000, network_batch=256)
    p.save(tmp_path)
    restored = load_agent(tmp_path)
    assert restored.depth == 3 and restored.max_edges == 500000 and restored.network_batch == 256
    b = np.array([[1,1,2,3,2,3,4,5,3,4,5,6,4,5,6,7]], np.uint8)
    np.testing.assert_allclose(restored.planned_values(b), p.planned_values(b))
