import json

import numpy as np
import pytest

from rl2048.agents.feature_q import FEATURE_NAMES, FeatureQAgent, features, load_agent
from rl2048.agents.q_learning import QLearningAgent
from rl2048.evaluate import evaluate
from rl2048.game import Game2048, legal_actions
from rl2048.train import TrainConfig
from rl2048.variations import train_features
from rl2048.view import save_replay


def test_features_have_interpretable_values_without_mutating_board():
    board = np.zeros((4, 4), dtype=np.int64)
    board[0, :2] = [64, 64]
    original = board.copy()
    phi = features(board, 3)
    assert len(phi) == len(FEATURE_NAMES)
    assert phi[0] == 1
    assert phi[1] == 15 / 16
    assert phi[2] == 1  # reward 128 / scale 128
    assert phi[3] == 7 / 16  # log2(128) / 16
    assert phi[4] == 1  # largest tile is in a corner
    np.testing.assert_array_equal(board, original)
    assert np.isfinite(phi).all()


def test_semigradient_matches_hand_calculation_and_terminal_bootstrap():
    env = Game2048()
    board, info = env.reset(seed=7)
    action = int(np.flatnonzero(info["action_mask"])[0])
    next_board, reward, _, _, next_info = env.step(action)
    agent = FeatureQAgent(alpha=.01, gamma=.95)
    agent.weights = np.arange(len(FEATURE_NAMES), dtype=float) / 10
    weights = agent.weights.copy()
    phi = features(board, action)
    target = reward / 128 + .95 * max(features(next_board, a) @ weights for a in np.flatnonzero(next_info["action_mask"]))
    error = target - phi @ weights
    trace = agent.update(board, action, reward, next_board, False, next_info["action_mask"])
    np.testing.assert_allclose(agent.weights, weights + .01 * error * phi)
    assert trace.target == pytest.approx(target)
    terminal = agent.update(board, action, 128, next_board, True, np.zeros(4, dtype=bool))
    assert terminal.target == 1 and terminal.next_value == 0


def test_no_hardcoded_policy_and_readonly_evaluation(tmp_path):
    agent = FeatureQAgent(seed=1)
    env = Game2048()
    board, info = env.reset(seed=1)
    np.testing.assert_array_equal(agent.values(board), np.zeros(4))
    assert set(agent.act(board, info["action_mask"]) for _ in range(100)) == set(np.flatnonzero(info["action_mask"]))
    # The first learned update must generalize to a different state.
    next_board, reward, terminated, _, next_info = env.step(1)
    agent.update(board, 1, 128, next_board, terminated, next_info["action_mask"])
    assert np.any(agent.values(next_board) != 0)
    weights = agent.weights.copy()
    rng = agent.rng.bit_generator.state
    summary, _ = evaluate(agent, [4_000_000], max_steps=20)
    np.testing.assert_array_equal(agent.weights, weights)
    assert agent.rng.bit_generator.state == rng
    assert summary["algorithm"] == agent.name
    assert summary["known_state_fraction"] is None  # no exact-board table
    path = agent.save(tmp_path / "features.npz")
    loaded = load_agent(path)
    np.testing.assert_array_equal(loaded.values(board), agent.values(board))
    assert loaded.updates == agent.updates
    assert isinstance(load_agent(QLearningAgent().save(tmp_path / "table.npz")), QLearningAgent)
    save_replay(loaded, tmp_path / "replay.html")
    assert "shared features" in (tmp_path / "replay.html").read_text()


def test_feature_training_outputs(tmp_path):
    agent, summary = train_features(TrainConfig(steps=40, alpha=.01, gamma=.95, max_episode_steps=10), tmp_path / "run")
    assert summary["environment_transitions"] == agent.updates == 40
    assert summary["completed_episodes"] == 4
    assert summary["partial_episode"] is None
    assert (tmp_path / "run/checkpoint.npz").exists()
    config = json.loads((tmp_path / "run/config.json").read_text())
    assert config["reward_scale"] == 128


def test_fresh_evaluation_can_run_in_a_separate_process(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from rl2048.compare_variations import test_checkpoint

    QLearningAgent().save(tmp_path / "baseline_s0/checkpoint.npz")
    with ProcessPoolExecutor(max_workers=1) as pool:
        name, summary = pool.submit(test_checkpoint, (tmp_path, "baseline", 0)).result(timeout=30)
    assert name == "baseline" and summary["episodes"] == 200
    assert summary["seeds"][0] == 4_000_000 and summary["seeds"][-1] == 4_000_199
    assert (tmp_path / "baseline_s0/test/summary.json").exists()
