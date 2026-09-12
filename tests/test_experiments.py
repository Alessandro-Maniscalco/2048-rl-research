import json

import numpy as np
import pytest

from rl2048.agents.q_learning import QLearningAgent
from rl2048.agents.random_agent import RandomAgent
from rl2048.evaluate import evaluate, rollout
from rl2048.game import legal_actions, move
from rl2048.train import TrainConfig, train
from rl2048.view import board_html, play, replay_widget, save_replay


def test_smoke_checkpoint_evaluation_is_frozen_and_repeatable(tmp_path):
    agent, summary = train(TrainConfig(steps=200, max_episode_steps=20), tmp_path / "run", verbose=False)
    assert summary["environment_transitions"] == 200
    assert summary["completed_episodes"] == 10
    assert summary["partial_episode"] is None
    loaded = QLearningAgent.load(tmp_path / "run/checkpoint.npz")
    before = loaded.coverage()
    rng_before = loaded.rng.bit_generator.state
    result, rows = evaluate(loaded, [1_000_000, 1_000_001], max_steps=20)
    again, other_rows = evaluate(loaded, [1_000_000, 1_000_001], max_steps=20)
    assert before == loaded.coverage() and rng_before == loaded.rng.bit_generator.state
    assert result["environment_transitions"] == 40 and result["truncated_episodes"] == 2
    assert [r["score"] for r in rows] == [r["score"] for r in other_rows]
    assert result["mean_score"] == again["mean_score"]
    with pytest.raises(FileExistsError):
        train(TrainConfig(steps=1), tmp_path / "run", verbose=False)


def test_training_budget_does_not_invent_completed_episode(tmp_path):
    _, summary = train(TrainConfig(steps=1), tmp_path / "tiny", verbose=False)
    assert summary["completed_episodes"] == 0
    assert summary["partial_episode"]["length"] == 1


def test_complete_random_game_and_replay_transitions(tmp_path):
    result, frames = rollout(RandomAgent(), seed=2_000_000, record=True)
    assert result["terminated"] and not result["truncated"]
    assert len(frames) == result["length"] + 1
    assert sum(frame["reward"] for frame in frames) == result["score"]
    assert not legal_actions(np.array(frames[-1]["board"])).any()
    for previous, current in zip(frames, frames[1:]):
        before = np.array(previous["board"])
        after = np.array(current["board"])
        shifted, reward, moved = move(before, current["action"])
        difference = after - shifted
        assert moved and reward == current["reward"]
        assert np.count_nonzero(difference) == 1
        assert difference[difference != 0][0] in (2, 4)
        assert np.all(shifted[difference != 0] == 0)
    path = tmp_path / "replay.html"
    save_replay(RandomAgent(), path)
    assert "__REPLAY_DATA__" not in path.read_text()
    data = json.loads(path.with_suffix(".json").read_text())
    assert data["result"]["score"] == result["score"]


def test_notebook_controls_and_labels():
    board = np.zeros((4, 4), dtype=np.int64)
    board[0] = [0, 64, 128, 8192]
    html = board_html(board)
    assert ">64</div>" in html and ">128</div>" in html and ">8192</div>" in html
    assert ">0</div>" not in html and "-128" not in html
    panel = play(seed=0)
    screen, directions, reset = panel.children
    before = screen.value
    next(button for button in directions.children if not button.disabled).click()
    assert screen.value != before and "reward" in screen.value
    reset.click()
    assert "Score: 0" in screen.value
    frames = [{"board": board.tolist(), "score": 0, "action": None, "reward": 0},
              {"board": board.tolist(), "score": 4, "action": 1, "reward": 4}]
    replay = replay_widget(frames)
    replay.children[1].children[1].value = 1
    assert "right" in replay.children[0].value and "Score: 4" in replay.children[0].value
