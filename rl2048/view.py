"""Notebook game controls, portable replays, and learning curves.

Rendering consumes game state; it never changes the RL update or reward.
"""

import argparse
import csv
from html import escape
import json
from pathlib import Path
import webbrowser

import numpy as np

from rl2048.agents.q_learning import QLearningAgent
from rl2048.agents.random_agent import RandomAgent
from rl2048.agents.feature_q import load_agent
from rl2048.evaluate import rollout
from rl2048.game import ACTION_NAMES, Game2048
from rl2048.train import write_json

COLORS = {
    0: "#cdc1b4", 2: "#eee4da", 4: "#ede0c8", 8: "#f2b179",
    16: "#f59563", 32: "#f67c5f", 64: "#f65e3b", 128: "#edcf72",
    256: "#edcc61", 512: "#edc850", 1024: "#edc53f", 2048: "#edc22e",
}


def board_html(board: np.ndarray, score: int = 0, caption: str = "") -> str:
    tiles = []
    for value in board.flat:
        tile = int(value)
        color = COLORS.get(tile, "#3c3a32")
        text_color = "#776e65" if tile < 8 else "#f9f6f2"
        label = str(tile) if tile else ""
        tiles.append(f'<div style="background:{color};color:{text_color};height:72px;'
                     f'display:flex;align-items:center;justify-content:center;border-radius:5px;'
                     f'font-size:24px;font-weight:700">{label}</div>')
    return ('<div style="font-family:system-ui;color:#776e65;max-width:360px">'
            f'<p><b>Score: {int(score):,}</b> · {escape(caption)}</p>'
            '<div style="display:grid;grid-template-columns:repeat(4,minmax(0,1fr));grid-template-rows:repeat(4,72px);gap:8px;'
            'background:#bbada0;padding:8px;border-radius:8px">'
            + "".join(tiles) + '</div></div>')


def play(seed: int = 0):
    """Return an ipywidgets panel. Display it as the last expression in a cell."""
    import ipywidgets as widgets

    env = Game2048()
    board, info = env.reset(seed=seed)
    screen = widgets.HTML()
    buttons = [widgets.Button(description=name.title(), layout=widgets.Layout(width="80px"))
               for name in ACTION_NAMES]

    def refresh(board, info, message="Choose a direction"):
        screen.value = board_html(board, info["score"], message)
        for index, button in enumerate(buttons):
            button.disabled = not bool(info["action_mask"][index])

    def on_move(action):
        board, reward, terminated, _, info = env.step(action)
        message = "Game over" if terminated else f"{ACTION_NAMES[action]} · reward +{int(reward)}"
        refresh(board, info, message)

    for action, button in enumerate(buttons):
        button.on_click(lambda _, action=action: on_move(action))
    reset = widgets.Button(description="New game", icon="refresh")

    def on_reset(_):
        board, info = env.reset()
        refresh(board, info)

    reset.on_click(on_reset)
    refresh(board, info)
    return widgets.VBox([screen, widgets.HBox(buttons), reset])


def replay_widget(frames: list[dict]):
    """Play or scrub through actual recorded frames; no external video codec."""
    import ipywidgets as widgets

    if not frames:
        raise ValueError("A replay needs at least one frame.")
    screen = widgets.HTML()
    slider = widgets.IntSlider(min=0, max=len(frames) - 1, description="Move")
    player = widgets.Play(min=0, max=len(frames) - 1, interval=100)
    widgets.jslink((player, "value"), (slider, "value"))

    def draw(change=None):
        frame = frames[slider.value]
        action = frame["action"]
        caption = "Initial board" if action is None else f"{ACTION_NAMES[action]} · reward +{int(frame['reward'])}"
        screen.value = board_html(np.array(frame["board"], dtype=np.int64), frame["score"], caption)

    slider.observe(draw, names="value")
    draw()
    return widgets.VBox([screen, widgets.HBox([player, slider])])


def save_replay(agent, path: str | Path, *, seed: int = 2_000_000, max_steps: int = 40_000) -> dict:
    """Save both raw replay.json and a standalone HTML viewer next to it."""
    path = Path(path)
    result, frames = rollout(agent, seed=seed, max_steps=max_steps, record=True)
    data = {"algorithm": getattr(agent, "display_name", "Q-learning" if isinstance(agent, QLearningAgent) else "Random baseline"),
            "result": result, "frames": frames, "colors": COLORS, "actions": ACTION_NAMES}
    write_json(path.with_suffix(".json"), data)
    template = Path(__file__).with_name("replay.html").read_text()
    path.write_text(template.replace("__REPLAY_DATA__", json.dumps(data).replace("<", "\\u003c")))
    return result


def plot_training(run_dir: str | Path):
    """Completed-episode scores, unique-board growth, and exploration."""
    import matplotlib.pyplot as plt

    run_dir = Path(run_dir)
    with (run_dir / "episodes.csv").open() as file:
        episodes = list(csv.DictReader(file))
    with (run_dir / "coverage.csv").open() as file:
        coverage = list(csv.DictReader(file))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), layout="constrained")
    x = np.array([int(row["transitions"]) for row in episodes])
    scores = np.array([float(row["score"]) for row in episodes])
    if len(scores):
        axes[0].plot(x, scores, alpha=0.25, color="#197970", label="Episode score")
        width = min(20, len(scores))
        axes[0].plot(x[width - 1:], np.convolve(scores, np.ones(width) / width, mode="valid"),
                     color="#197970", label=f"{width}-episode mean")
        axes[0].legend(fontsize=8)
    else:
        axes[0].text(0.5, 0.5, "No completed episodes yet", ha="center", transform=axes[0].transAxes)
    steps = [int(row["transitions"]) for row in coverage]
    axes[1].plot(steps, [int(row["unique_states"]) for row in coverage], label="Distinct boards")
    axes[1].plot(steps, steps, "--", color="gray", label="Every update is new")
    axes[1].legend(fontsize=8)
    axes[2].plot(steps, [float(row["epsilon"]) for row in coverage], color="#be7431")
    axes[2].set_ylim(0, 1.05)
    for axis, title, ylabel in zip(axes, ["Training scores", "State coverage", "Exploration"],
                                   ["Raw score", "Distinct boards", "Epsilon"], strict=True):
        axis.set(title=title, xlabel="Environment transitions", ylabel=ylabel)
        axis.grid(alpha=0.15)
    return fig


def plot_comparison(summaries: dict[str, dict]):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    names = list(summaries)
    axes[0].bar(names, [summaries[name]["mean_score"] for name in names], color=["#bbada0", "#197970", "#be7431"][:len(names)])
    axes[0].set(title="Held-out evaluation", ylabel="Mean raw score")
    thresholds = ["128", "256", "512", "1024", "2048"]
    for name, summary in summaries.items():
        axes[1].plot(thresholds, [summary["tile_reaching_rates"][tile] for tile in thresholds], "o-", label=name)
    axes[1].set(title="Tile-reaching rates", xlabel="Tile", ylabel="Fraction of games", ylim=(0, 1))
    axes[1].legend()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Record a frozen agent and save an interactive HTML replay.")
    parser.add_argument("--checkpoint", type=Path, help="Omit to watch a random agent.")
    parser.add_argument("--out", type=Path, default=Path("runs/random_replay.html"))
    parser.add_argument("--seed", type=int, default=2_000_000)
    parser.add_argument("--open", action="store_true", help="Open the saved viewer in your browser.")
    args = parser.parse_args()
    agent = load_agent(args.checkpoint) if args.checkpoint else RandomAgent()
    result = save_replay(agent, args.out, seed=args.seed)
    print(json.dumps(result, indent=2))
    print(f"Replay: {args.out.resolve()}")
    if args.open:
        webbrowser.open(args.out.resolve().as_uri())


if __name__ == "__main__":
    main()


def q_inspector(checkpoint, seed=0):
    """Play while inspecting the actual 16 numeric inputs and four Q outputs."""
    import ipywidgets as widgets
    import torch
    from rl2048.agents.ntuple import encode

    agent = load_agent(checkpoint)
    if not hasattr(agent, 'policy') or not hasattr(agent.policy, 'encode_inputs'):
        raise ValueError('Choose a scalar DQN checkpoint for this inspector.')
    env = Game2048()
    board, info = env.reset(seed=seed)
    screen, numbers = widgets.HTML(), widgets.HTML()
    moves = [widgets.Button(description=name.title(), layout=widgets.Layout(width='80px'))
             for name in ACTION_NAMES]
    take_best = widgets.Button(description='Agent move', button_style='success')
    reset = widgets.Button(description='New game')
    selected = None

    def refresh(board, info, caption='Frozen Q-network'):
        nonlocal selected
        state = torch.tensor(encode(board)[None])
        with torch.no_grad():
            inputs = agent.policy.encode_inputs(state)[0].cpu().numpy()
            values = agent.policy(state)[0].cpu().numpy()
        mask = info['action_mask']
        selected = int(np.where(mask, values, -np.inf).argmax()) if mask.any() else None
        screen.value = board_html(board, info['score'], caption)
        input_rows = ''.join('<tr>' + ''.join(f'<td style="padding:6px">{value:.5g}</td>' for value in row) + '</tr>'
                             for row in inputs.reshape(4,4))
        q_rows = ''.join(f'<tr><td>{name}</td><td style="padding:6px">{value:.5f}</td>'
                         f'<td>{"← chosen" if index == selected else "" if mask[index] else "invalid"}</td></tr>'
                         for index, (name, value) in enumerate(zip(ACTION_NAMES, values)))
        numbers.value = (f'<div style="font:14px system-ui;padding:12px"><b>16 inputs · {agent.policy.input_encoding}</b>'
                         f'<table style="font-family:monospace">{input_rows}</table><br>'
                         f'<b>4 Q-values</b><table>{q_rows}</table>'
                         '<p>Q-values use the learning reward and discount recorded in this checkpoint.</p></div>')
        for index, button in enumerate(moves):
            button.disabled = not bool(mask[index])
        take_best.disabled = selected is None

    def step(action):
        board, reward, terminal, _, info = env.step(action)
        refresh(board, info, 'Game over' if terminal else f'{ACTION_NAMES[action]} · merge reward {int(reward)}')

    for action, button in enumerate(moves):
        button.on_click(lambda _, action=action: step(action))
    take_best.on_click(lambda _: step(selected))
    reset.on_click(lambda _: refresh(*env.reset()))
    refresh(board, info)
    return widgets.VBox([widgets.HBox([screen, numbers]), widgets.HBox(moves), widgets.HBox([take_best, reset])])
