"""The complete tabular Q-learning loop. Read train() from top to bottom."""

import argparse
import csv
from dataclasses import asdict, dataclass
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time

import gymnasium as gym

from rl2048.agents.q_learning import QLearningAgent
from rl2048.game import Game2048


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 100_000
    seed: int = 0
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    decay_steps: int = 100_000
    max_episode_steps: int = 10_000

    def __post_init__(self):
        if self.steps < 1 or self.decay_steps < 1 or self.max_episode_steps < 1:
            raise ValueError("Step counts must be positive.")
        if not 0 <= self.seed < 1_000_000:
            raise ValueError("Training seeds must be in [0, 1_000_000).")
        if not 0 < self.alpha <= 1 or not 0 <= self.gamma <= 1:
            raise ValueError("Require 0 < alpha <= 1 and 0 <= gamma <= 1.")
        if not 0 <= self.epsilon_start <= 1 or not 0 <= self.epsilon_end <= 1:
            raise ValueError("Exploration probabilities must be in [0, 1].")


def epsilon_at(step: int, config: TrainConfig) -> float:
    """step is the number of transitions already collected, starting at zero."""
    fraction = min(max(step, 0) / max(1, config.decay_steps - 1), 1.0)
    return config.epsilon_start + fraction * (config.epsilon_end - config.epsilon_start)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def train(config: TrainConfig, output_dir: str | Path, *, verbose: bool = True) -> tuple[QLearningAgent, dict]:
    """Train, save an inspectable run, and return the agent and summary.

    The transition budget can end mid-game. That partial game is kept separate
    from completed episode statistics. Ctrl-C saves progress collected so far.
    A checkpoint restores the learner; it is not an exact in-game resume.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists():
        raise FileExistsError(f"Run already exists: {output_dir}. Choose a new --out directory.")
    write_json(output_dir / "config.json", asdict(config))
    write_json(output_dir / "runtime.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": "cpu", "packages": {name: version(name) for name in ("numpy", "gymnasium", "torch")},
    })
    env = gym.wrappers.TimeLimit(Game2048(), max_episode_steps=config.max_episode_steps)
    agent = QLearningAgent(alpha=config.alpha, gamma=config.gamma, seed=config.seed)
    board, info = env.reset(seed=config.seed)
    episodes = []
    coverage = []
    episode_length = 0
    interrupted = False
    start = time.perf_counter()

    try:
        for step in range(config.steps):
            # 1. Choose an action using the CURRENT board and its legal moves.
            epsilon = epsilon_at(step, config)
            action = agent.act(board, info["action_mask"], epsilon)

            # 2. The environment returns a transition (s, a, r, s').
            next_board, reward, terminated, truncated, next_info = env.step(action)

            # 3. Learn before resetting. Only natural termination removes
            #    the bootstrap; a time limit does not make future value zero.
            agent.update(board, action, reward, next_board, terminated,
                         next_info["action_mask"])
            episode_length += 1
            board, info = next_board, next_info

            # 4. Finish the episode, or carry the next state into the next step.
            if terminated or truncated:
                episodes.append({
                    "episode": len(episodes) + 1, "transitions": step + 1,
                    "score": info["score"], "length": episode_length,
                    "max_tile": info["max_tile"], "terminated": terminated,
                    "truncated": truncated, "epsilon": epsilon,
                    "elapsed_seconds": time.perf_counter() - start,
                })
                board, info = env.reset()
                episode_length = 0

            if (step + 1) % 1000 == 0 or step + 1 == config.steps:
                coverage.append({"transitions": step + 1, "epsilon": epsilon,
                                 **agent.coverage()})
            if verbose and ((step + 1) % 10_000 == 0 or step + 1 == config.steps):
                elapsed = time.perf_counter() - start
                print(f"{step + 1:,}/{config.steps:,} transitions | "
                      f"{len(episodes)} episodes | {len(agent.q):,} states | "
                      f"epsilon={epsilon:.3f} | {elapsed:.1f}s", flush=True)
    except KeyboardInterrupt:
        interrupted = True
        if verbose:
            print("Interrupted: saving completed updates.", flush=True)
    finally:
        env.close()

    elapsed = time.perf_counter() - start
    summary = {
        "algorithm": "tabular_q_learning", "environment_transitions": agent.updates,
        "completed_episodes": len(episodes), "elapsed_seconds": elapsed,
        "transitions_per_second": agent.updates / max(elapsed, 1e-9),
        "interrupted": interrupted, "coverage": agent.coverage(),
        "partial_episode": {"length": episode_length, "score": info["score"],
                            "max_tile": info["max_tile"]} if episode_length else None,
    }
    agent.save(output_dir / "checkpoint.npz")
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "episodes.csv", episodes, [
        "episode", "transitions", "score", "length", "max_tile", "terminated",
        "truncated", "epsilon", "elapsed_seconds",
    ])
    write_csv(output_dir / "coverage.csv", coverage, [
        "transitions", "epsilon", "updates", "unique_states", "repeated_updates",
        "states_visited_more_than_once", "repeated_update_fraction",
    ])
    return agent, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/q_learning"))
    for field, default in asdict(TrainConfig()).items():
        parser.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)
    args = vars(parser.parse_args())
    output_dir = args.pop("out")
    _, summary = train(TrainConfig(**args), output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
