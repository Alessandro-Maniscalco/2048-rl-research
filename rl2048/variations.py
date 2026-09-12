"""Controlled Q-learning experiments. Every run has a checkpoint and raw scores.

Use --kind table for parameter changes to the first lesson. --kind features
uses linear Q-learning with engineered board features; its explicit loop stays
here so the first lesson's training loop remains unchanged.
"""

import argparse
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
import platform
import time

import gymnasium as gym

from rl2048.agents.feature_q import FEATURE_NAMES, FeatureQAgent
from rl2048.evaluate import save_evaluation
from rl2048.game import Game2048
from rl2048.train import TrainConfig, epsilon_at, train, write_csv, write_json

VALIDATION_SEEDS = tuple(range(3_000_000, 3_000_100))
TEST_SEEDS = tuple(range(4_000_000, 4_000_200))


def train_features(config, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if (output_dir / "config.json").exists():
        raise FileExistsError(f"Run already exists: {output_dir}")
    write_json(output_dir / "config.json", {**asdict(config), "representation": "linear_features",
                                           "reward_scale": 128, "feature_names": FEATURE_NAMES})
    write_json(output_dir / "runtime.json", {
        "python": platform.python_version(), "platform": platform.platform(),
        "device": "cpu", "packages": {name: version(name) for name in ("numpy", "gymnasium")},
    })
    env = gym.wrappers.TimeLimit(Game2048(), max_episode_steps=config.max_episode_steps)
    agent = FeatureQAgent(config.alpha, config.gamma, config.seed)
    board, info = env.reset(seed=config.seed)
    rows = []
    start = time.perf_counter()
    for step in range(config.steps):
        action = agent.act(board, info["action_mask"], epsilon_at(step, config))
        next_board, reward, terminated, truncated, next_info = env.step(action)
        agent.update(board, action, reward, next_board, terminated, next_info["action_mask"])
        board, info = next_board, next_info
        if terminated or truncated:
            rows.append({"episode": len(rows) + 1, "transitions": step + 1,
                         "score": info["score"], "length": info["steps"], "max_tile": info["max_tile"],
                         "terminated": terminated, "truncated": truncated})
            board, info = env.reset()
        if (step + 1) % 10_000 == 0:
            print(f"{step+1:,}/{config.steps:,} | {len(rows)} games | {time.perf_counter()-start:.1f}s", flush=True)
    env.close()
    summary = {"algorithm": agent.name, "environment_transitions": agent.updates,
               "completed_episodes": len(rows), "elapsed_seconds": time.perf_counter() - start,
               "weights": dict(zip(FEATURE_NAMES, agent.weights.tolist())),
               "partial_episode": {"length": info["steps"], "score": info["score"]} if info["steps"] else None}
    agent.save(output_dir / "checkpoint.npz")
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "episodes.csv", rows,
              ["episode", "transitions", "score", "length", "max_tile", "terminated", "truncated"])
    return agent, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["table", "features"], required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, help="Default: 0.1 for table, 0.01 for features.")
    parser.add_argument("--gamma", type=float, help="Default: 0.99 for table, 0.95 for features.")
    parser.add_argument("--epsilon-end", type=float, default=0.1)
    parser.add_argument("--decay-steps", type=int, default=100_000)
    args = parser.parse_args()
    if args.alpha is None:
        args.alpha = 0.1 if args.kind == "table" else 0.01
    if args.gamma is None:
        args.gamma = 0.99 if args.kind == "table" else 0.95
    config = TrainConfig(steps=args.steps, seed=args.seed, alpha=args.alpha, gamma=args.gamma,
                         epsilon_end=args.epsilon_end, decay_steps=args.decay_steps)
    agent, _ = (train if args.kind == "table" else train_features)(config, args.out)
    summary = save_evaluation(agent, args.out / "validation", seeds=VALIDATION_SEEDS,
                              checkpoint=str(args.out / "checkpoint.npz"))
    print(f"Validation: score {summary['mean_score']:.1f}, >=512 {summary['tile_reaching_rates']['512']:.0%}", flush=True)


if __name__ == "__main__":
    main()
