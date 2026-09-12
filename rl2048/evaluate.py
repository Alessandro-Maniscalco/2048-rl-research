"""Evaluate frozen agents on the same 100 held-out game seeds."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import gymnasium as gym
import numpy as np

from rl2048.agents.q_learning import QLearningAgent, state_key
from rl2048.agents.random_agent import RandomAgent
from rl2048.agents.feature_q import load_agent
from rl2048.game import Game2048
from rl2048.train import write_csv, write_json

EVALUATION_SEEDS = tuple(range(1_000_000, 1_000_100))
TILE_THRESHOLDS = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536)


def rollout(agent, *, seed: int, max_steps: int = 10_000, record: bool = False) -> tuple[dict, list[dict]]:
    """One frozen-policy game. Both learning agents default to epsilon=0.

    Seed action tie-breaking separately from tile spawning. No update() calls
    occur here. Evaluation does not add unseen boards to the Q table.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive.")
    env = gym.wrappers.TimeLimit(Game2048(), max_episode_steps=max_steps)
    previous_rng = agent.rng
    agent.rng = np.random.default_rng(seed + 10_000_000)
    board, info = env.reset(seed=seed)
    frames = []
    if record:
        frames.append({"board": board.tolist(), "score": 0, "action": None, "reward": 0})
    seen = 0
    start = time.perf_counter()
    try:
        while True:
            if isinstance(agent, QLearningAgent) and state_key(board) in agent.q:
                seen += 1
            action = agent.act(board, info["action_mask"])
            board, reward, terminated, truncated, info = env.step(action)
            if record:
                frames.append({"board": board.tolist(), "score": info["score"],
                               "action": action, "reward": reward})
            if terminated or truncated:
                break
    finally:
        agent.rng = previous_rng
        env.close()
    return {
        "seed": seed, "score": info["score"], "length": info["steps"],
        "max_tile": info["max_tile"], "terminated": terminated, "truncated": truncated,
        "elapsed_seconds": time.perf_counter() - start,
        "known_state_fraction": seen / info["steps"] if isinstance(agent, QLearningAgent) else None,
    }, frames


def evaluate(agent, seeds=EVALUATION_SEEDS, *, max_steps: int = 10_000) -> tuple[dict, list[dict]]:
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds:
        raise ValueError("Evaluation needs at least one seed.")
    start = time.perf_counter()
    rows = [rollout(agent, seed=seed, max_steps=max_steps)[0] for seed in seeds]
    scores = np.array([row["score"] for row in rows])
    lengths = np.array([row["length"] for row in rows])
    max_tiles = np.array([row["max_tile"] for row in rows])
    summary = {
        "algorithm": getattr(agent, "algorithm", getattr(agent, "name", "tabular_q_learning" if isinstance(agent, QLearningAgent) else "random")),
        "episodes": len(rows), "seeds": list(seeds), "max_episode_steps": max_steps,
        "mean_score": float(scores.mean()), "median_score": float(np.median(scores)),
        "score_std": float(scores.std()), "mean_episode_length": float(lengths.mean()),
        "max_tile_distribution": dict(sorted(Counter(int(tile) for tile in max_tiles).items())),
        "tile_reaching_rates": {str(tile): float((max_tiles >= tile).mean()) for tile in TILE_THRESHOLDS},
        "environment_transitions": int(lengths.sum()), "elapsed_seconds": time.perf_counter() - start,
        "truncated_episodes": sum(row["truncated"] for row in rows),
        "known_state_fraction": (
            sum(row["known_state_fraction"] * row["length"] for row in rows) / int(lengths.sum())
            if isinstance(agent, QLearningAgent) else None
        ),
    }
    return summary, rows


def save_evaluation(agent, output_dir: str | Path, *, seeds=EVALUATION_SEEDS,
                    max_steps: int = 10_000, checkpoint: str | None = None) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary, rows = evaluate(agent, seeds, max_steps=max_steps)
    summary["checkpoint"] = checkpoint
    write_json(output_dir / "summary.json", summary)
    write_csv(output_dir / "episodes.csv", rows, list(rows[0]))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, help="Omit to evaluate the random baseline.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--seed-start", type=int, default=EVALUATION_SEEDS[0])
    parser.add_argument("--episodes", type=int, default=len(EVALUATION_SEEDS))
    args = parser.parse_args()
    agent = load_agent(args.checkpoint) if args.checkpoint else RandomAgent()
    summary = save_evaluation(agent, args.out, max_steps=args.max_steps,
                              seeds=range(args.seed_start, args.seed_start + args.episodes),
                              checkpoint=str(args.checkpoint) if args.checkpoint else None)
    print(json.dumps({key: value for key, value in summary.items() if key != "seeds"}, indent=2))


if __name__ == "__main__":
    main()
