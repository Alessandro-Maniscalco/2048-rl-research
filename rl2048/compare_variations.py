"""Choose a setting on validation games, then test all three trained seeds.

Run after training the named variations. Outputs remain local under runs/.
The final test set is never used to choose the winning setting or replay seed.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from rl2048.agents.feature_q import FeatureQAgent, GreedyMergeAgent, load_agent
from rl2048.agents.random_agent import RandomAgent
from rl2048.evaluate import save_evaluation
from rl2048.train import write_json
from rl2048.variations import TEST_SEEDS
from rl2048.view import save_replay

CANDIDATES = {
    "baseline": "Original table · 100k",
    "high_alpha": "Table · learning rate 0.5",
    "fast_exploration": "Table · faster exploration decay",
    "long_table": "Table · 500k transitions",
    "features": "Shared features · 100k",
    "features_low_alpha": "Shared features · learning rate 0.001",
    "features_myopic": "Shared features · gamma zero",
    "features_short": "Shared features · 20k",
    "features_fast": "Shared features · 100k, faster decay",
}


def aggregate(summaries):
    means = [s["mean_score"] for s in summaries]
    return {"mean_score": float(np.mean(means)),
            "training_seed_sd": float(np.std(means, ddof=1)) if len(means) > 1 else None,
            "seed_scores": means,
            "games": sum(s["episodes"] for s in summaries),
            "tile_reaching_rates": {key: float(np.mean([s["tile_reaching_rates"][key] for s in summaries]))
                                    for key in summaries[0]["tile_reaching_rates"]}}


def test_checkpoint(item):
    """Separate processes let CPU-bound game loops use separate Mac cores."""
    base, name, seed = item
    run = Path(base) / f"{name}_s{seed}"
    summary = save_evaluation(load_agent(run / "checkpoint.npz"), run / "test", seeds=TEST_SEEDS,
                              checkpoint=str(run / "checkpoint.npz"))
    print(f"Test {name} seed {seed}: {summary['mean_score']:.1f}", flush=True)
    return name, summary


def compare(base):
    base = Path(base)
    validation = {}
    for name, label in CANDIDATES.items():
        paths = [base / f"{name}_s{seed}" for seed in range(3)]
        summaries = [json.loads((p / "validation/summary.json").read_text()) for p in paths]
        config = json.loads((paths[0] / "config.json").read_text())
        validation[name] = {"label": label, "config": config, **aggregate(summaries)}
    winner = max(validation, key=lambda name: validation[name]["mean_score"])
    # Use the middle validation performer for the demonstration, not the best
    # individual seed. All three seeds count equally in the final test result.
    demo_seed = int(np.argsort(validation[winner]["seed_scores"])[1])
    selection = {"winner": winner, "demo_training_seed": demo_seed,
                 "validation": validation, "training_seeds": [0, 1, 2],
                 "validation_game_seeds": [3_000_000, 3_000_099],
                 "final_test_game_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1]]}
    write_json(base / "selection.json", selection)
    print(f"Selected on validation: {winner}; demo uses training seed {demo_seed}", flush=True)

    selected_names = list(dict.fromkeys(["baseline", winner]))
    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(test_checkpoint, [(base, name, seed) for name in selected_names for seed in range(3)]))
    test = {name: {"label": CANDIDATES[name], **aggregate([s for n, s in results if n == name])}
            for name in selected_names}
    for name, agent, label in [("random", RandomAgent(), "Random legal moves"),
                                ("greedy_merge", GreedyMergeAgent(), "Biggest immediate merge · no learning"),
                                ("untrained_features", FeatureQAgent(), "Shared features · zero weights")]:
        summary = save_evaluation(agent, base / "controls" / name, seeds=TEST_SEEDS)
        test[name] = {"label": label, **aggregate([summary])}
    report = {**selection, "test": test}
    write_json(base / "comparison.json", report)
    demo = load_agent(base / f"{winner}_s{demo_seed}" / "checkpoint.npz")
    save_replay(demo, base / "winner_replay.html", seed=2_000_000)
    plot_report(report, base / "comparison.png")
    return report


def plot_report(report, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    for axis, split, title in zip(axes, ["validation", "test"],
                                 ["Validation · select a setting", "Fresh test games · frozen settings"], strict=True):
        rows = sorted(report[split].items(), key=lambda item: item[1]["mean_score"])
        values = [row["mean_score"] for _, row in rows]
        errors = [row["training_seed_sd"] or 0 for _, row in rows]
        axis.barh([row["label"] for _, row in rows], values, xerr=errors,
                  color=["#197970" if name.startswith("features") else "#b8a68e" for name, _ in rows],
                  capsize=3)
        axis.set(title=title, xlabel="Mean raw game score")
        axis.tick_params(axis="y", labelsize=9)
        axis.grid(axis="x", alpha=.15)
        axis.set_axisbelow(True)
    fig.suptitle("Q-learning variations · error bars show SD across three training seeds\nFixed controls have one 200-game test, without a training-seed error bar", fontsize=12)
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("runs/variations"))
    compare(parser.parse_args().directory)
