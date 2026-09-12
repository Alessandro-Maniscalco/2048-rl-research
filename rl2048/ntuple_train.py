"""Time-bounded optimistic afterstate TD, then temporal-coherence fine-tuning.

Warm up with one worker to reduce contention on common initial patterns.
Afterward, compiled threads update sparse shared weights asynchronously
(Hogwild). This is intentionally nondeterministic for workers > 1.
Checkpoint and evaluation happen only after all workers finish their batches.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import time

import numpy as np

from rl2048.agents.ntuple import NTupleAgent, row_tables
from rl2048.fast2048 import train_batch, evaluate_game
from rl2048.train import write_json


def evaluate_fast(agent, seeds, *, depth=None, workers=8, cutoff=None):
    depth = agent.depth if depth is None else depth
    cutoff = agent.cutoff if cutoff is None else cutoff
    row_moves, row_rewards = row_tables()
    seeds = list(seeds)
    start = time.perf_counter()
    # Warm the compiled specialization before calling concurrently.
    if seeds:
        first = evaluate_game(agent.weights, agent.patterns, row_moves, row_rewards,
                              int(seeds[0]), depth, cutoff, 40000, agent.downgrade_threshold)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(evaluate_game, agent.weights, agent.patterns, row_moves, row_rewards,
                               int(seed), depth, cutoff, 40000, agent.downgrade_threshold) for seed in seeds[1:]]
        outcomes = [first] + [future.result() for future in futures] if seeds else []
    records = [{"seed": seed, "score": int(row[0]), "length": int(row[1]),
                "max_tile": int(row[2]), "truncated": bool(row[3])}
               for seed, row in zip(seeds, outcomes, strict=True)]
    scores = np.array([row["score"] for row in records])
    summary = {
        "games": len(records), "mean_score": float(scores.mean()), "median_score": float(np.median(scores)),
        "score_sd": float(scores.std()), "max_score": int(scores.max()),
        "max_tile": max(row["max_tile"] for row in records),
        "reaching_rates": {str(tile): sum(row["max_tile"] >= tile for row in records) / len(records)
                           for tile in (2048, 4096, 8192, 16384, 32768, 65536)},
        "transitions": sum(row["length"] for row in records), "truncated": sum(row["truncated"] for row in records),
        "seconds": time.perf_counter() - start, "depth": depth, "cutoff": cutoff,
        "downgrade_threshold": agent.downgrade_threshold,
        "seeds": seeds, "rng": "Numba MT19937; different stream from Game2048's NumPy Generator",
    }
    return summary, records


def run(args):
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError(f"Choose a new output directory: {out}")
    write_json(out / "config.json", {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()})
    agent = NTupleAgent.load(args.resume) if args.resume else NTupleAgent(args.layout, args.initial_value, args.seed)
    agent.metadata = agent.metadata | {"method": "backward OTD+TC", "workers": args.workers, "seed": args.seed,
                      "resume_checkpoint": str(args.resume) if args.resume else None,
                      "prior_training_games": agent.training_games,
                      "alpha_otd": args.alpha, "alpha_tc": args.tc_alpha, "tc_fraction": args.tc_fraction,
                      "normalize_collisions": args.normalize_collisions,
                      "parallel_updates": "Hogwild; no bitwise reproducibility guarantee"}
    rows, rewards = row_tables()
    dummy = np.zeros((1, 1), np.float32)
    tc_sum, tc_abs = dummy, dummy
    # Compile outside the timed optimization budget, with a separate tiny model.
    tiny = NTupleAgent("4x4")
    train_batch(tiny.weights, tiny.patterns, rows, rewards, 1, 0, .1, dummy, dummy, False,
                normalize_collisions=args.normalize_collisions)
    del tiny
    start = time.perf_counter()
    end = start + args.seconds
    next_report = start
    best_score = -1
    records = []
    block = 0
    use_tc = False
    training_seconds = 0.
    new_moves = 0
    new_games = 0
    with (out / "progress.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["seconds", "games", "transitions", "phase", "validation_score", "train_moves_per_second"])
        writer.writeheader()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            while time.perf_counter() < end:
                if not use_tc and args.tc_fraction > 0 and time.perf_counter() - start >= args.seconds * (1 - args.tc_fraction):
                    use_tc = True
                    tc_sum = np.zeros_like(agent.weights)
                    tc_abs = np.zeros_like(agent.weights)
                    print(f"Switching to temporal coherence (alpha={args.tc_alpha}, adaptive per-weight rates)", flush=True)
                # New networks start serially; transferred networks are warmed already.
                active_workers = 1 if agent.training_games < args.warmup_games else args.workers
                batch_start = time.perf_counter()
                futures = [pool.submit(train_batch, agent.weights, agent.patterns, rows, rewards,
                                       args.batch_games, args.seed + block * 100 + worker,
                                       args.tc_alpha if use_tc else args.alpha, tc_sum, tc_abs, use_tc,
                                       normalize_collisions=args.normalize_collisions)
                           for worker in range(active_workers)]
                batches = [f.result() for f in futures]
                training_seconds += time.perf_counter() - batch_start
                block += 1
                for batch in batches:
                    agent.training_games += len(batch)
                    agent.training_transitions += int(batch[:, 1].sum())
                    new_games += len(batch)
                    new_moves += int(batch[:, 1].sum())
                now = time.perf_counter()
                if now >= next_report or now >= end:
                    validation, _ = evaluate_fast(agent, range(6_000_000, 6_000_020), depth=1, workers=args.workers)
                    entry = {"seconds": time.perf_counter() - start, "games": agent.training_games,
                             "transitions": agent.training_transitions, "phase": "TC" if use_tc else "OTD",
                             "validation_score": validation["mean_score"],
                             "train_moves_per_second": new_moves / max(training_seconds, 1e-9)}
                    writer.writerow(entry)
                    file.flush()
                    records.append(entry)
                    print(json.dumps(entry), flush=True)
                    if entry["validation_score"] > best_score:
                        best_score = entry["validation_score"]
                        agent.save(out / "best")
                    next_report = now + args.report_every
    agent.save(out / "last")
    if use_tc:
        np.save(out / "last/tc_sum.npy", tc_sum)
        np.save(out / "last/tc_abs.npy", tc_abs)
    summary = {"elapsed_seconds": time.perf_counter() - start, "new_games": new_games,
               "new_transitions": new_moves, "training_seconds": training_seconds,
               "best_validation_score": best_score, "progress": records}
    write_json(out / "summary.json", summary)
    print("Completed", out, best_score, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--layout", choices=["4x4", "4x6", "8x6"], default="4x6")
    parser.add_argument("--initial-value", type=float, default=80000)
    parser.add_argument("--alpha", type=float, default=.1)
    parser.add_argument("--tc-alpha", type=float, default=1., help="Base learning rate multiplied by temporal coherence")
    parser.add_argument("--normalize-collisions", action="store_true",
                        help="Normalize updates by squared feature norm, accounting for repeated symmetric weights")
    parser.add_argument("--tc-fraction", type=float, default=.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup-games", type=int, default=2000)
    parser.add_argument("--batch-games", type=int, default=100)
    parser.add_argument("--report-every", type=float, default=45)
    args = parser.parse_args()
    if args.seconds <= 0 or args.workers < 1 or args.batch_games < 1 or not 0 <= args.tc_fraction <= 1 or not 0 < args.tc_alpha <= 1:
        parser.error("Invalid duration, worker count, batch size or TC fraction.")
    run(args)


if __name__ == "__main__":
    main()
